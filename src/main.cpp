#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <Eigen/Dense>

#define PI 3.141592653589793238

using namespace Eigen;
using namespace std;

struct imageData {
    std::string imageName = "";
    double latitude = 0;
    double longitude = 0;
    double altitudeFeet = 0;
    double altitudeMeter = 0;
    double roll = 0;
    double pitch = 0;
    double yaw = 0;
};

cv::Mat computeUnRotMatrix (imageData& pose) {
    float a = pose.yaw * PI / 180;
    float b = pose.roll * PI / 180;
    float g = pose.pitch * PI / 180;
    Matrix3f Rz;
    Rz << cos (a), -sin (a), 0,
          sin (a), cos (a), 0,
          0, 0, 1;
    Matrix3f Ry;
    Ry << cos (b), 0, sin (b),
          0, 1, 0,
          -sin (b), 0, cos (b);
    Matrix3f Rx;
    Rx << 1, 0, 0,
          0, cos (g), -sin (g),
          0, sin (g), cos (g);
    Matrix3f R = Rz * (Rx * Ry);
    R(0,2) = 0;
    R(1,2) = 0;
    R(2,2) = 1;
    Matrix3f Rtrans = R.transpose ();
    Matrix3f InvR = Rtrans.inverse ();
    cv::Mat transformation = (cv::Mat_<double>(3,3) << InvR(0,0), InvR(0,1), InvR(0,2),
                                                       InvR(1,0), InvR(1,1), InvR(1,2),
                                                       InvR(2,0), InvR(2,1), InvR(2,2));
    return transformation;
}

void printMat (cv::Mat& mat) {
    int rows = mat.rows;
    int cols = mat.cols;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << mat.at<double>(i,j) << " ";
        }
        std::cout << std::endl;
    }
}

cv::Mat warpPerspectiveWithPadding (const cv::Mat& image, cv::Mat& transformation) {
    int height = image.rows;
    int width = image.cols;
    cv::Mat small_img;
    cv::resize (image, small_img, cv::Size (width/2, height/2));
    std::vector<cv::Point2f> corners = {cv::Point2f (0,0), cv::Point2f (0,height/2),
                                   cv::Point2f (width/2,height/2), cv::Point2f (width/2,0)};
    std::vector<cv::Point2f> warpedCorners;
    cv::perspectiveTransform (corners, warpedCorners, transformation);
    float xMin = 1e9, xMax = -1e9;
    float yMin = 1e9, yMax = -1e9;
    for (int i = 0; i < 4; i++) {
        xMin = (xMin > warpedCorners[i].x)? warpedCorners[i].x : xMin;
        xMax = (xMax < warpedCorners[i].x)? warpedCorners[i].x : xMax;
        yMin = (yMin > warpedCorners[i].y)? warpedCorners[i].y : yMin;
        yMax = (yMax < warpedCorners[i].y)? warpedCorners[i].y : yMax;
    }
    int xMin_ = (xMin - 0.5);
    int xMax_ = (xMax + 0.5);
    int yMin_ = (yMin - 0.5);
    int yMax_ = (yMax + 0.5);
    cv::Mat translation = (cv::Mat_<double>(3,3) << 1, 0, -xMin_, 0, 1, -yMin_, 0, 0, 1);
    cv::Mat fullTransformation = translation * transformation;
    cv::cuda::GpuMat result;
    cv::cuda::GpuMat gpu_img (small_img);
    cv::cuda::GpuMat gpu_ft (fullTransformation);
    cv::cuda::warpPerspective (gpu_img, result, fullTransformation,
                                cv::Size (xMax_-xMin_, yMax_-yMin_));
    cv::Mat result_ (result.size(), result.type());
    result.download (result_);
    return result_;
}

cv::Mat combinePair (cv::Mat& img1, cv::Mat& img2) {
    cv::cuda::GpuMat img1_gpu (img1), img2_gpu (img2);
    cv::cuda::GpuMat img1_gray_gpu, img2_gray_gpu;

    cv::cuda::cvtColor (img1_gpu, img1_gray_gpu, cv::COLOR_BGR2GRAY);
    cv::cuda::cvtColor (img2_gpu, img2_gray_gpu, cv::COLOR_BGR2GRAY);

    cv::cuda::GpuMat mask1;
    cv::cuda::GpuMat mask2;

    cv::cuda::threshold (img1_gray_gpu, mask1, 1, 255, cv::THRESH_BINARY);
    cv::cuda::threshold (img2_gray_gpu, mask2, 1, 255, cv::THRESH_BINARY);

    cv::Ptr<cv::cuda::SURF_CUDA> detector = cv::cuda::SURF_CUDA::create (1000);

    cv::cuda::GpuMat keypoints1_gpu, descriptors1_gpu;
    detector->detectWithDescriptors (img1_gray_gpu, mask1,
                                    keypoints1_gpu, descriptors1_gpu);
    std::vector<cv::KeyPoint> keypoints1;
    detector->uploadKeypoints (keypoints1, keypoints1_gpu);

    cv::cuda::GpuMat keypoints2_gpu, descriptors2_gpu;
    detector->detectWithDescriptors (img2_gray_gpu, mask2,
                                    keypoints2_gpu, descriptors2_gpu);
    std::vector<cv::KeyPoint> keypoints2;
    detector->uploadKeypoints (keypoints2, keypoints2_gpu);

    cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
        cv::cuda::DescriptorMatcher::createBFMatcher (cv::NORM_HAMMING);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch (descriptors2_gpu, descriptors1_gpu, knn_matches, 2);
    std::cout << "knn_matches=" << knn_matches.size() << std::endl;

    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>>::const_iterator it;
    for (it = knn_matches.begin(); it != knn_matches.end(); ++it) {
        if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.55) {
            matches.push_back((*it)[0]);
        }
    }

    std::vector<cv::Point2f> src_pts;
    std::vector<cv::Point2f> dst_pts;
    for (auto m : matches) {
        src_pts.push_back (keypoints2[m.queryIdx].pt);
        dst_pts.push_back (keypoints1[m.trainIdx].pt);
    }

    cv::Mat A = cv::estimateRigidTransform(src_pts, dst_pts, false);
    float height1 = img1.rows, width1 = img1.cols;
    float height2 = img2.rows, width2 = img2.cols;

    std::vector<std::vector<float>> corners1 {{0,0},{0,height1},{width1,height1},{width1,0}};
    std::vector<std::vector<float>> corners2 {{0,0},{0,height2},{width2,height2},{width2,0}};

    std::vector<std::vector<float>> warpedCorners2 (4, std::vector<float>(2));
    std::vector<std::vector<float>> allCorners = corners1;

    for (int i = 0; i < 4; i++) {
        float cornerX = corners2[i][0];
        float cornerY = corners2[i][1];
        warpedCorners2[i][0] = A.at<double> (0,0) * cornerX +
                            A.at<double> (0,1) * cornerY + A.at<double> (0.2);
        warpedCorners2[i][1] = A.at<double> (1,0) * cornerX +
                            A.at<double> (1,1) * cornerY + A.at<double> (1,2);
        allCorners.push_back (warpedCorners2[i]);
    }

    float xMin = 1e9, xMax = -1e9;
    float yMin = 1e9, yMax = -1e9;
    for (int i = 0; i < 4; i++) {
        xMin = (xMin > warpedCorners[i].x)? warpedCorners[i].x : xMin;
        xMax = (xMax < warpedCorners[i].x)? warpedCorners[i].x : xMax;
        yMin = (yMin > warpedCorners[i].y)? warpedCorners[i].y : yMin;
        yMax = (yMax < warpedCorners[i].y)? warpedCorners[i].y : yMax;
    }
    int xMin_ = (xMin - 0.5);
    int xMax_ = (xMax + 0.5);
    int yMin_ = (yMin - 0.5);
    int yMax_ = (yMax + 0.5);

    cv::Mat translation = (cv::Mat_<double>(3,3) << 1, 0, -xMin_, 0, 1, -yMin_, 0, 0, 1);
    cv::cuda::GpuMat warpedImageTemp;
    cv::cuda::warpPerspective (img2_gpu, warpedImageTemp, translation,
                                cv::Size (xMax_ - xMin_, yMax_ - yMin_));
    cv::cuda::GpuMat warpedImage2;
    cv::cuda::warpAffine (warpedImageTemp, warpedImage2, A,
                          cv::Size (xMax_ - xMin_, yMax_ - yMin_));
    cv::cuda::GpuMat dst;
    cv::cuda::addWeighted (img1_gpu, 1, warpedImage2, 0, 0, dst);
    cv::Mat ret;
    dst.download (ret);

    return ret;
}

cv::Mat combine (std::vector<cv::Mat>& imageList) {
    cv::Mat result = imageList[0];
    for (int i = 1; i < imageList.size(); i++) {
        cv::Mat image = imageList[i];
        result = combinePair (result, image);
        float h = result.rows;
        float w = result.cols;
        if (h > 4000 && w > 4000) {
            if (h > 4000) {
                float hx = 4000/h;
                h = h * hx;
                w = w * hx;
            }
            else if (w > 4000) {
                float wx = 4000/w;
                w = w * wx;
                h = h * wx;
            }
        }
        cv::resize (result, result, cv::Size (w, h));
    }
    return result;
}

void readData (std::string& filename,
               std::vector<imageData>& dataMatrix) {
    std::ifstream file;
    file.open (filename);
    if (file.is_open()) {
        std::string line;
        while (getline (file, line)) {
            std::stringstream ss (line);
            std::string word;
            imageData id;
            int i = 0;
            while (getline (ss, word, ',')) {
                if (i == 0)	{ id.imageName = word; }
                else if (i == 1) { id.latitude = stof(word); }
                else if (i == 2) { id.longitude = stof(word); }
                else if (i == 3) {
                    id.altitudeFeet = stof (word);
                    id.altitudeMeter = id.altitudeFeet * 0.3048;
                }
                else if (i == 4) { id.yaw = stof(word); }
                else if (i == 5) { id.pitch = stof(word); }
                else if (i == 6) { id.roll = stof(word); }
                i++;
            }
            dataMatrix.push_back (id);
        }
    }
}

void getImageList (std::vector<cv::Mat>& imageList,
                   std::vector<imageData>& dataMatrix,
                   std::string base_path) {
    for (auto data : dataMatrix) {
        std::string img_path = base_path + data.imageName;
        cv::Mat img = cv::imread (img_path, 1);
        imageList.push_back (img);
    }
}

void changePerspective (std::vector<cv::Mat>& imageList,
                        std::vector<imageData>& dataMatrix) {
    std::cout << "Warping Images Now" << std::endl;
    int n = imageList.size();
    for (int i = 0; i < n; i++) {
        cv::Mat M = computeUnRotMatrix (dataMatrix[i]);
        cv::Mat correctedImage = warpPerspectiveWithPadding (imageList[i], M);

        cv::imwrite ("/home/ksakash/misc/Drone-Image-Stitching/temp/"
                     +dataMatrix[i].imageName+".png", correctedImage);
    }
    std::cout << "Image Warping Done" << std::endl;
}

int main () {
    std::string filename = "/home/ksakash/misc/Drone-Image-Stitching/datasets/imageData.txt";
    std::vector<imageData> dataMatrix;
    readData (filename, dataMatrix);
    std::vector<cv::Mat> imageList;
    std::string base_path = "/home/ksakash/misc/Drone-Image-Stitching/datasets/images/";
    getImageList (imageList, dataMatrix, base_path);
    changePerspective (imageList, dataMatrix);
    imageList.clear();
    base_path = "/home/ksakash/misc/Drone-Image-Stitching/temp/";
    getImageList (imageList, dataMatrix, base_path);
    cv::Mat result = combine (imageList);
    cv::imwrite ("/home/ksakash/misc/Drone-Image-Stitching/result/result.png", result);
    return 0;
}
