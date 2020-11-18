#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <Eigen/Dense>
#include <math.h>

#define PI 3.141592653589793238

cv::Mat computeUnRotMatrix (vector<int>& pose) {
    float a = pose[3] * PI / 180;
    float b = pose[4] * PI / 180;
    float g = pose[5] * PI / 180;
    Matrix3f Rz << cos (a), -sin (a), 0,
                   sin (a), cos (a), 0,
                   0, 0, 1;
    Matrix3f Ry << cos (b), 0, sin (b),
                   0, 1, 0,
                   -sin (b), 0, cos (b);
    Matrix3f Rx << 1, 0, 0,
                   0, cos (g), -sin (g),
                   0, sin (g), cos (g);
    Matrix3f R = Rz * (Rx * Ry);
    R[0][2] = 0;
    R[1][2] = 0;
    R[2][2] = 1;
    Matrix3f Rtrans = R.transpose ();
    Matrix3f InvR = Rtrans.inverse ();
    cv::Mat transformation = (cv::Mat_<double>(3,3) << InvR[0][0], InvR[0][1], InvR[0][2],
                                                       InvR[1][0], InvR[1][1], InvR[1][2],
                                                       InvR[2][0], InvR[2][1], InvR[2][2]);
    return transformation;
}

bool cmp (cv::Point2f& a, cv::Point2f& b) {
    return (a.x == b.x)? a.y < b.y : a.x < b.x;
}

void warpPerspectiveWithPadding (const cv::Mat& image, cv::Mat& transformation,
                                 cv::Mat& dst) {
    int height = image.row();
    int width = image.col();
    vector<cv::Point2f> corners = {cv::Point2f (0,0), cv::Point2f (0,height),
                                   cv::Point2f (width,height), cv::Point2f (width,0)};
    vector<cv::Point2f> warpedCorners;
    cv::perspectiveTransform (corners, warpedCorners, transformation);
    sort (warpedCorners.begin(), warpedCorners.end(), cmp);
    float xMin = warpCorners[0].x - 0.5, yMin = warpCorners[0].y - 0.5;
    float xMax = warpCorners[3].x + 0.5, yMax = warpCorners[3].y + 0.5;
    cv::Mat translation = (cv::Mat_<double>(3,3) << 1, 0, -xMin, 0, 1, -yMin, 0, 0, 1);
    cv::Mat fullTransformation = translation * transformation;
    cv::cuda::GpuMat result;
    cv::cuda::warpPerspective (cv::cuda::GpuMat (image), result,
                               cv::cuda::GpuMat (fulltransformation),
                               cv::Size (xMax-xMin, yMax-yMin));
    dst = result;
}

void combinePair (cv::Mat& img1, cv::Mat& img2) {
    cv::cuda::GpuMat img1_gpu (img1), img2_gpu (img2);
    cv::cuda::GpuMat img1_gray_gpu, img2_gray_gpu;

    cv::cuda::cvtColor (img1_gpu, img1_gray_gpu, CV_BGR2GRAY);
    cv::cuda::cvtColor (img2_gpu, img2_gray_gpu, CV_BGR2GRAY);

    cv::cuda::GpuMat mask1;
    cv::cuda::GpuMat mask2;

    cv::cuda::threshold (img1_gray_gpu, mask1, 1, 255, CV_THRESH_BINARY);
    cv::cuda::threshold (img2_gray_gpu, mask2, 1, 255, CV_THRESH_BINARY);

    cv::Ptr<cv::cuda::SURF_CUDA> detector = cv::cuda::SURF_CUDA::create (1000);

    cv::cuda::GpuMat keypoints1_gpu, descriptors1_gpu;
    detector->detectAndComputeAsync (img1_gray_gpu, mask1,
                                    keypoints1_gpu, descriptors1_gpu);
    std::vector<cv::KeyPoint> keypoints1;
    detector->convert (keypoints1_gpu, keypoints1);

    std::vector<cv::KeyPoint> keypoints2;
    cv::cuda::GpuMat descriptors2_gpu;
    detector->detectAndCompute (img2_gray_gpu, mask2,
                                keypoints2, descriptors2_gpu);

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

    std::vector<cv::KeyPoint> src_pts;
    std::vector<cv::KeyPoint> dst_pts;
    for (auto m : matches) {
        src_pts.push_back (keypoints2[m.queryIdx].pt);
        dst_pts.push_back (keypoints1[m.trainIdx].pt);
    }

    cv::Mat A = cv::estimateRigidTransform(src_pts, dst_pts, false);
    int height1 = img1.row(), width1 = img1.col();
    int height2 = img2.row(), width2 = img2.col();

    vector<vector<float>> corners1 {{0,0},{0,height1},{width1,height1},{width1,0}};
    vector<vector<float>> corners2 {{0,0},{0,height2},{width2,height2},{width2,0}};

    vector<vector<float>> warpedCorners2 (4, vector<float>(2));
    vector<vector<float>> allCorners = corners1;

    for (int i = 0; i < 4; i++) {
        float cornerX = corners2[i][0];
        float cornerY = corners2[i][1];
        warpedCorners2[i][0] = A[0][0] * cornerX + A[0][1] * cornerY + A[0][2];
        warpedCorners2[i][1] = A[1][0] * cornerX + A[1][1] * cornerY + A[1][2];
        allCorners.push_back (warpedCorners2[i]);
    }

    sort (allCorners.begin(), allCorners.end());
    float minX = allCorners[0][0] - 0.5, minY = allCorners[0][1] - 0.5;
    float maxX = allCorners[7][0] + 0.5, maxY = allCorners[7][1] + 0.5;

    cv::Mat translation = (cv::Mat_<double>(3,3) << 1, 0, -minX, 0, 1, -minY, 0, 0, 1);
    cv::cuda::GpuMat warpedImageTemp;
    cv::cuda::warpPerspective (img2_gpu, warpedImageTemp, cv::cuda::GpuMat (translation),
                                cv::Size (maxX - minX, maxY - minY));
    cv::cuda::GpuMat warpedImage2;
    cv::cuda::warpAffine (warpedImageTemp, warpedImage2, cv::cuda::GpuMat (A),
                          cv::Size (maxX - minX, maxY - minY));
    cv::cuda::GpuMat dst;
    cv::cuda::addWeighted (img1_gpu, 1, warpedImage2, 0, 0, dst);
    cv::Mat ret = dst;

    return ret;
}

cv::Mat combine () {
    vector<cv::Mat> imageList = {};
    cv::Mat result = imageList[0];
    for (int i = 1; i < imageList.size(); i++) {
        cv::Mat image = imageList[i];
        result = combinePair (result, image);
        float h = result.row(), w = result.col();
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

int main () {
    return 0;
}
