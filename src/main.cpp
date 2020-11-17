#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>

// computeUnRotMatrix () {}
// warpPerspectiveWithPadding () {}
// combine () {}

cv::Mat combinePair (const cv::Mat& img1, const cv::Mat& img2) {
    cv::cuda::GpuMat img1_gpu (img1), img2_gpu (img2);
    cv::cuda::GpuMat img1_gray_gpu, img2_gray_gpu;

    cv::cuda::cvtColor (img1_gpu, img1_gray_gpu, CV_BGR2GRAY);
    cv::cuda::cvtColor (img2_gpu, img2_gray_gpu, CV_BGR2GRAY);

    cv::Ptr<cv::cuda::ORB> orb =
        cv::cuda::ORB::create (500, 1.2f, 8, 31, 0, 2, 0, 31, 20, true);

    cv::cuda::GpuMat keypoints1_gpu, descriptors1_gpu;
    orb->detectAndComputeAsync (img1_gray_gpu, cv::cuda::GpuMat(),
                                keypoints1_gpu, descriptors1_gpu);
    std::vector<cv::KeyPoint> keypoints1;
    orb->convert (keypoints1_gpu, keypoints1);

    std::vector<cv::KeyPoint> keypoints2;
    cv::cuda::GpuMat descriptors2_gpu;
    orb->detectAndCompute (img2_gray_gpu, cv::cuda::GpuMat(),
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
    vector<float> minm = {allCorners[0][0] - 0.5, allCorners[0][1] - 0.5};
    vector<float> maxm =
            {allCorners[allCorners.size()-1] + 0.5, allCorners[allCorners.size()-1] + 0.5};

    // vector<vector<float>> translation = {{1,0,-minm[0]},{0,1,-minm[1]},{0,0,1}};
    cv::Mat translation = (Mat_<double>(3,3) << 1, 0, -minm[0], 0, 1, -minm[1], 0, 0, 1);
    cv::cuda::GpuMat warpedImageTemp;
    cv::cuda::warpPerspective (img2_gpu, warpedImageTemp, cv::cuda::GpuMat (translation),
                                cv::Size (maxm[0] - minm[0], maxm[1] - minm[1]), cv::INTER_LINEAR,
                                cv::BORDER_CONSTANT, 0, Stream::Null());
    cv::cuda::GpuMat warpedImage2;
    cv::cuda::warpAffine (warpedImageTemp, warpedImage2, cv::cuda::GpuMat (A),
                          cv::Size (maxm[0] - minm[0], maxm[1] - minm[1]), cv::INTER_LINEAR,
                          cv::BORDER_CONSTANT, 0, Stream::Null());
    cv::cuda::GpuMat dest;
    cv::cuda::addWeighted (img1_gpu, 1, warpedImage2, 0, 0, dst);

    cv::Mat rest = dest;

    return rest;

    // https://stackoverflow.com/questions/19068085/shift-image-content-with-opencv
    // Mat trans_mat = (Mat_<double>(2,3) << 1, 0, offsetx, 0, 1, offsety);
}

int main () {
    return 0;
}
