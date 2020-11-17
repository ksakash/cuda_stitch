#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>

void example_with_full_gpu(const cv::Mat &img1, const cv::Mat img2) {
//Upload from host memory to gpu device memeory
cv::cuda::GpuMat img1_gpu(img1), img2_gpu(img2);
cv::cuda::GpuMat img1_gray_gpu, img2_gray_gpu;

//Convert RGB to grayscale as gpu detectAndCompute only allow grayscale GpuMat
cv::cuda::cvtColor(img1_gpu, img1_gray_gpu, CV_BGR2GRAY);
cv::cuda::cvtColor(img2_gpu, img2_gray_gpu, CV_BGR2GRAY);

//Create a GPU ORB feature object
//blurForDescriptor=true seems to give better results
//http://answers.opencv.org/question/10835/orb_gpu-not-as-good-as-orbcpu/
cv::Ptr<cv::cuda::ORB> orb = cv::cuda::ORB::create(500, 1.2f, 8, 31, 0, 2, 0, 31, 20, true);

cv::cuda::GpuMat keypoints1_gpu, descriptors1_gpu;
//Detect ORB keypoints and extract descriptors on train image (box.png)
orb->detectAndComputeAsync(img1_gray_gpu, cv::cuda::GpuMat(), keypoints1_gpu, descriptors1_gpu);
std::vector<cv::KeyPoint> keypoints1;
//Convert from CUDA object to std::vector<cv::KeyPoint>
orb->convert(keypoints1_gpu, keypoints1);
std::cout << "keypoints1=" << keypoints1.size() << " ; descriptors1_gpu=" << descriptors1_gpu.rows
    << "x" << descriptors1_gpu.cols << std::endl;

std::vector<cv::KeyPoint> keypoints2;
cv::cuda::GpuMat descriptors2_gpu;
//Detect ORB keypoints and extract descriptors on query image (box_in_scene.png)
//The conversion from internal data to std::vector<cv::KeyPoint> is done implicitly in detectAndCompute()
orb->detectAndCompute(img2_gray_gpu, cv::cuda::GpuMat(), keypoints2, descriptors2_gpu);
std::cout << "keypoints2=" << keypoints2.size() << " ; descriptors2_gpu=" << descriptors2_gpu.rows
    << "x" << descriptors2_gpu.cols << std::endl;

//Create a GPU brute-force matcher with Hamming distance as we use a binary descriptor (ORB)
cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

std::vector<std::vector<cv::DMatch> > knn_matches;
//Match each query descriptor to a train descriptor
matcher->knnMatch(descriptors2_gpu, descriptors1_gpu, knn_matches, 2);
std::cout << "knn_matches=" << knn_matches.size() << std::endl;

std::vector<cv::DMatch> matches;
//Filter the matches using the ratio test
for(std::vector<std::vector<cv::DMatch> >::const_iterator it = knn_matches.begin(); it != knn_matches.end(); ++it) {
    if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.8) {
        matches.push_back((*it)[0]);
    }
}

cv::Mat imgRes;
//Display and save the image with matches
cv::drawMatches(img2, keypoints2, img1, keypoints1, matches, imgRes);
cv::imshow("imgRes", imgRes);
cv::imwrite("GPU_ORB-matching.png", imgRes);

cv::waitKey(0);
}
