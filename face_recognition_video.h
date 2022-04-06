//
// Created by Jesus Armando Anaya on 05/04/22.
//

#ifndef OPENCVSAMPLES_FACE_RECOGNITION_VIDEO_H
#define OPENCVSAMPLES_FACE_RECOGNITION_VIDEO_H

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/ocl.hpp>
#if USING_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaobjdetect.hpp>
#endif
#include <iostream>

class FaceDetectorGPU
{
private:
    std::string face_cascade_name;
    cv::VideoCapture capture;
#ifdef USING_OPENCL
    cv::CascadeClassifier cascade_classifier;
#elif USING_CUDA
    cv::Ptr<cv::cuda::CascadeClassifier> cascade_classifier;
#endif
    void start_cascade_classifier();
    static void rescale_frame(cv::UMat &frame, cv::UMat &scaled_frame, int percent);
public:
    explicit FaceDetectorGPU(const std::string& face_cascade_name);
    ~FaceDetectorGPU();
    void detect();
};

#endif //OPENCVSAMPLES_FACE_RECOGNITION_VIDEO_H
