
#include "face_recognition_video.h"

const int HIGH_VALUE = 10000;
const int WIDTH = HIGH_VALUE;
const int HEIGHT = HIGH_VALUE;

FaceDetectorGPU::FaceDetectorGPU(const std::string& face_cascade_name)
{
    // To use logitech c920 camera, for other camera, change the camera id to 0
    this->capture.open(-1);

    if (!this->capture.isOpened())
    {
        std::cerr << "Failed to open camera" << std::endl;
        exit(1);
    }

    this->capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    this->capture.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH);
    this->capture.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);

    this->face_cascade_name = face_cascade_name;
    this->start_cascade_classifier();
}

FaceDetectorGPU::~FaceDetectorGPU()
{
    capture.release();
}

void FaceDetectorGPU::start_cascade_classifier()
{
#ifdef USING_OPENCL
    this->cascade_classifier.load(this->face_cascade_name);
    if (cascade_classifier.empty())
    {
        std::cerr << "Failed to load face cascade classifier" << std::endl;
        exit(1);
    }
#elif USING_CUDA
    this->cascade_classifier = cv::cuda::CascadeClassifier::create(this->face_cascade_name);
    if (!cascade_classifier->empty())
    {
        cascade_classifier->setScaleFactor(1.1);
        cascade_classifier->setMinNeighbors(5);
        cascade_classifier->setMinObjectSize(cv::Size(30, 30));
    }
    else
    {
        std::cerr << "Failed to load face cascade classifier" << std::endl;
        exit(1);
    }
#endif
}

void FaceDetectorGPU::rescale_frame(cv::UMat &frame, cv::UMat &scaled_frame, int percent)
{
    int width = frame.cols * percent / 100;
    int height = frame.rows * percent / 100;
    cv::resize(frame, scaled_frame, cv::Size(width, height), 0, 0, cv::INTER_AREA);
}

void FaceDetectorGPU::detect()
{
    cv::UMat frame;
    cv::UMat frame_resized;
    cv::UMat frame_gray;
    std::vector<cv::Rect> faces;
#ifdef USING_CUDA
    cv::cuda::GpuMat frame_gpu;
    cv::cuda::GpuMat objbuf_gpu;
#endif

    while(true)
    {
        this->capture.read(frame);
        // check if we succeeded
        if (frame.empty())
        {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }

        // resize the frame to make it faster
        this->rescale_frame(frame, frame_resized, 30);
        cv::cvtColor(frame_resized, frame_gray, cv::COLOR_BGR2GRAY);
#ifdef USING_OPENCL
        this->cascade_classifier.detectMultiScale(frame_gray,
                                                  faces,
                                                  1.1,
                                                  2,
                                                  0 | cv::CASCADE_SCALE_IMAGE,
                                                  cv::Size(30, 30));
#elif USING_CUDA
        frame_gpu.upload(frame_gray);
        this->cascade_classifier->detectMultiScale(frame_gpu, objbuf_gpu);
        this->cascade_classifier->convert(objbuf_gpu, faces);
#endif
        for (cv::Rect &face : faces)
        {
            cv::rectangle(frame_resized, face, cv::Scalar(255, 0, 0), 2);
        }

        // show live and wait for a key with timeout long enough to show images
        cv::imshow("Faces", frame_resized);

        // Wait for a key with timeout long enough to show images
        if (cv::waitKey(5) == 'q')
        {
            break;
        }
    }
}
