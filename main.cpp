#include "face_recognition_video.h"

int main(int, char**)
{
    FaceDetectorGPU face_detector("./data/haarcascades/haarcascade_frontalface_default.xml");
    face_detector.detect();
    return 0;
}
