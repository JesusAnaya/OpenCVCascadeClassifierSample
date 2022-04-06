# OpenCV with OpenCL and CUDA acceleration example (C++)

This repository contains a simple example of OpenCV with OpenCL and CUDA acceleration.

Please read the official OpenCV documentation for more information:
- https://docs.opencv.org/4.5.5/db/d28/tutorial_cascade_classifier.html

## Compile

To compile, first edit the CMakeLists.txt file and change the `USING_OPENCL` to `USING_CUDA` in case you
are using cuda.

After that, run the following command:

```bash
cmake CMakeLists.txt
```

It will generate a Makefile. Now to make the project, run the following command:

```bash
make
```

The project executable will be created with the project name, by default it is `face_recognition_video`, 
but you can change it in the CMakeLists.txt file.
