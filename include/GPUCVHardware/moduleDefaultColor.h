// Define default color for plugins...
#include <GPUCVHardware/moduleInfo.h>

#define GPUCV_IMPL_CUDA_COLOR_START CreateModColor(20, 140, 20,255)
#define GPUCV_IMPL_CUDA_COLOR_STOP CreateModColor(20, 255, 20,255)

#define GPUCV_IMPL_GLSL_COLOR_START CreateModColor(20, 20, 140,255)
#define GPUCV_IMPL_GLSL_COLOR_STOP CreateModColor(20, 20, 255,255)

#define GPUCV_IMPL_OPENCV_COLOR_START CreateModColor(10, 10, 10,255)
#define GPUCV_IMPL_OPENCV_COLOR_STOP CreateModColor(150, 150, 150,255)

#define GPUCV_IMPL_OPENCL_COLOR_START CreateModColor(140, 20, 20,255)
#define GPUCV_IMPL_OPENCL_COLOR_STOP CreateModColor(140, 20, 20,255)
