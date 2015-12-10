#include "StdAfx.h"
#include <GPUCVSwitch/macro.h>
#include <GPUCVCore/GpuTextureManager.h>
#include <GPUCVSwitch/Cl_Dll.h>
#include <GPUCVSwitch/switch.h>

#define _GPUCV_FORCE_OPENCV_NP 1
#include <includecv.h>
#include <highgui.h>


using namespace std;
using namespace GCV;
#define CVAPI(MSG) MSG