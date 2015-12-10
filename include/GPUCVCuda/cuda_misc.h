//CVG_LicenseBegin==============================================================
//
//	Copyright@ Institut TELECOM 2005
//		http://www.institut-telecom.fr/en_accueil.html
//	
//	This software is a GPU accelerated library for computer-vision. It 
//	supports an OPENCV-like extensible interface for easily porting OPENCV 
//	applications.
//	
//	Contacts :
//		patrick.horain@it-sudparis.eu
//		gpucv-developers@picoforge.int-evry.fr
//	
//	Project's Home Page :
//		https://picoforge.int-evry.fr/cgi-bin/twiki/view/Gpucv/Web/WebHome
//	
//	This software is governed by the CeCILL-B license under French law and
//	abiding by the rules of distribution of free software.  You can  use, 
//	modify and/ or redistribute the software under the terms of the CeCILL-B
//	license as circulated by CEA, CNRS and INRIA at the following URL
//	"http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html". 
//	
//================================================================CVG_LicenseEnd



/** \brief Contains some functions to control GpuCV-CUDA plugin.
*	\author Yannick Allusse
*/

#ifndef __GPUCV_CUDA_MISC_H
#define __GPUCV_CUDA_MISC_H
#include "GPUCVCuda/config.h"
#include <GPUCVHardware/moduleInfo.h>

#if _GPUCV_COMPILE_CUDA

//namespace GCV{

/** \sa DLLInfos;
*/

//DLLInfos* modGetDLLInfos(void);


/*!
*	\brief Set the device ID of a CUDA capable GPU. Default device ID is the one returned by cutGetMaxGflopsDeviceId(). Must be called before cvgcuInit().
*	\param _devID -> CUDA device ID, if -1 the ID will be the one returned by cutGetMaxGflopsDeviceId().
*	\sa cvgTerminate(), cvgInit(), GpuCVInit(), GpuCVTerminate()
*   \author Yannick Allusse
*/
_GPUCV_CUDA_EXPORT 
void cvgCudaSetProcessingDevice(char _devID=-1);

/*!
*	\brief Initialize GpuCV library and framework to use with CUDA. Check that host is compatible with CUDA.
*	\param InitGLContext -> define if library should create its own GL context or use an existing one
*	\param isMultiThread -> allow to use GPUCV library on multi-thread programs (in development)
*	\return int -> status
*	\sa cvgTerminate(), cvgInit(), GpuCVInit(), GpuCVTerminate(), cvgCudaSetProcessingDevice()
*   \author Yannick Allusse
*/
_GPUCV_CUDA_EXPORT 
int  cvgcuInit(bool InitGLContext=true, bool isMultiThread=false);

_GPUCV_CUDA_EXPORT 
void cvgCudaThreadSynchronize();

//}//namespace GCV

#endif
#endif
