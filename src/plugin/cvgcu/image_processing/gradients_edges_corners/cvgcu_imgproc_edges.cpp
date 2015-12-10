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
#include "StdAfx.h"
#include <cvgcu/cvgcu.h>
//=================================
//CVG_IMGPROC__GRAD_EDGE_CORNER_GRP
//=================================
#if _GPUCV_COMPILE_CUDA



#include <vector_types.h>
#include <SugoiTracer/tracer_c.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <GPUCVTexture/fbo.h>

using namespace GCV;

_GPUCV_CVGCU_EXPORT_CU void CudaSobelIpl(IplImage* src, 
										IplImage* dst, 
										int xorder, int yorder, 
										int aperture_size);
_GPUCV_CVGCU_EXPORT_CU void CudaSobelIplTEst(IplImage* src, 
											IplImage* dst, 
											int xorder, int yorder, 
											int aperture_size);

_GPUCV_CVGCU_EXPORT_CU void CudaLaplaceIpl(IplImage* src, IplImage* dst, int aperture_size);
_GPUCV_CVGCU_EXPORT_CU void gcuConvolutionTPL_OptimizedKernel(int _ConvolType, IplImage* src,IplImage* dst, IplConvKernel* element,int iterations,int aperture_size=0);
_GPUCV_CVGCU_EXPORT_CU bool gcuSobel(void* src, void* dst, int xorder, int yorder, int aperture_size);
void cvgCudaSobel(CvArr* src, CvArr* dst, int xorder, int yorder, int aperture_size/*=3*/ )
{
	GPUCV_START_OP(cvSobel(src, dst, xorder, yorder, aperture_size),
		"cvgCudaSobel", 
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_TODO_ASSERT(NULL, "Operator not stable!");
	GCV_OPER_ASSERT(src, "No input images!");
	GCV_OPER_ASSERT(dst, "No destination image!");
	GCV_OPER_ASSERT(GetCVDepth(dst)!=IPL_DEPTH_8S, "Destination image can not be 8bits!");
	GCV_OPER_ASSERT(GetCVDepth(dst)!=IPL_DEPTH_8U, "Destination image can not be 8bits!");
	GCV_OPER_COMPAT_ASSERT(gcuGetnChannels(dst)==1, "Destination must be single channel!");
	GCV_OPER_COMPAT_ASSERT(gcuGetnChannels(src)==1, "Source must be single channel!");
	GCV_OPER_COMPAT_ASSERT(gcuGetWidth(src)==gcuGetHeight(src), "Image must be squared!");
	GCV_OPER_ASSERT(aperture_size<=CV_MAX_SOBEL_KSIZE, "Maximum aperture size reach!");

	GCV_OPER_COMPAT_ASSERT(gcuSobel(src, dst, xorder, yorder, aperture_size),"Operator does not support this input set of parameters");

	GPUCV_STOP_OP(
		cvSobel(src, dst, xorder, yorder, aperture_size),
		src, dst, NULL, NULL
		);
}
//==========================================================================================
void cvgCudaLaplace(CvArr* src, CvArr* dst, int aperture_size/*=3*/ )
{
	GPUCV_START_OP(cvLaplace(src, dst, aperture_size),
		"cvgCudaLaplace", 
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");
	GCV_OPER_COMPAT_ASSERT(aperture_size==1, "only aperture_size of 3 is processed on CUDA!");

	cvgSetOptions(dst, DataContainer::DEST_IMG, true);
	cvgSetOptions(src, DataContainer::DEST_IMG, false);
	//gcuConvolutionTPL_OptimizedKernel(CONVOLUTION_KERNEL_LAPLACE,(IplImage*)src, (IplImage*)dst, NULL, 1,aperture_size);
	cvgSetOptions(dst, DataContainer::DEST_IMG, false);

	GPUCV_STOP_OP(
		cvLaplace(src, dst, aperture_size),
		src, dst, NULL, NULL
		);
}
#endif//CUDA_COMPILE
