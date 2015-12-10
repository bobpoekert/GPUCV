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

#if _GPUCV_COMPILE_CUDA

// Required to include CUDA vector types
#include <vector_types.h>
#include <SugoiTracer/tracer_c.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/cuda_misc.h>
#include <GPUCVTexture/fbo.h>
#include <GPUCVHardware/moduleInfo.h>
#include <GPUCVHardware/moduleInfo.h>


using namespace GCV;


_GPUCV_CVGCU_EXPORT_C LibraryDescriptor *modGetLibraryDescriptor(void)
{
	static LibraryDescriptor * pLibraryDescriptor=NULL;
	if(pLibraryDescriptor ==NULL)//init
	{
		pLibraryDescriptor = new LibraryDescriptor();
		pLibraryDescriptor->SetVersionMajor("1");
		pLibraryDescriptor->SetVersionMinor("0");
		pLibraryDescriptor->SetSvnRev("570");
		pLibraryDescriptor->SetSvnDate("");
		pLibraryDescriptor->SetWebUrl(DLLINFO_DFT_URL);
		pLibraryDescriptor->SetAuthor(DLLINFO_DFT_AUTHOR);
		pLibraryDescriptor->SetDllName("cvgcu");
		pLibraryDescriptor->SetImplementationName("CUDA");
		pLibraryDescriptor->SetBaseImplementationID(GPUCV_IMPL_CUDA);
		pLibraryDescriptor->SetUseGpu(true);
		pLibraryDescriptor->SetStartColor(GPUCV_IMPL_CUDA_COLOR_START);
		pLibraryDescriptor->SetStopColor(GPUCV_IMPL_CUDA_COLOR_STOP);
	}
	return pLibraryDescriptor;
}
_GPUCV_CVGCU_EXPORT_C int cvgCudaDLLInit(bool InitGLContext, bool isMultiThread)
{
	return cvgcuInit(InitGLContext, isMultiThread);
}


_GPUCV_CVGCU_EXPORT_CU void gcuConvolutionTPL_OptimizedKernel(int _ConvolType, IplImage* src,IplImage* dst, IplConvKernel* element,int iterations,int aperture_size=0);

#if 0
template <int TConvolutionType>
void cvgCudaConvolutionTpl(CvArr* src, CvArr* dst, IplConvKernel* element/*=NULL*/, int iterations/*=1*/)
{
#if (TConvolutionType == CONVOLUTION_KERNEL_ERODE)
	std::string TplFctName = "cvgCudaConvolutionTpl<CONVOLUTION_KERNEL_ERODE>";
#define CV_CONVOLUTION_FCT cvErode
#elif (TConvolutionType == CONVOLUTION_KERNEL_DILATE)
	std::string TplFctName = "cvgCudaConvolutionTpl<CONVOLUTION_KERNEL_DILATE>";
#define CV_CONVOLUTION_FCT cvDilate
#else
	std::string TplFctName = "cvgCudaConvolutionTpl<UNKNOWN>";
#define CV_CONVOLUTION_FCT 
#endif
	GPUCV_START_OP(CV_CONVOLUTION_FCT(src, dst, element, iterations),
		TplFctName, 
		dst,
		GenericGPU::PROFILE_4);

	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");

	cvgSetOptions(dst, DataContainer::DEST_IMG, true);
	cvgSetOptions(src, DataContainer::DEST_IMG, false);
	gcuConvolutionTPL_OptimizedKernel(TConvolutionType, (IplImage*)src, (IplImage*)dst, element, iterations);
	cvgSetOptions(dst, DataContainer::DEST_IMG, false);


	GPUCV_STOP_OP(
		CV_CONVOLUTION_FCT(src, dst, element, iterations),
		src, dst, NULL, NULL
		);
}
#endif
void cvgCudaDilate(CvArr* src, CvArr* dst, IplConvKernel* element/*=NULL*/, int iterations/*=1*/)
{
#if 0
	cvgCudaConvolutionTpl<CONVOLUTION_KERNEL_DILATE>(src, dst, element, iterations);
#else
	GPUCV_START_OP(cvDilate(src, dst, element, iterations),
		"cvgCudaDilate", 
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");

	//	cvgSetOptions(dst, DataContainer::DEST_IMG, true);
	//	cvgSetOptions(src, DataContainer::DEST_IMG, false);
//	gcuConvolutionTPL_OptimizedKernel(CONVOLUTION_KERNEL_DILATE, (IplImage*)src, (IplImage*)dst, element, iterations);
	//	cvgSetOptions(dst, DataContainer::DEST_IMG, false);


	GPUCV_STOP_OP(
		cvDilate(src, dst, element, iterations),
		src, dst, NULL, NULL
		);
#endif
}

void cvgCudaErode(CvArr* src, CvArr* dst, IplConvKernel* element/*=NULL*/, int iterations/*=1*/)
{
	GPUCV_START_OP(cvErode(src, dst, element, iterations),
		"cvgCudaErode", 
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");

	//	cvgSetOptions(dst, DataContainer::DEST_IMG, true);
	//	cvgSetOptions(src, DataContainer::DEST_IMG, false);
//		gcuConvolutionTPL_OptimizedKernel(CONVOLUTION_KERNEL_ERODE, (IplImage*)src, (IplImage*)dst, element, iterations);
	//	cvgSetOptions(dst, DataContainer::DEST_IMG, false);


	GPUCV_STOP_OP(
		cvErode(src, dst, element, iterations),
		src, dst, NULL, NULL
		);
}
void cvgCudaErodeNoShare(CvArr* src, CvArr* dst, IplConvKernel* element/*=NULL*/, int iterations/*=1*/)
{
	GPUCV_START_OP(cvErode(src, dst, element, iterations),
		"cvgCudaErode", 
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");

	cvgSetOptions(dst, DataContainer::DEST_IMG, true);
	cvgSetOptions(src, DataContainer::DEST_IMG, false);
//		gcuConvolutionTPL_OptimizedKernel(CONVOLUTION_KERNEL_ERODE_NOSHARE, (IplImage*)src, (IplImage*)dst, element, iterations);
	cvgSetOptions(dst, DataContainer::DEST_IMG, false);


	GPUCV_STOP_OP(
		cvErode(src, dst, element, iterations),
		src, dst, NULL, NULL
		);
}

//==========================================================================================
_GPUCV_CVGCU_EXPORT_CU void CudaSmouthIplTPL(IplImage* src,IplImage* dst, int smoothtype/*=CV_GAUSSIAN*/,
											int param1/*=3*/, int param2/*=0*/, double param3/*=0*/, double param4/*=0*/);

void cvgCudaSmooth(CvArr* src, CvArr* dst, int smoothtype/*=CV_GAUSSIAN*/, int param1/*=3*/, int param2/*=0*/, double param3/*=0*/, double param4/*=0*/)
{
	GPUCV_START_OP(cvSmooth(src, dst, smoothtype,param1, param2, param3, param4),
		"cvgCudaSmooth", 
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");

	
#ifdef _LINUX
	GCV_OPER_COMPAT_ASSERT(0, "not currently compatible with LINUX");
#else
	cvgSetOptions(dst, DataContainer::DEST_IMG, true);
	cvgSetOptions(src, DataContainer::DEST_IMG, false);
	CudaSmouthIplTPL((IplImage*)src, (IplImage*)dst, smoothtype,param1, param2, param3, param4);
	cvgSetOptions(dst, DataContainer::DEST_IMG, false);
#endif

	GPUCV_STOP_OP(
		cvSmooth(src, dst, smoothtype,param1, param2, param3, param4),
		src, dst, NULL, NULL
		);
}

#endif




