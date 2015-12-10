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
#include <GPUCVHardware/gcvGL.h>
#include <cvgcu/cvgcu.h>

//=================================
//CVG_IMGPROC__FILTERS_CLR_CONV_GRP
//=================================
#if _GPUCV_COMPILE_CUDA

#include <vector_types.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <cxcoreg/cxcoreg.h>
#include <cxcoregcu/cxcoregcu.h>


using namespace GCV;

//cvThreshold================================================
_GPUCV_CVGCU_EXPORT_CU void gcuThreshold(CvArr* srcArr, CvArr* dstArr, double threshold, double max_value, int threshold_type );
void cvgCudaThreshold(CvArr* srcArr, CvArr* dstArr, double threshold, double max_value, int threshold_type )
{
	GPUCV_START_OP(cvThreshold(srcArr, dstArr, threshold, max_value,threshold_type),
		"cvgCudaThreshold",
		dstArr,
		GenericGPU::HRD_PRF_CUDA);

	//		GCU_Assert((threshold_type < 4) && (threshold_type > 0), "cvgCudaThreshold(), bad value of threshold type");
	GCV_OPER_ASSERT(srcArr, "no input images src!");
	GCV_OPER_ASSERT(dstArr, "no destination image!");
	GCV_OPER_COMPAT_ASSERT(((IplImage*)srcArr)->depth != IPL_DEPTH_64F, "Image depth should NOT be IPL_DEPTH_64F!");

	GCV_OPER_ASSERT(GetnChannels(srcArr)==1, "source must be 1 channels!");
	GCV_OPER_ASSERT(GetnChannels(dstArr)==1, "source must be 1 channels!");
	gcuThreshold((IplImage*)srcArr, (IplImage*)dstArr, threshold, max_value,threshold_type);

	GPUCV_STOP_OP(
		cvThreshold(srcArr, dstArr, threshold, max_value, threshold_type),
		srcArr, dstArr, NULL, NULL
		);
}

//cvIntegral================================================
#ifdef _GPUCV_CUDA_SUPPORT_CUDPP

_GPUCV_CVGCU_EXPORT_CU void initializeSAT(int width, int height, int channels, unsigned int _glDepth);
_GPUCV_CVGCU_EXPORT_CU void gcuSAT(CvArr* src, CvArr* dst);
_GPUCV_CVGCU_EXPORT_CU void finalizeSAT(bool _preserveMemory);

void cvgCudaIntegral(CvArr* image, CvArr* sum, CvArr* sqsum, CvArr* tilted_sum)
{
	GPUCV_START_OP(cvIntegral(image, sum, sqsum, tilted_sum),
		"cvgCudaIntegral",
		sqsum,
		GenericGPU::HRD_PRF_CUDA);
	int width		 = gcuGetWidth(image);
	int height		 = gcuGetHeight(image);
	int src_depth	 = gcuGetGLDepth(image);



	/*
	GLuint InternalFormat=0;
	GLuint Format=0;
	GLuint PixType=0;
	cvgConvertCVMatrixFormatToGL(src_depth,InternalFormat,Format,PixType);
	int src_channels = gcuGetnChannels(image);
	*/
	GCV_OPER_ASSERT(image,	"no input images src!");
	GCV_OPER_ASSERT(image,	"source must be int 3 channels!");
	GCV_OPER_ASSERT(sum,	"no destination image!");
	//GCV_OPER_ASSERT(sqsum, "do not support Square Sum!");

	initializeSAT(GetWidth(image), GetHeight(image), GetnChannels(image), GetGLDepth(sum));
	if(sum)//simple integral
	{

		int sum_depth	 = gcuGetGLDepth(sum);
		GCV_OPER_COMPAT_ASSERT((sum_depth==GL_UNSIGNED_INT)||
			(sum_depth==GL_INT)||
			(sum_depth==GL_FLOAT)||
			(sum_depth==GL_DOUBLE)
			,"sum type is incorrect, must be GL_UNSIGNED_INT, GL_FLOAT, or GL_DOUBLE!");
		gcuSAT(image, sum);
	}

	if (sqsum)
	{
		int sqsum_depth	 = gcuGetGLDepth(sqsum);
		GCV_OPER_COMPAT_ASSERT((sqsum_depth==GL_DOUBLE)
			||(sqsum_depth==GL_FLOAT)//double is not always supported by graphics cards
			,"sqsum type is incorrect, must be GL_DOUBLE!");
		CvSize size = cvGetSize(image);
		//size.width  +=1;
		//size.height +=1;
		CvArr * powArr = NULL;

		if(CV_IS_MAT(image))
			powArr = cvgCreateMat(size.width, size.height, GetCVDepth(sqsum));
		else if (CV_IS_IMAGE(image))
			powArr = cvgCreateImage(size, GetCVDepth(sqsum),GetnChannels(sqsum));

		cvgSetOptions(powArr, DataContainer::CPU_RETURN, 0);

		cvgCudaPow(image,powArr,2);
		gcuSAT(powArr,sqsum);

		if(CV_IS_MAT(image))
			cvgReleaseMat((CvMat**)&powArr);
		else if (CV_IS_IMAGE(image))
			cvgReleaseImage((IplImage**)&powArr);
	}
	if(tilted_sum)
	{
		GPUCV_WARNING("cvgCudaIntegral() do not support Tilted Sum!");
		GCV_OPER_COMPAT_ASSERT( gcuGetGLDepth(sum)== gcuGetGLDepth(tilted_sum)
			,"tilted_sum must be same type as sum!");
		//int sqsum_depth	 = gcuGetGLDepth(tilted_sum);
		//rotation by 45 degree before integral...
		//gcuSAT(powArr, sum);
	}
	finalizeSAT(true);//we preserve memory and delete all temporary objects

	GPUCV_STOP_OP(cvIntegral(image, sum, sqsum, tilted_sum),
		image, NULL,sum, NULL
		);
}
#endif
//================================================

//cvCvtColor================================================
_GPUCV_CVGCU_EXPORT_CU void gcuCvtColor(CvArr* srcArr, CvArr* dstArr, int code);
 
void cvgCudaCvtColor(CvArr* src, CvArr* dst, int code)
{
	GPUCV_START_OP(cvCvtColor(src, dst, code),
		"cvgCudaCvtColor",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src, "no input images src!");
	GCV_OPER_ASSERT(GetnChannels(src)==3, "source must be in 3 channels!");
	GCV_OPER_ASSERT(dst, "no destination image!");
	switch (code) {
				case CV_BGR2GRAY:
					SG_Assert(GetnChannels(dst) == 1, "BGR to GRAY conversion : nb of destination image channels is not correct.\n");
					break;
				case CV_RGB2GRAY:
					SG_Assert(GetnChannels(dst) == 1, "RGB to GRAY conversion : nb of destination image channels is not correct.\n");
					break;
		/*		case CV_BGR2YCrCb:
					SG_Assert(GetnChannels(dst) == 3, "BGR to YCrCb conversion : nb of destination image channels is not correct.\n");
					break;
				case CV_RGB2YCrCb:
					SG_Assert(GetnChannels(dst) == 3, "RGB to YCrCb conversion : nb of destination image channels is not correct.\n");
					break;
		*/		default:
					GCV_OPER_TODO_ASSERT(0, "requested color conversion not done yet");
	}
	gcuCvtColor(src, dst, code);

	GPUCV_STOP_OP(cvCvtColor(src, dst, code),
		src, dst, NULL, NULL
		);
}
//================================================
#endif//CUDA_COMPILE

