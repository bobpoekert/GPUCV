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
#include <cxcoregcu/cxcoregcu.h>

#if _GPUCV_COMPILE_CUDA
#include <vector_types.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>

using namespace GCV;

/*===========================================================================
CudaDFT impmemented using FFT Algorithm and cufft library
============================================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU void gcuDFT(CvArr* src1,CvArr* dst,int flags,int nr);


void  cvgCudaDFT(CvArr* src1,CvArr* dst,int flags,int nr)__GPUCV_THROW()
{
	GPUCV_START_OP(cvDFT(src1,dst,flags,nr),
		"cvgCudaDFT",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, "no input images src1!");
	GCV_OPER_ASSERT(dst,  "no destination image!");

	GCV_OPER_ASSERT(!(GetnChannels(src1) ==1 && GetnChannels(dst) ==1),  "At least one of source or destination must be 2 channels");
	GCV_OPER_COMPAT_ASSERT(GetGLDepth(src1) == GL_FLOAT || GetGLDepth(src1) == GL_DOUBLE,  "Input image must be float or double");
	GCV_OPER_COMPAT_ASSERT(GetGLDepth(dst) == GL_FLOAT || GetGLDepth(dst) == GL_DOUBLE,  "Destination image must be float or double");

	gcuDFT((IplImage*)src1,(IplImage*) dst,flags,nr);

	GPUCV_STOP_OP(cvDFT(src1,dst,flags,nr),
		src1,dst,NULL,NULL
		);
}
#endif//_GPUCV_COMPILE_CUDA
