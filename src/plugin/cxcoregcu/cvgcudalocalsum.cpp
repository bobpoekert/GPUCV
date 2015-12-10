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


_GPUCV_CXCOREGCU_EXPORT_CU
void gcuLocalSum(CvArr* src1,CvArr* dst, int height , int width);

void  cvgCudaLocalSum(CvArr* src1,CvArr* dst, int height , int width)__GPUCV_THROW()
{
	GPUCV_START_OP(cvCloneImage((IplImage *) src1),
		"cvCloneImage",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, "no input images src1!");
	GCV_OPER_ASSERT(dst,  "no destination image!");
	unsigned int a=GetnChannels(src1);
	GCV_OPER_ASSERT((a==1),"the images must be single channeled " );

#ifdef _LINUX	
	GCV_OPER_COMPAT_ASSERT(0,  "not currently compatible with linux!");
#else
	gcuLocalSum(src1,dst ,height,width);
#endif


	GPUCV_STOP_OP(
		cvCloneImage((IplImage *) src1),
		src1,dst,NULL,NULL
		);
}
#endif


