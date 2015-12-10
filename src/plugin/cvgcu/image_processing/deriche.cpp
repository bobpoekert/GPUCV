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
// Required to include CUDA vector type

#include <vector_types.h>
#include <SugoiTracer/tracer_c.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCV/cv_new.h>

using namespace GCV;


_GPUCV_CVGCU_EXPORT_CU void gcuDeriche(CvArr* src, CvArr* dst, double alpha );

void cvgCudaDeriche(CvArr* src, CvArr* dst, double alpha )
{
	GPUCV_START_OP(cvDeriche(src, dst, alpha),
		"cvgCudaDeriche",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");

	GCV_OPER_COMPAT_ASSERT(GetnChannels(src)==1, "source must be 1 channels!");
	GCV_OPER_COMPAT_ASSERT(GetnChannels(dst)==1, "destination must be 1 channels!");
	GCV_OPER_COMPAT_ASSERT(GetGLDepth(dst)==GL_FLOAT, "destination must be 32 float!");
	GCV_OPER_COMPAT_ASSERT(GetWidth(dst)==GetHeight(dst), "Does not works on rectangle images, must be squares!");
	GCV_OPER_COMPAT_ASSERT(GetWidth(dst)>=64,	"Does not works on image width < 64!");
	GCV_OPER_COMPAT_ASSERT(GetHeight(dst)>=64,	"Does not works on image height < 64!");
	gcuDeriche(src, dst, alpha);

	GPUCV_STOP_OP(
		cvDeriche(src, dst, alpha),
		src, dst, NULL, NULL
		);
}
#endif//_GPUCV_COMPILE_CUDA
