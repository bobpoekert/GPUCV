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
#include <cxcoregcu/config.h>
#include <cxcoregcu/cxcoregcu.h>
#include <GPUCV/misc.h>

using namespace GCV;


#if 1//_GPUCV_COMPILE_CUDA

_GPUCV_CXCOREGCU_EXPORT_CU
void gcuPow(CvArr* src1, CvArr* dst,double power);

void  cvgCudaPow(CvArr* src1, CvArr* dst,double power)__GPUCV_THROW()
{
	GPUCV_START_OP(cvPow(src1,dst,power),
		"cvgCudaPow",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, "no input images src1!");
	GCV_OPER_ASSERT(dst,	"no destination image!");
	//unsigned int a=GetDepth(src1);
	//GCU_Assert((a==IPL_DEPTH_8S) ||(a==IPL_DEPTH_8U), "cvgCudaLUT(), the source image must be 8U or 8S");

	gcuPow(src1, dst, power);

	GPUCV_STOP_OP(
		cvPow(src1,dst,power),
		src1,dst,NULL,NULL
		);
}
#endif

