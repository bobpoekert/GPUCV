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

//==========================================================
_GPUCV_CXCOREGCU_EXPORT_CU void gcuAddWeighted_scalar(CvArr* src1,gcuScalar s1,CvArr* src2,gcuScalar s2,gcuScalar gamma,CvArr* dst);

void  cvgCudaScaleAdd(CvArr* src1, CvScalar scale, CvArr* src2, CvArr* dst)__GPUCV_THROW()
{
	GPUCV_START_OP(cvTranspose(src, dst),
		"cvgCudaScaleAdd",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, "no input images src!");
	GCV_OPER_ASSERT(src2, "no input images src!");
	GCV_OPER_ASSERT(dst,  "no destination image!");
	gcuScalar scale1, scale2, gamma;
	for(int i=0;i<4;i++)
	{
		scale1.val[i]	= scale.val[i];
		scale2.val[i]	=	1;
		gamma.val[i]	=	0;
	}
	gcuAddWeighted_scalar(src1, scale1, src2, scale2, gamma, dst);//here we recycle another kernel...
	GPUCV_STOP_OP(
		cvTranspose(src, dst),
		src1, src2,dst, NULL
		);
}
//==========================================================
_GPUCV_CXCOREGCU_EXPORT_CU void gcuTranspose(CvArr* src, CvArr* dst);

void  cvgCudaTranspose(CvArr* src, CvArr* dst)__GPUCV_THROW()
{
	GPUCV_START_OP(cvTranspose(src, dst),
		"cvgCudaTranspose",
		dst,
		GenericGPU::HRD_PRF_CUDA);

		GCV_OPER_ASSERT(src, "no input images src!");
		GCV_OPER_ASSERT(dst, "no destination image!");
		gcuTranspose((CvArr*)src, (CvArr*)dst);
	
	GPUCV_STOP_OP(
		cvTranspose(src, dst),
		src, NULL,dst, NULL
		);
}
//==========================================================
#endif
