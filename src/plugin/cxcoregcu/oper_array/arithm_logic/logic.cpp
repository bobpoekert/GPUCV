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

/** \brief Contains all logical CPP launcher
*/

#if _GPUCV_COMPILE_CUDA
#include <vector_types.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>

using namespace GCV;

/*============================================================
	AND & ANDS
============================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU void gcuAnd(CvArr* src1,CvArr* src2, CvArr* dst, CvArr * mask);

void cvgCudaAnd(CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask)__GPUCV_THROW()
{
	GPUCV_START_OP(cvAnd(src1, src2, dst, mask),
		"cvgCudaAnd",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_32F, "The source image must be integer values");
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src2)!=IPL_DEPTH_64F, "The source image must be integer values");

	gcuAnd(src1, src2, dst, mask);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, src2,dst, mask
		);
}
//=========================================================================
_GPUCV_CXCOREGCU_EXPORT_CU void gcuAndS(CvArr* src1,CvScalar scalar, CvArr* dst, CvArr * mask);

void cvgCudaAndS(CvArr* src1, CvScalar scalar, CvArr* dst, CvArr* mask)__GPUCV_THROW()
{
	GPUCV_START_OP(cvAndS(src1, scalar, dst, mask),
		"cvgCudaAndS",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(dst,	"No destination image!");
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_32F, "The source image must be integer values");

	gcuAndS(src1, scalar, dst, mask);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, NULL,dst, mask
		);
}
//======================================================================

/*============================================================
OR & ORS
============================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU void gcuOr(CvArr* src1,CvArr* src2, CvArr* dst, CvArr * mask);

void cvgCudaOr(CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask)__GPUCV_THROW()
{
	GPUCV_START_OP(cvOr(src1, src2, dst, mask),
		"cvgCudaOr",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_32F, "The source image must be integer values");
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_64F, "The source image must be integer values");

	gcuOr(src1, src2, dst, mask);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, src2,dst, mask
		);
}
//=========================================================================
_GPUCV_CXCOREGCU_EXPORT_CU void gcuOrS(CvArr* src1,CvScalar scalar, CvArr* dst, CvArr * mask);

void cvgCudaOrS(CvArr* src1, CvScalar scalar, CvArr* dst, CvArr* mask)__GPUCV_THROW()
{
	GPUCV_START_OP(cvOrS(src1, scalar, dst, mask),
		"cvgCudaOrS",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(dst,	"No destination image!");
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_32F, "The source image must be integer values");

	gcuOrS(src1, scalar, dst, mask);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, NULL,dst, mask
		);
}
//======================================================================


/*============================================================
XOR & XORS
============================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU void gcuXor(CvArr* src1,CvArr* src2, CvArr* dst, CvArr * mask);

void cvgCudaXor(CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask)__GPUCV_THROW()
{
	GPUCV_START_OP(cvXor(src1, src2, dst, mask),
		"cvgCudaXor",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_32F, "The source image must be integer values");
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src2)!=IPL_DEPTH_64F, "The source image must be integer values");

	gcuXor(src1, src2, dst, mask);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, src2,dst, mask
		);
}
//=========================================================================
_GPUCV_CXCOREGCU_EXPORT_CU void gcuXorS(CvArr* src1,CvScalar scalar, CvArr* dst, CvArr * mask);

void cvgCudaXorS(CvArr* src1, CvScalar scalar, CvArr* dst, CvArr* mask)__GPUCV_THROW()
{
	GPUCV_START_OP(cvXorS(src1, scalar, dst, mask),
		"cvgCudaXorS",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(dst,	"No destination image!");
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_32F, "The source image must be integer values");

	gcuXorS(src1, scalar, dst, mask);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, NULL,dst, mask
		);
}
//======================================================================
_GPUCV_CXCOREGCU_EXPORT_CU void gcuNot(CvArr* src1,CvArr* dst);

void cvgCudaNot(CvArr* src1, CvArr* dst)__GPUCV_THROW()
{
	GPUCV_START_OP(cvNot(src1, dst),
		"cvgCudaNot",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(dst,	"No destination image!");
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_32F, "The source image must be integer values");

	gcuNot(src1, dst);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, NULL,dst, NULL
		);
}
#endif
