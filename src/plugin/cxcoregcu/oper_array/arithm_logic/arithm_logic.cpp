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
#include <GPUCVCuda/base_kernels/cxcoregcu_arithm_fct.kernel.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>

GCULogicStruct		varLocalLogic;//!< Used only for template specialization, does not contain any data.
GCUArithmStruct		varLocalArithm;//!< Used only for template specialization, does not contain any data.


using namespace GCV;


//Key tags:TUTO_CREATE_OP_CUDA__STP1__LAUNCHER_A
/*=========================================================================
LUT
=========================================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU	void gcuLut(CvArr* src1, CvArr* dst,CvArr* lut);

void  cvgCudaLUT(CvArr* src1, CvArr* dst, CvArr* lut)__GPUCV_THROW()
{
	GPUCV_START_OP(cvLUT(src1,dst,lut),
		"cvgCudaLUT",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(dst,	"No destination image!");
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src1)==IPL_DEPTH_8U, "The source image must be 8U");

	gcuLut(src1, dst, lut);

	GPUCV_STOP_OP(
		cvLUT(src1,dst,lut),
		src1,dst,NULL,NULL
		);
}
/*=========================================================================
CONVERTSCALE
=========================================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU void gcuConvertScale(CvArr* src1,CvArr* dst,double s1,double s2);

void  cvgCudaConvertScale(CvArr* src1,CvArr* dst,double s1,double s2)__GPUCV_THROW()
{
	GPUCV_START_OP(cvConvertScale(src1,dst,s1,s2),
		"cvgCudaConvertScale",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, "no input images src1!");
	GCV_OPER_ASSERT(dst,  "no destination image!");

	gcuConvertScale(src1,dst,s1,s2);

	GPUCV_STOP_OP(
		cvConvertScale(src1,dst,s1,s2),
		src1,dst,NULL,NULL
		);
}
/*=========================================================================
ADD, ADDS
=========================================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU void gcuAdd(CvArr* src1,CvArr* src2, CvArr* dst, CvArr * mask);
_GPUCV_CXCOREGCU_EXPORT_CU void gcuAddS(CvArr* src1,CvScalar scalar, CvArr* dst, CvArr * mask);
_GPUCV_CXCOREGCU_EXPORT_CU void gcuAddWeighted_f4(CvArr* src1,float4 s1,CvArr* src2,float4 s2, float4 gamma,CvArr* dst);
_GPUCV_CXCOREGCU_EXPORT_CU void gcuAddWeighted_d(CvArr* src1,double s1,CvArr* src2,double s2, double gamma,CvArr* dst);
_GPUCV_CXCOREGCU_EXPORT_CU void gcuAddWeighted_scalar(CvArr* src1,gcuScalar s1,CvArr* src2,gcuScalar s2, gcuScalar gamma,CvArr* dst);


void cvgCudaAdd(CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask)__GPUCV_THROW()
{
	GPUCV_START_OP(cvAdd(src1, src2, dst, mask),
		"cvgCudaAdd",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	gcuAdd(src1, src2, dst, mask);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, src2,dst, mask
		);
}


void cvgCudaAddS(CvArr* src1, CvScalar scalar, CvArr* dst, CvArr* mask)__GPUCV_THROW()
{
	GPUCV_START_OP(cvAdd(src1,NULL, dst, mask),
		"cvgCudaAddS",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(dst,	"No destination image!");


	gcuAddS(src1, scalar, dst, mask);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, NULL,dst, mask
		);
}


void  cvgCudaAddWeighted(CvArr* src1,double s1,CvArr* src2,double s2,double gamma,CvArr* dst)__GPUCV_THROW()
{
	GPUCV_START_OP(cvAddWeighted(src1,s1,src2,s2,gamma,dst),
		"cvgCudaAdd",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	unsigned int a=GetCVDepth(src1);
	unsigned int b=GetCVDepth(src2);
	unsigned int c=GetCVDepth(dst);
	GCV_OPER_ASSERT((a==b) && (b==c), "cvgCudaAddWeighted(), the images must be same");

	gcuAddWeighted_d(src1,s1,src2,s2,gamma,dst);

	GPUCV_STOP_OP(
		cvAddWeighted(src1,s1,src2,s2,gamma,dst),
		src1,src2,dst,NULL
		);
}


/*=========================================================================
 SUB, SUBS, SUBRS
=========================================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU void gcuSub(CvArr* src1,CvArr* src2, CvArr* dst, CvArr * mask);
_GPUCV_CXCOREGCU_EXPORT_CU void gcuSubS(CvArr* src1,CvScalar scalar, CvArr* dst, CvArr * mask);
_GPUCV_CXCOREGCU_EXPORT_CU void gcuSubRS(CvArr* src1,CvScalar scalar, CvArr* dst, CvArr * mask);

void cvgCudaSub(CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask)__GPUCV_THROW()
{
	GPUCV_START_OP(cvSub(src1, src2, dst, mask),
		"cvgCudaSub",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	gcuSub(src1, src2, dst, mask);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, src2,dst, mask
		);
}

void cvgCudaSubS(CvArr* src1, CvScalar scalar, CvArr* dst, CvArr* mask)__GPUCV_THROW()
{
	GPUCV_START_OP(cvSub(src1, NULL, dst, mask),
		"cvgCudaSubS",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	gcuSubS(src1, scalar, dst, mask);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, NULL,dst, mask
		);
}

void cvgCudaSubRS(CvArr* src1, CvScalar scalar, CvArr* dst, CvArr* mask)__GPUCV_THROW()
{
	GPUCV_START_OP(cvSubRS(src1, scalar, dst, mask),
		"cvgCudaSubRS",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	gcuSubRS(src1, scalar, dst, mask);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, NULL,dst, mask
		);
}

/*=========================================================================
MUL, DIV
=========================================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU void gcuMul(CvArr* src1,CvArr* src2, CvArr* dst, double scale);
_GPUCV_CXCOREGCU_EXPORT_CU void gcuDiv(CvArr* src1,CvArr* src2, CvArr* dst, double scale);

void cvgCudaMul(CvArr* src1, CvArr* src2, CvArr* dst, double scale)__GPUCV_THROW()
{
	GPUCV_START_OP(cvMul(src1, src2, dst, scale),
		"cvgCudaMul",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	gcuMul(src1, src2, dst, scale);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, src2,dst,NULL
		);
}

void cvgCudaDiv(CvArr* src1, CvArr* src2, CvArr* dst, double scale)__GPUCV_THROW()
{
	GPUCV_START_OP(cvDiv(src1, src2, dst, scale),
		"cvgCudaDiv",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	gcuDiv(src1, src2, dst, scale);

	GPUCV_STOP_OP(
		,//cvAdd(src1, src2, dst, mask),
		src1, src2,dst,NULL
		);
}

/*=========================================================================
MIN, MINS
=========================================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU void gcuMin(CvArr* src1,CvArr* src2, CvArr* dst);
_GPUCV_CXCOREGCU_EXPORT_CU void gcuMinS(CvArr* src1,double val, CvArr* dst);

void cvgCudaMin(CvArr* src1,CvArr* src2, CvArr* dst)__GPUCV_THROW()
{
	GPUCV_START_OP(cvMin(src1,src2, dst),
		"cvgCudaMin",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	gcuMin(src1, src2, dst);

	GPUCV_STOP_OP(
		,//cvMin(src1, src2, dst, mask),
		src1, src2 ,dst, NULL
		);
}

void cvgCudaMinS(CvArr* src1,double val, CvArr* dst)__GPUCV_THROW()
{
	GPUCV_START_OP(cvMinS(src1,val, dst),
		"cvgCudaMin",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	gcuMinS(src1,val,dst);

	GPUCV_STOP_OP(
		,//cvMin(src1, src2, dst, mask),
		src1, NULL ,dst, NULL
		);
}
/*=========================================================================
MAX, MAXS
=========================================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU void gcuMax(CvArr* src1,CvArr* src2, CvArr* dst);
_GPUCV_CXCOREGCU_EXPORT_CU void gcuMaxS(CvArr* src1,double val, CvArr* dst);

void cvgCudaMax(CvArr* src1,CvArr* src2, CvArr* dst)__GPUCV_THROW()
{
	GPUCV_START_OP(cvMax(src1,src2, dst),
		"cvgCudaMax",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	gcuMax(src1, src2, dst);

	GPUCV_STOP_OP(
		,//cvMax(src1, src2, dst, mask),
		src1, src2 ,dst, NULL
		);
}

void cvgCudaMaxS(CvArr* src1,double val, CvArr* dst)__GPUCV_THROW()
{
	GPUCV_START_OP(cvMaxS(src1,val, dst),
		"cvgCudaMaxS",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	gcuMaxS(src1,val,dst);

	GPUCV_STOP_OP(
		,//cvMax(src1, src2, dst, mask),
		src1, NULL ,dst, NULL
		);
}
/*=========================================================================
ABS_DIFF, ABS_DIFFS
=========================================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU void gcuAbsDiff(CvArr* src1, CvArr* src2, CvArr* dst);
_GPUCV_CXCOREGCU_EXPORT_CU void gcuAbsDiffS(CvArr* src1, CvScalar val, CvArr* dst);

void cvgCudaAbsDiff(CvArr* src1,CvArr* src2, CvArr* dst)__GPUCV_THROW()
{
	GPUCV_START_OP(cvAbsDiff(src1,src2, dst),
		"cvgCudaAbsDiff",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	gcuAbsDiff(src1, src2, dst);

	GPUCV_STOP_OP(
		,//cvMax(src1, src2, dst, mask),
		src1, src2 ,dst, NULL
		);
}

void cvgCudaAbsDiffS(CvArr* src1, CvArr* dst, CvScalar val)__GPUCV_THROW()
{
	GPUCV_START_OP(cvAbsDiffS(src1,dst, val),
		"cvgCudaAbsDiffS",
		dst,
		GenericGPU::HRD_PRF_CUDA);


	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	gcuAbsDiffS(src1,val,dst);

	GPUCV_STOP_OP(
		,//cvMax(src1, src2, dst, mask),
		src1, NULL ,dst, NULL
		);
}
//=========================================================================

/*=========================================================================
COPY
=========================================================================*/
void cvgCudaCopy(CvArr* src1,
				 CvArr* dst,
				 CvArr* mask)__GPUCV_THROW()
{
	if(mask)
	{
		cvgCudaAddS(src1, cvScalarAll(0), dst, mask);
	}
	else
	{
		GPUCV_START_OP(cvAdd(src1,NULL, dst, mask),
			"cvgCudaAddS",
			dst,
			GenericGPU::HRD_PRF_CUDA);

		GCV_OPER_ASSERT(src1, 	"No input images src1!");
		GCV_OPER_ASSERT(dst,	"No destination image!");

		//get local objects
		CvgArr * gcvSrc1	= dynamic_cast<CvgArr*>(GPUCV_GET_TEX(src1));
		CvgArr * gcvDst		= dynamic_cast<CvgArr*>(GPUCV_GET_TEX(dst));

		//copy only last active Data from gcvSrc1
		gcvDst->_CopyActiveDataDsc(*gcvSrc1, true);

		GPUCV_STOP_OP(
			,//cvAdd(src1, src2, dst, mask),
			src1, NULL,dst, mask
			);
	}
}

/*=========================================================================
CMP, CMPS
=========================================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU
void gcuCmpAll(CvArr* src1, CvArr* src2,CvArr* dst,int cmp_op, float4 * value);

void  cvgCudaCmp(CvArr* src1,CvArr* src2, CvArr* dst,int cmp_op)__GPUCV_THROW()
{
	GPUCV_START_OP(cvCmp(src1,src2,dst,cmp_op),
		"cvgCudaCmp",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");
	unsigned int a=GetnChannels(src1);
	unsigned int b=GetnChannels(src2);
	GCV_OPER_ASSERT((a==b)& (a==1) , "cvgCudaCmp(), the images must have same no. of channels");

	//!\todo Fix cvgCudaCmp() with 32F input
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_32F, "32F format not managed");
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_64F, "64F format not managed");

	GCV_OPER_TODO_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_16U, "16U is not working yet");

	gcuCmpAll(src1, src2, dst ,cmp_op, NULL);

	GPUCV_STOP_OP(
		cvCmp(src1,src2,dst,cmp_op),
		src1,src2,dst,NULL
		);
}

void  cvgCudaCmpS(CvArr* src1,double value, CvArr* dst,int cmp_op)__GPUCV_THROW()
{
	GPUCV_START_OP(cvCmpS(src1,value,dst,cmp_op),
		"cvgCudaCmp",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(dst,	"No destination image!");
	unsigned int a=GetnChannels(src1);
	//!\todo Fix cvgCudaCmpS() with 32F input
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_32F, "32F format not managed");
	GCV_OPER_COMPAT_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_64F, "64F format not managed");
	GCV_OPER_TODO_ASSERT(GetCVDepth(src1)!=IPL_DEPTH_16U, "16U is not working yet");

	float4 TempScalar;
	TempScalar.x =	value;
	TempScalar.y =	value;
	TempScalar.z =	value;
	TempScalar.w =	value;

 	gcuCmpAll(src1, NULL, dst ,cmp_op, &TempScalar);

	GPUCV_STOP_OP(
		cvCmpS(src1,value,dst,cmp_op),
		src1,NULL,dst,NULL
		);
}
#endif
