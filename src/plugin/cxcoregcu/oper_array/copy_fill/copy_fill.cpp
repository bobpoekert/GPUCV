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

using namespace GCV;

_GPUCV_CXCOREGCU_EXPORT_CU
void gcuSet(CvArr* src, gcuScalar &scalar,CvArr* mask=NULL);
//====================================================
void cvgCudaSetZero( CvArr* arr )
{
	GPUCV_START_OP(cvSetZero(arr),
		"cvgCudaSetZero",
		arr,
		GenericGPU::HRD_PRF_4);

		gcuScalar cudavalue;
		cudavalue.val[0]=cudavalue.val[1]=cudavalue.val[2]=cudavalue.val[3]=0;

		gcuSet(arr, cudavalue);

	GPUCV_STOP_OP(
		cvSetZero(arr),
		arr, NULL, NULL, NULL
		);
}
//====================================================
void cvgCudaSet( CvArr* arr,  CvScalar value, CvArr* mask/*=NULL*/)
{
	GPUCV_START_OP(cvSet(arr,value, mask),
		"cvgCudaSet",
		arr,
		GenericGPU::HRD_PRF_4);

		gcuScalar cudavalue;
		cudavalue.val[0]=value.val[0];
		cudavalue.val[1]=value.val[1];
		cudavalue.val[2]=value.val[2];
		cudavalue.val[3]=value.val[3];
		gcuSet(arr, cudavalue, mask);

	GPUCV_STOP_OP(
		cvSet(arr,value, mask),
		arr, NULL, NULL, NULL
		);
}
//====================================================
#endif
