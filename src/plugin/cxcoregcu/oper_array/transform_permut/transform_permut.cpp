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

//=======================================================
_GPUCV_CXCOREGCU_EXPORT_CU void gcuSplit(CvArr* src, CvArr* dst0 = NULL, CvArr* dst1 = NULL, CvArr* dst2= NULL, CvArr* dst3= NULL);

void cvgCudaSplit(  CvArr* src, CvArr* dst0, CvArr* dst1, CvArr* dst2, CvArr* dst3 )__GPUCV_THROW()
{
	GPUCV_START_OP(cvSplit(src, dst0, dst1, dst2, dst3),
		"cvgCudaSplit", 
		dst0,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src, "no input images src!");
	GCV_OPER_ASSERT(dst0 || dst1 || dst2 || dst3, "GCV_OPER_ASSERT, no destination image!");

//	GCU_Assert(0,"cvgCudaSplit() not functionnal under linux yet");
	gcuSplit(src, dst0, dst1, dst2, dst3);

	GPUCV_STOP_OP(
		cvSplit(src, dst0, dst1, dst2, dst3),
		src, dst0,dst1, dst2
		);
}
//=======================================================
_GPUCV_CXCOREGCU_EXPORT_CU void gcuMerge(CvArr* src0, CvArr* src1, CvArr* src2, CvArr* src3, CvArr* dst );

void cvgCudaMerge(CvArr* src0, CvArr* src1, CvArr* src2, CvArr* src3, CvArr* dst )__GPUCV_THROW()
{
	GPUCV_START_OP(cvMerge(src0, src1, src2, src3, dst),
		"cvgCudaMerge", 
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src0 || src1 || src2 || src3, "no input images src!");
	GCV_OPER_ASSERT(dst, "no destination image!");

	gcuMerge(src0, src1, src2, src3, dst);

	GPUCV_STOP_OP(
		cvMerge(src0, src1, src2, src3, dst),
		src0, src1,src2, dst
		);
}
//=======================================================



//-------------------------------------------
_GPUCV_CXCOREGCU_EXPORT_CU 	void gcuFlip(CvArr* src, CvArr* dst, int flip_mode/*=0*/);

void cvgCudaFlip(CvArr* src, CvArr* dst, int flip_mode/*=0*/)__GPUCV_THROW()
{
	GPUCV_START_OP(cvFlip(src, dst, flip_mode),
		"cvgCudaFlip", 
		dst,
		GenericGPU::HRD_PRF_CUDA);


	GCV_OPER_ASSERT(src, "no input images src!");
	GCV_OPER_ASSERT(dst, "no destination image!");

	gcuFlip (src,dst, flip_mode);

	//cvgSetOptions(dst, DataContainer::DEST_IMG, false);

	GPUCV_STOP_OP(
		cvFlip(src, dst, flip_mode),
		src, dst, NULL, NULL
		);
}

#endif
