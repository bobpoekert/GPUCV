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
#include "StdAfx.h"//ned twice here under VS2005..???
#include <cxcoregcu/cxcoregcu.h>

#if _GPUCV_COMPILE_CUDA
// Required to include CUDA vector types
#include <vector_types.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>


using namespace GCV;
//=========================================================================

/*============================================================
CUDA Sum
============================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU CvScalar gcuSumArr(CvArr* _arr);

CvScalar cvgCudaSum(CvArr* _arr)__GPUCV_THROW()
{
	CvScalar result;

	GPUCV_START_OP(cvSum(_arr),
		"cvgCudaSum", 
		_arr,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(_arr, "no input images _arr!");
	//cvgSetOptions(_arr, DataContainer::DEST_IMG, false);
	result = gcuSumArr(_arr);

	GPUCV_STOP_OP(
		cvSum(_arr),//in case if error this operator is called
		_arr, NULL, NULL, NULL //in case of error, we get this images back to CPU, so the opencv operator can be called
		);

	return result;
}
/*============================================================
CUDA avg
============================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU CvScalar gcuAvg(CvArr* arr,CvArr* mask);

CvScalar cvgCudaAvg(CvArr* arr,CvArr* mask/*=NULL*/)__GPUCV_THROW()
{
	CvScalar result;
	result.val[0]=result.val[1]=result.val[2]=result.val[3]=0;

	GPUCV_START_OP(cvAvg(arr, mask),//cvAdd(src1, src2, dst, mask),
		"cvgCudaAvg", 
		arr,
		GenericGPU::HRD_PRF_CUDA);


	GCV_OPER_ASSERT(!mask, "mask not supported!");
	cvgSetOptions(arr, DataContainer::DEST_IMG, false);
	//result = gcuAvg(arr, mask);


	GPUCV_STOP_OP(
		cvAvg(arr, mask),//cvAdd(src1, src2, dst, mask),//in case if error this operator is called
		arr, NULL, NULL, mask //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
	return result;
}

/*============================================================
CUDA GEMM
============================================================*/
_GPUCV_CXCOREGCU_EXPORT_CU 
void  gcuGEMM(CvArr* src1,
										 CvArr* src2, 
										 double alpha,
										 CvArr* src3, 
										 double beta, 
										 CvArr* dst, 
										 int tABC);

void  cvgCudaGEMM(CvArr* src1,
				  CvArr* src2, 
				  double alpha,
				  CvArr* src3, 
				  double beta, 
				  CvArr* dst, 
				  int tABC/*=0*/)__GPUCV_THROW()
{
	//CvScalar result;
	GPUCV_START_OP(cvGEMM(src1, src2, alpha, src3, beta, dst, tABC),//cvAdd(src1, src2, dst, mask),
		"cvgCudaGEMM", 
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src1, "no input image src1!");
	GCV_OPER_ASSERT(src2, "no input image src2!")
	GCV_OPER_ASSERT(dst, "no destination image dst!");

	//cublas use SRC3 has output.
	if (src3!=dst)
	{
		if(src3==NULL)
			cvgCudaSetZero(dst);
		else
			cvgCudaCopy(src3, dst);
	}

	gcuGEMM(src1, src2, alpha, src3, beta, dst, tABC);

	GPUCV_STOP_OP(
		cvGEMM(src1, src2, alpha, src3, beta, dst, tABC),//cvAdd(src1, src2, dst, mask),//in case if error this operator is called
		src1, src2, src3, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}

/*============================================================
CUDA ...
============================================================*/

#endif
