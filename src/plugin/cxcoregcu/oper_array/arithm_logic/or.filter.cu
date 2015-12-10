//CVG_LicenseBegin========================================== ====================
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
#include <cxcoregcu/oper_array/arithm_logic/arithm_logic.h>
#if _GPUCV_COMPILE_CUDA

_GPUCV_CXCOREGCU_EXPORT_CU
void gcuOr(CvArr* src1,CvArr* src2, CvArr* dst, CvArr * mask)
{
	CudaArithm_SwitchCHANNELS<KERNEL_LOGIC_OPER_OR, GCULogicStruct>(&varLocalLogic, src1, src2, dst, mask, 1.0, NULL);
} 

_GPUCV_CXCOREGCU_EXPORT_CU
void gcuOrS(CvArr* src1,gcuScalar scalar, CvArr* dst, CvArr * mask)
{
	float4 TempScalar;
	TempScalar.x = scalar.val[0]; 
	TempScalar.y = scalar.val[1];
	TempScalar.z = scalar.val[2];
	TempScalar.w = scalar.val[3];

	CudaArithm_SwitchCHANNELS<KERNEL_LOGIC_OPER_OR, GCULogicStruct>(&varLocalLogic, src1, NULL , dst, mask, 1.0, &TempScalar);
}
#endif


