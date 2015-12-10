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
#include <cxcoregcu/oper_array/arithm_logic/arithm_logic.h>
#if _GPUCV_COMPILE_CUDA

_GPUCV_CXCOREGCU_EXPORT_CU  
void gcuCmpAll(CvArr* src1,CvArr* src2, CvArr* dst,int op, float4 * dbl_value)
{
	switch (op)
	{
		case 0: CudaArithm_SwitchCHANNELS<KERNEL_LOGIC_OPER_EQUAL, GCULogicStruct>			(&varLocalLogic, src1, src2, dst, NULL, 255.0, dbl_value);	break;
		case 1: CudaArithm_SwitchCHANNELS<KERNEL_LOGIC_OPER_GREATER, GCULogicStruct>		(&varLocalLogic, src1, src2, dst, NULL, 255.0, dbl_value);	break;
		case 2: CudaArithm_SwitchCHANNELS<KERNEL_LOGIC_OPER_GREATER_OR_EQUAL, GCULogicStruct>(&varLocalLogic, src1, src2, dst, NULL, 255.0, dbl_value);	break;
		case 3: CudaArithm_SwitchCHANNELS<KERNEL_LOGIC_OPER_LESS, GCULogicStruct>			(&varLocalLogic, src1, src2, dst, NULL, 255.0, dbl_value);	break;
		case 4: CudaArithm_SwitchCHANNELS<KERNEL_LOGIC_OPER_LESS_OR_EQUAL, GCULogicStruct>	(&varLocalLogic, src1, src2, dst, NULL, 255.0, dbl_value);	break;
		case 5: CudaArithm_SwitchCHANNELS<KERNEL_LOGIC_OPER_NOT_EQUAL, GCULogicStruct>		(&varLocalLogic, src1, src2, dst, NULL, 255.0, dbl_value);	break;
	}		

}
#endif