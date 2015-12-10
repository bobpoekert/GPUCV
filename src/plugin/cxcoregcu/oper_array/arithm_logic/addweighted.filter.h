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
#ifndef __GPUCV_CUDA_CXCOREGCU_ADDWEIGHTED_H
#define __GPUCV_CUDA_CXCOREGCU_ADDWEIGHTED_H

#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>

template <int TPLChannels,typename TPLSrc, typename TPLDst,typename TPLDataSrcFULL,typename TPLDataDstFULL>
__GCU_FCT_GLOBAL 
void gcudaKernel_AddWeighted(TPLDataSrcFULL *src1,float4 sk1
						  ,TPLDataSrcFULL *src2,float4 sk2
						  ,float4 kgamma
						  ,TPLDataDstFULL *dst
						  ,uint width
						  ,uint height
						  ,uint pitch)

{	
	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;
	//Pitch does not seems to work here...
	//unsigned int _IndexOut= __mul24(pitch, xBlock + threadIdx.y) + yBlock + threadIdx.x;
	unsigned int _IndexOut= __mul24(yIndex, width) + xIndex;

	if (xIndex < width && yIndex < height)
	{
		float4 TempVal1, TempVal2;
		TPLDataDstFULL TemPest;
		GCU_OP_MULTIPLEXER <TPLChannels,KERNEL_ARITHM_OPER_MUL, float4, TPLDataSrcFULL, float4>::Do(TempVal1, src1[_IndexOut], sk1);
		GCU_OP_MULTIPLEXER <TPLChannels,KERNEL_ARITHM_OPER_MUL, float4, TPLDataSrcFULL, float4>::Do(TempVal2, src2[_IndexOut], sk2);
		//BUG? do not knwo why, but it need a 4th paramter..so we add a forth one with 0. values
		float4 EmptyFloat;
		EmptyFloat.x = EmptyFloat.y = EmptyFloat.z = EmptyFloat.w = 0.;
		
#if 0
		GCU_OP_MULTIPLEXER <TPLChannels,KERNEL_ARITHM_OPER_ADD, float4, float4,  float4, float4>::Do(TempVal2, TempVal2, TempVal1, kgamma);//,EmptyFloat/*??*/);
		GCU_OP_MULTIPLEXER <TPLChannels,KERNEL_ARITHM_OPER_CLAMP, TPLDataDstFULL, float4>::Do(TemPest, TempVal2);
#else
		GCU_MULTIPLEX_4(TPLChannels,KERNEL_ARITHM_OPER_ADD, (TempVal2, TempVal2, TempVal1, kgamma,EmptyFloat/*??*/), float4, float4, float4, float4, float4);

		GCU_MULTIPLEX_1(TPLChannels,KERNEL_ARITHM_OPER_CLAMP, (TemPest, TempVal2),TPLDataDstFULL, float4);
#endif
		dst[_IndexOut] = TemPest;
	}
}
#endif // __GPUCV_CUDA_CXCOREGCU_ADDWEIGHTED_H
