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
#ifndef __GPUCV_CUDA_CXCOREGCU_COPY_FILL_SET_H
#define __GPUCV_CUDA_CXCOREGCU_COPY_FILL_SET_H

#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/base_kernels/cxcoregcu_arithm_fct.kernel.h>

/** \brief GpuCV CUDA Kernel to perform Set() operator.
*/

template <int TPLchannels, typename TPLDataDst>
__global__ static //__GCU_FCT_GLOBAL 
void gcuSetKernel_Mask(								
					TPLDataDst * _dst,
					char1 * _mask,
					unsigned int _width,
					unsigned int _height,
					float4 _scalar1
					,float4 _scalar2
					)
{																		
	GCUDA_KRNL_DBG_FIRST_THREAD("gcuSetKernel", int a=0;)
	unsigned int xIndex = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	unsigned int yIndex = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
	TPLDataDst TempDst;

	if (xIndex < _width && yIndex < _height)
	{
		unsigned int indexIn  = __mul24(__mul24(blockDim.x, gridDim.x), yIndex) + xIndex;
		unsigned int indexOut = __mul24(__mul24(blockDim.x, gridDim.x),/*_pitch*/yIndex) + xIndex;

		if(_mask[indexIn].x!=0)
			GCU_OP_MULTIPLEXER<TPLchannels,StAffectFilterTex,TPLDataDst,float4>::Do(TempDst,_scalar1);
		else
			GCU_OP_MULTIPLEXER<TPLchannels,StAffectFilterTex,TPLDataDst,float4>::Do(TempDst,_scalar2);
			
		_dst[indexOut] = TempDst;
	}                       																	
	GCUDA_KRNL_DBG_LAST_THREAD("gcuSetKernel", int a=0;)
}

template <int TPLchannels, typename TPLDataDst>
__global__ static //__GCU_FCT_GLOBAL 
void gcuSetKernel(								
					TPLDataDst * _dst,
					unsigned int _width,
					unsigned int _height,
					float4 _scalar)
{																		
	GCUDA_KRNL_DBG_FIRST_THREAD("gcuSetKernel", int a=0;)
	unsigned int xIndex = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	unsigned int yIndex = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
	TPLDataDst TempDst;
	if (xIndex < _width && yIndex < _height)
	{
		unsigned int indexIn  = __mul24(__mul24(blockDim.x, gridDim.x), yIndex) + xIndex;
		unsigned int indexOut = __mul24(__mul24(blockDim.x, gridDim.x),/*_pitch*/yIndex) + xIndex;
		GCU_OP_MULTIPLEXER<TPLchannels,StAffectFilterTex,TPLDataDst,float4>::Do(TempDst,_scalar);
		_dst[indexOut] = TempDst;
	}                       																	
	GCUDA_KRNL_DBG_LAST_THREAD("gcuSetKernel", int a=0;)
}
#endif // __GPUCV_CUDA_CXCOREGCU_ARRAY_TRANSFORM_SPLIT_H
