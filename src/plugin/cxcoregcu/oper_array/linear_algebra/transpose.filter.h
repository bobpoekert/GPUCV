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
#ifndef __GPUCV_CUDA_CXCOREGCU_ARRAY_TRANSFORM_TRANSPOSE_H
#define __GPUCV_CUDA_CXCOREGCU_ARRAY_TRANSFORM_TRANSPOSE_H

#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/base_kernels/cxcoregcu_arithm_fct.kernel.h>


template <typename TPLSrcFULL, typename TPLDstFULL, int channels, int block_width, int block_height>
__GCU_FCT_GLOBAL 
void gcuTransposeKernel_Shared(TPLSrcFULL *in,
						TPLDstFULL *out,
						int width,
						int height)
{
	GCUDA_KRNL_DBG_FIRST_THREAD("gcuTransposeKernel", int a=0;);
	__shared__ TPLSrcFULL block[block_width*block_height];

	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);	
	unsigned int xIndex_in = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	unsigned int yIndex_in = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
	
	//transposed index
	unsigned int xIndex_out = __mul24(blockDim.y, blockIdx.y) + threadIdx.x;
	unsigned int yIndex_out = __mul24(blockDim.x, blockIdx.x) + threadIdx.y;
	
	unsigned int index_block;
	//write input block to shared block using transpose X/Y
	if (xIndex_in < width && yIndex_in < height)
	{
		// load block into shared mem
		index_block = __mul24(block_height,threadIdx.x) + threadIdx.y;
		unsigned int index_in = __mul24(width, yIndex_in) + xIndex_in;		
		block[index_block] = in[index_in];
	}

	__syncthreads();

	if (xIndex_in < width && yIndex_in < height)
	{//we just copy the block to a new location
		unsigned int index_block = __mul24(block_width,threadIdx.y) + threadIdx.x;
		unsigned int index_out = __mul24(width, yIndex_out) + xIndex_out;			
		out[index_out] = block[index_block];
	}
	GCUDA_KRNL_DBG_LAST_THREAD("gcuTransposeKernel", int a=0;);
}


template <typename TPLSrcFULL, typename TPLDstFULL, int channels>
__GCU_FCT_GLOBAL 
void gcuTransposeKernel(TPLSrcFULL *in,
						TPLDstFULL *out,
						int width,
						int height)
{
	GCUDA_KRNL_DBG_FIRST_THREAD("gcuTransposeKernel", int a=0;);

	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;
	unsigned int indexOut = __mul24(height, xIndex) + yIndex;
	unsigned int indexIn = __mul24(width, yIndex) + xIndex;
	if (xIndex < width && yIndex < height)
	{
		out[indexOut] = in[indexIn];			 
	}
	GCUDA_KRNL_DBG_LAST_THREAD("gcuTransposeKernel", int a=0;);
}

#endif // __GPUCV_CUDA_CXCOREGCU_ARRAY_TRANSFORM_TRANSPOSE_H
