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

#ifndef __GPUCV_CUDA_LUT_KERNEL_H
#define __GPUCV_CUDA_LUT_KERNEL_H

#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>
#include <GPUCVCuda/base_kernels/cxcoregcu_arithm_fct.kernel.h>




template <int TPLChannels,typename TPLSrc, typename TPLDst, typename TPLDataSrcFULL, typename TPLDataDstFULL, int block_width,int block_height>
__GCU_FCT_GLOBAL  
void gcuLutKernel(TPLDataSrcFULL *src1,TPLDataDstFULL *dst,TPLDst *lut,unsigned int width,
				  unsigned int height,
				  unsigned int pitch,int delta)

{	
	__shared__ TPLDst lutSharedBlock[256];//Lut is same type as destination
//	__shared__ TPLDst dst_block[16*16];

	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;
	unsigned int _IndexOut= __mul24(width/*pitch*/, yBlock + threadIdx.y) + 
		xBlock + threadIdx.x;
	unsigned int index_block = __mul24(threadIdx.y, block_width) + threadIdx.x;

	int4 * IntBlockPtr = (int4 *)lutSharedBlock;
	int4 * IntLutPtr = (int4 *)lut;
//	int4 * Intdst_BlockPtr = (int4 *)dst_block;
//	int4 * IntdstPtr = (int4 *)dst;

	//Load into shared memory
	unsigned int index_block_B = (__mul24(threadIdx.y, block_width) + threadIdx.x)*sizeof(TPLDst);
//	unsigned int _IndexOut_B  = (__mul24(pitch, yBlock + threadIdx.y) + xBlock + threadIdx.x)*sizeof(TPLDataSrcFULL);

	unsigned int index_block_IT4	= index_block_B/sizeof(int4);
//	unsigned int index_in_IT4		= _IndexOut_B/sizeof(int4);

	if (index_block < 256/*/sizeof(int)*sizeof(TPLDst)*/)
	{
		IntBlockPtr[index_block_IT4] = IntLutPtr[index_block_IT4];
	}
	__syncthreads();

	if (xIndex < width && yIndex < height)
	{
		TPLDataSrcFULL px1 = src1[_IndexOut];
		TPLSrc * pLocalSrcPixel = (TPLSrc*)&px1;
		TPLDataDstFULL DstPixel;
		TPLDst * pLocalDstPixel = (TPLDst*)&DstPixel;
		
		pLocalDstPixel[0]		= lutSharedBlock[((uint)(pLocalSrcPixel[0]))+ delta];
		if(TPLChannels>1)			
			pLocalDstPixel[1]	= lutSharedBlock[((uint)(pLocalSrcPixel[1]))+ delta];
		if(TPLChannels>2)			
			pLocalDstPixel[2]	= lutSharedBlock[((uint)(pLocalSrcPixel[2]))+ delta];
		if(TPLChannels>3)			
			pLocalDstPixel[3]	= lutSharedBlock[((uint)(pLocalSrcPixel[3]))+ delta];
	
		dst[_IndexOut] = DstPixel;
	}
}
#endif

