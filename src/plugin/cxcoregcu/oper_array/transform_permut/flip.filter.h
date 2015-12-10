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
#ifndef __GPUCV_CUDA_CXCOREGCU_ARRAY_TRANSFORM_FLIP_H
#define __GPUCV_CUDA_CXCOREGCU_ARRAY_TRANSFORM_FLIP_H

#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/base_kernels/cxcoregcu_arithm_fct.kernel.h>

#if 1
/* Flip with Cuda
* Device code.
*/
#define BLOCK_DIM 16
#define USE_SHARED 0 
//!\todo Ecplore using Shared memory?
template < typename TPLDataDstFULL,typename TPLDataSrcFULL>
__global__ void gcudaKernel_Flip(TPLDataSrcFULL* srcArr, TPLDataDstFULL* dstArr, int src_width, int src_height, int flip_mode=0)
{
	GCUDA_KRNL_DBG_FIRST_THREAD("gcuFlipKernel", int a=0;)
	unsigned int xIndex = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	unsigned int yIndex = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
	unsigned int index_in = yIndex * src_width + xIndex;
	unsigned int index_out = 0;
	//We write the flip matrix in each case
	if((xIndex < src_width) && (yIndex <src_height))
	{
		if(flip_mode==0)
		{
			yIndex = src_height - yIndex;
			index_out = yIndex * src_width + xIndex;
		}

		if(flip_mode==1)
		{
			xIndex = src_width - xIndex;
			index_out = yIndex * src_width + xIndex;
		}

		if(flip_mode==-1)
		{
			yIndex = src_height - yIndex;
			xIndex = src_width - xIndex;
			index_out = yIndex * src_width + xIndex;
		}

		dstArr[index_out] = srcArr[index_in];			
	}       
}
#endif
#endif // _FLIP_KERNEL_H_
