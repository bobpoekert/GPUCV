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
#ifndef __GPUCV_CUDA_CXCOREGCU_ARRAY_TRANSFORM_SPLIT_H
#define __GPUCV_CUDA_CXCOREGCU_ARRAY_TRANSFORM_SPLIT_H

#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/base_kernels/cxcoregcu_arithm_fct.kernel.h>

/** \brief GpuCV CUDA Kernel to perform channel split from one multichannel image to several single channel images(1-4).
\note Template mechanism allow data type conversion on the fly.
\author Yannick Allusse
*/

template <int TPLchannels, typename TPLDataDst,typename TPLDataSrc, typename TPLDataSrcFULL>
__global__ static //__GCU_FCT_GLOBAL 
void gcuSplitKernel(								
					TPLDataSrc * _src,
					TPLDataDst * _dst0,	
					TPLDataDst * _dst1,	
					TPLDataDst * _dst2,	
					TPLDataDst * _dst3,
					unsigned int _width,
					unsigned int _height,
					int _pitch,
					float _scale)
{																		
	GCUDA_KRNL_DBG_FIRST_THREAD("gcuSplitKernel", int a=0;)
	unsigned int xIndex = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	unsigned int yIndex = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

	if (xIndex < _width && yIndex < _height)
	{
		unsigned int indexIn  = __mul24(__mul24(blockDim.x, gridDim.x), yIndex) + xIndex;
		unsigned int indexOut = __mul24(__mul24(blockDim.x, gridDim.x),/*_pitch*/yIndex) + xIndex;

			TPLDataSrcFULL pixel = *((TPLDataSrcFULL*)(&_src[indexIn*TPLchannels]));
			TPLDataSrc * pLocalPixel = (TPLDataSrc*)&pixel;
			
			if(_dst0)_dst0[indexOut] = pLocalPixel[0];// *_scale;
			if(TPLchannels>1)				
				if(_dst1)_dst1[indexOut] = pLocalPixel[1];// *_scale;
			if  (TPLchannels>2)
				if(_dst2)_dst2[indexOut] = pLocalPixel[2];// *_scale;
			if  (TPLchannels>3)
				if(_dst3)_dst3[indexOut] = pLocalPixel[3];// *_scale;
	}                       																	
	GCUDA_KRNL_DBG_LAST_THREAD("gcuSplitKernel", int a=0;)
}

#if 1
template <int TPLchannels, typename TPLDataDst,typename TPLDataSrc, typename TPLDataSrcFULL>\
__global__ static //__GCU_FCT_GLOBAL 
void gcuSplitKernel_SubImage(								\
							 TPLDataSrc * _src,											\
							 TPLDataDst * _dst0,										\
							 TPLDataDst * _dst1,										\
							 TPLDataDst * _dst2,										\
							 TPLDataDst * _dst3,										\
							 unsigned int _src_width,										\
							 unsigned int _src_height,										\
							 unsigned int _dst_width,										\
							 unsigned int _dst_height,										\
							 TPLDataDst		extra_val,
							 float _scale)												\
{																		\
GCUDA_KRNL_DBG_FIRST_THREAD("gcuSplitKernel", int a=0;)\
unsigned int xIndex = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
unsigned int yIndex = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

	//if index is in range of image src, do normal split
	if (xIndex < _src_width && yIndex < _src_height)
	{
		unsigned int indexIn  = __mul24(__mul24(blockDim.x, gridDim.x), yIndex) + xIndex;
		unsigned int indexOut = __mul24(__mul24(blockDim.x, gridDim.x),/*_pitch*/yIndex) + xIndex;

		TPLDataSrcFULL pixel = *((TPLDataSrcFULL*)(&_src[indexIn*TPLchannels]));
		TPLDataSrc * pLocalPixel = (TPLDataSrc*)&pixel;
		
		if(_dst0)_dst0[indexOut] = pLocalPixel[0];// *_scale;
		if(TPLchannels>1)				
			if(_dst1)_dst1[indexOut] = pLocalPixel[1];// *_scale;
		if  (TPLchannels>2)
			if(_dst2)_dst2[indexOut] = pLocalPixel[2];// *_scale;
		if  (TPLchannels>3)
			if(_dst3)_dst3[indexOut] = pLocalPixel[3];// *_scale;

		//unsigned int indexIn  = __mul24(_src_width/*__mul24(blockDim.x, gridDim.x)*/, yIndex) + xIndex;
		//unsigned int indexOut = __mul24(_dst_width/*__mul24(blockDim.x, gridDim.x)*/,/*_pitch*/yIndex) + xIndex;
	}
	else if(xIndex < _dst_width && yIndex < _dst_height)
	{
		unsigned int indexOut = __mul24(_dst_width/*__mul24(blockDim.x, gridDim.x)*/,/*_pitch*/yIndex) + xIndex;
		if(_dst0)_dst0[indexOut] = extra_val;// *_scale;
		if(_dst1)_dst1[indexOut] = extra_val;// *_scale;
		if(_dst2)_dst2[indexOut] = extra_val;// *_scale;
		if(_dst3)_dst3[indexOut] = extra_val;// *_scale;
	}
	GCUDA_KRNL_DBG_LAST_THREAD("gcuSplitKernel", int a=0;)\
}
#endif

#endif // __GPUCV_CUDA_CXCOREGCU_ARRAY_TRANSFORM_SPLIT_H
