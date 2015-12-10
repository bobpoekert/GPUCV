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
#ifndef __GPUCV_CUDA_CXCOREGCU_ARRAY_TRANSFORM_MERGE_H
#define __GPUCV_CUDA_CXCOREGCU_ARRAY_TRANSFORM_MERGE_H

#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/base_kernels/cxcoregcu_arithm_fct.kernel.h>

//! \note Inspired from cudpp deinterleaveRGBA8toFloat32()

/** \brief GpuCV CUDA Kernel to perform channel merge from several single channel images(1-4) to one multichannel image.
\note Template mechanism allow data type conversion on the fly.
\author Yannick Allusse
*/
template <int TPLchannels, typename TPLDataDst, typename TPLDataSrc, typename TPLDataDstFULL>
__GCU_FCT_GLOBAL
void gcuMergeKernel(
					TPLDataDstFULL * _dst,
					TPLDataSrc * _src0,	
					TPLDataSrc * _src1,
					TPLDataSrc * _src2,
					TPLDataSrc * _src3,
					unsigned int _width,
					unsigned int _height,		
					unsigned int _pitch,
					float _scale)
{
	GCUDA_KRNL_DBG_FIRST_THREAD("gcuMergeKernel", int a=0;)
	unsigned int xIndex = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	unsigned int yIndex = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

	if (xIndex < _width && yIndex < _height)
	{
		unsigned int indexOut  = __mul24(__mul24(blockDim.x, gridDim.x), yIndex) + xIndex;
		TPLDataDstFULL pixel;
		TPLDataDst * pLocalPixel = (TPLDataDst*)&pixel;
	
		pLocalPixel[0] = (_src0)?_src0[indexOut]:0;// *_scale;
		if(TPLchannels>1)				
			pLocalPixel[1] = (_src1)?_src1[indexOut]:0;// *_scale;
		if  (TPLchannels>2)
			pLocalPixel[2] = (_src2)?_src2[indexOut]:0;// *_scale;
		if  (TPLchannels>3)
			pLocalPixel[3] = (_src3)?_src3[indexOut]:0;// *_scale;
		_dst[indexOut] = pixel;
	}                       	
	GCUDA_KRNL_DBG_LAST_THREAD("gcuMergeKernel", int a=0;)
}



#if 1
template <int TPLchannels, typename TPLDataDst, typename TPLDataSrc, typename TPLDataDstFULL>
__global__ static //__GCU_FCT_GLOBAL 
void gcuMergeKernel_SubImage(
							 TPLDataDstFULL * _dst,											\
							 TPLDataSrc * _src0,										\
							 TPLDataSrc * _src1,										\
							 TPLDataSrc * _src2,										\
							 TPLDataSrc * _src3,										\
							 unsigned int _src_width,										\
							 unsigned int _src_height,										\
							 unsigned int _dst_width,										\
							 unsigned int _dst_height,										\
							 TPLDataSrc		extra_val,
							 float _scale)												\
{																		\
	GCUDA_KRNL_DBG_FIRST_THREAD("gcuMergeKernel", int a=0;)\
	unsigned int xIndex = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
	unsigned int yIndex = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

	if (xIndex < _src_width && yIndex < _src_height)
	{
	//	unsigned int indexIn  = __mul24(_src_width/*__mul24(blockDim.x, gridDim.x)*/, yIndex) + xIndex;
	//	unsigned int indexOut = __mul24(_dst_width/*__mul24(blockDim.x, gridDim.x)*/,/*_pitch*/yIndex) + xIndex;
		unsigned int indexOut  = __mul24(__mul24(blockDim.x, gridDim.x), yIndex) + xIndex;
		TPLDataDstFULL pixel;
		TPLDataDst * pLocalPixel = (TPLDataDst*)&pixel;
	
		pLocalPixel[0] = (_src0)?_src0[indexOut]:0;// *_scale;
		if(TPLchannels>1)				
			pLocalPixel[1] = (_src1)?_src1[indexOut]:0;// *_scale;
		if  (TPLchannels>2)
			pLocalPixel[2] = (_src2)?_src2[indexOut]:0;// *_scale;
		if  (TPLchannels>3)
			pLocalPixel[3] = (_src3)?_src3[indexOut]:0;// *_scale;
		_dst[indexOut] = pixel;

	}         
	else if(xIndex < _dst_width && yIndex < _dst_height)
	{
		unsigned int indexOut  = __mul24(__mul24(blockDim.x, gridDim.x), yIndex) + xIndex;
		TPLDataDstFULL pixel;
		TPLDataDst * pLocalPixel = (TPLDataDst*)&pixel;
	
		pLocalPixel[0] = (_src0)?extra_val:0;// *_scale;
		if(TPLchannels>1)				
			pLocalPixel[1] = (_src1)?extra_val:0;// *_scale;
		if  (TPLchannels>2)
			pLocalPixel[2] = (_src2)?extra_val:0;// *_scale;
		if  (TPLchannels>3)
			pLocalPixel[3] = (_src3)?extra_val:0;// *_scale;
		_dst[indexOut] = pixel;
	}
	GCUDA_KRNL_DBG_LAST_THREAD("gcuMergeKernel", int a=0;)\
}
#endif
#endif // __GPUCV_CUDA_CXCOREGCU_ARRAY_TRANSFORM_MERGE_H
