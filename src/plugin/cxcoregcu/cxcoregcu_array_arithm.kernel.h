//CVG_LicenseBegin==============================================================
//
//	Copyright@ GET 2005 (Groupe des Ecoles de Telecom)
//		http://www.get-telecom.fr/
//
//	This software is a GPU accelerated library for computer-vision. It
//	supports an OPENCV-like extensible interface for easily porting OPENCV
//	applications.
//
//	Contacts:
//		GpuCV core team: gpucv@picoforge.int-evry.fr
//		GpuCV developers newsgroup: gpucv-developers@picoforge.int-evry.fr
//
//	Project's Home Page:
//		https://picoforge.int-evry.fr/cgi-bin/twiki/view/Gpucv/Web/WebHome
//
//	This software is governed by the CeCILL  license under French law and
//	abiding by the rules of distribution of free software.  You can  use,
//	modify and/ or redistribute the software under the terms of the CeCILL
//	license as circulated by CEA, CNRS and INRIA at the following URL
//	"http://www.cecill.info".
//
//================================================================CVG_LicenseEnd
/** \brief Contains some CUDA arithmetic template kernels.
*	\author Yannick Allusse
*/
#ifndef __GPUCV_CUDA_CXCORE_CU_ARITHM_KERNEL_H
#define __GPUCV_CUDA_CXCORE_CU_ARITHM_KERNEL_H
#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>
#include <GPUCVCuda/base_kernels/cxcoregcu_arithm_fct.kernel.h>

#define CUDA_DEBUG_USE_TEX 1
#define _GCU_KERNEL_PRAGMA_UNROLL_NBR 8 //

#if _GPUCV_DEPRECATED
__device__ int make_color(float r, float g, float b, float a){
	return
		((int)(a * 255.0f) << 24) |
		((int)(b * 255.0f) << 16) |
		((int)(g * 255.0f) <<  8) |
		((int)(r * 255.0f) <<  0);
}
#endif

//#define IMIN(a, b) (a<b)?a:b
//#define OPERATOR(res, a, b) if(a>255)res = b; else res=a;

template <typename ARITHMFunction,int channels, typename TPLDataDst,typename TPLDataSrc, typename TPLDataSrc2>
__GCU_FCT_GLOBAL void GCUDA_KRNL_ARITHM_1BUFF(
	TPLDataDst * _dst,
	TPLDataSrc * _src,
	unsigned int _width,
	unsigned int _height,
	float4 _scale,
	TPLDataSrc2 _optVal)
{
	GCUDA_KRNL_DBG_FIRST_THREAD("GCUDA_KRNL_ARITHM_1BUFF", int a=0;)
	const unsigned int iy = (__mul24(blockIdx.y,blockDim.y) + threadIdx.y)*_GCU_KERNEL_PRAGMA_UNROLL_NBR;
	const unsigned int ix = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if (ix < _width && iy < _height)
	{
		unsigned int Pos = (iy*_width + ix);
#if _GCU_KERNEL_PRAGMA_UNROLL_NBR > 1
#pragma unroll
		for(int i = 0; i < _GCU_KERNEL_PRAGMA_UNROLL_NBR; i++)
#endif
		{
			//GCUDA_KRNL_DBG(printf("\nPos:%d/%d \tVal1/2:%d + %d\tSum:%d", ix, iy, Val1.x, Val2.x,Val1.x+ Val2.x);)
			ARITHMFunction::Do(_dst[Pos],_src[Pos], _optVal,_scale);
			Pos += _width;
		}
	}
	GCUDA_KRNL_DBG_LAST_THREAD("GCUDA_KRNL_ARITHM_1BUFF", int a=0;)
}

template <typename ARITHMFunction,int channels, typename TPLDataDst,typename TPLDataSrc, typename TPLDataSrc2>
__GCU_FCT_GLOBAL void GCUDA_KRNL_ARITHM_1BUFF_MASK(
												   TPLDataDst * _dst,
												   TPLDataSrc * _src,
												   uchar1 * _mask,
												   unsigned int _width,
												   unsigned int _height,
												   float4 _scale,
												   TPLDataSrc2 _optVal)
{
	GCUDA_KRNL_DBG_FIRST_THREAD("GCUDA_KRNL_ARITHM_1BUFF", int a=0;)
	const unsigned int iy = (__mul24(blockIdx.y,blockDim.y) + threadIdx.y)+_GCU_KERNEL_PRAGMA_UNROLL_NBR;
	const unsigned int ix = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	if (ix < _width && iy < _height)
	{
		unsigned int Pos = (iy*_width + ix);
		#if _GCU_KERNEL_PRAGMA_UNROLL_NBR > 1
#pragma unroll
		for(int i = 0; i < _GCU_KERNEL_PRAGMA_UNROLL_NBR; i++)
#endif
		{
			if(_mask[Pos].x)
			{//GCUDA_KRNL_DBG(printf("\nPos:%d/%d \tVal1/2:%d + %d\tSum:%d", ix, iy, Val1.x, Val2.x,Val1.x+ Val2.x);)
				ARITHMFunction::Do(_dst[Pos],_src[Pos], _optVal,_scale);
			}
			else
			{
			//GCU_OP_MULTIPLEXER<channels,KERNEL_ARITHM_OPER_CLEAR>::Do(_dst[Pos]);
			GCU_MULTIPLEX_0(channels,KERNEL_ARITHM_OPER_CLEAR, (_dst[Pos]), TPLDataDst);
			}
			Pos += _width;
		}
	}
	GCUDA_KRNL_DBG_LAST_THREAD("GCUDA_KRNL_ARITHM_1BUFF", int a=0;)
}


template <typename ARITHMFunction,int channels, typename TPLDataDst,typename TPLDataSrc>
__GCU_FCT_GLOBAL void GCUDA_KRNL_ARITHM_2BUFF(
											  TPLDataDst * _dst,
											  TPLDataSrc * _src1,
											  TPLDataSrc * _src2,
											  unsigned int _width,
											  unsigned int _height,
											  float4 _scale)
{
	GCUDA_KRNL_DBG_FIRST_THREAD("GCUDA_KRNL_ARITHM_1BUFF", int a=0;)

	const unsigned int iy = (__mul24(blockIdx.y,blockDim.y) + threadIdx.y)*_GCU_KERNEL_PRAGMA_UNROLL_NBR;
	const unsigned int ix = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	
	if (ix < _width && iy < _height)
	{
		unsigned int Pos = (iy*_width + ix);							
		//GCUDA_KRNL_DBG(printf("\nPos:%d/%d \tVal1/2:%d + %d\tSum:%d", ix, iy, Val1.x, Val2.x,Val1.x+ Val2.x);)
		TPLDataDst TempDest;
#if _GCU_KERNEL_PRAGMA_UNROLL_NBR > 1
#pragma unroll
		for(int i = 0; i < _GCU_KERNEL_PRAGMA_UNROLL_NBR; i++)
#endif
		{
			ARITHMFunction::Do(TempDest,_src1[Pos], _src2[Pos],_scale);
			_dst[Pos] = TempDest;
			Pos += _width;
		}
		GCUDA_KRNL_DBG_LAST_THREAD("GCUDA_KRNL_ARITHM_1BUFF", int a=0;)
	}
}

template <typename ARITHMFunction,int channels, typename TPLDataDst,typename TPLDataSrc>		\
__GCU_FCT_GLOBAL void GCUDA_KRNL_ARITHM_2BUFF_MASK(								\
												   TPLDataDst * _dst,											\
												   TPLDataSrc * _src1,										\
												   TPLDataSrc * _src2,										\
												   uchar1 * _mask,										\
												   unsigned int _width,										\
												   unsigned int _height,										\
												   float4 _scale)											\
{
#if 1
	GCUDA_KRNL_DBG_FIRST_THREAD("GCUDA_KRNL_ARITHM_1BUFF", int a=0;)\
	const unsigned int iy = (__mul24(blockIdx.y,blockDim.y) + threadIdx.y)*_GCU_KERNEL_PRAGMA_UNROLL_NBR;		\
	const unsigned int ix = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;		\
	if (ix < _width && iy < _height)											\
	{																			\
	unsigned int Pos = (iy*_width + ix);							\
	unsigned int PosMask = (iy*_width + ix);
#if _GCU_KERNEL_PRAGMA_UNROLL_NBR > 1
#pragma unroll
		for(int i = 0; i < _GCU_KERNEL_PRAGMA_UNROLL_NBR; i++)
#endif
		{
			//GCUDA_KRNL_DBG(printf("\nPos:%d/%d \tVal1/2:%d + %d\tSum:%d", ix, iy, Val1.x, Val2.x,Val1.x+ Val2.x);)
			if(_mask[PosMask].x)
			{//GCUDA_KRNL_DBG(printf("\nPos:%d/%d \tVal1/2:%d + %d\tSum:%d", ix, iy, Val1.x, Val2.x,Val1.x+ Val2.x);)
				ARITHMFunction::Do(_dst[Pos],_src1[Pos], _src2[Pos],_scale);\
			}
			else
			{
				//GCU_OP_MULTIPLEXER<channels,KERNEL_ARITHM_OPER_CLEAR>::Do(_dst[Pos]);
				GCU_MULTIPLEX_0(channels,KERNEL_ARITHM_OPER_CLEAR, (_dst[Pos]),TPLDataDst);
			}
			Pos += _width;
			PosMask += _width;
		}
	GCUDA_KRNL_DBG_LAST_THREAD("GCUDA_KRNL_ARITHM_2BUFF_MASK", int a=0;)
	}
#endif
}


#if 0
/** \brief Macro definition for arithmetic operators with image MASK
*	\note mask is always a single channel image.
*/


//, INPUT_TEX_MASK, INPUT_MASK_TYPE)
	__GCU_FCT_GLOBAL void CuTemplArithmKernel_uchar1(\
TPLDataSrc Val1 = tex2D(texA_uchar1_2,x, y);			\
TPLDataSrc Val2 = tex2D(texB_uchar1_2,x, y);			\

/** Performs arithmetic operation between 2 images
*/
#define GCU_SHARED_MEM_TEST_SIZE 1024
#define CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG2(INPUT_TEX_CHANNELS, INPUT_TEX_TYPE)\
	template <typename ARITHMFunction,int channels, typename TPLDataDst,typename TPLDataSrc>		\
	__GCU_FCT_GLOBAL void GCUDA_KRNL_ARITHM_2TEX_##INPUT_TEX_TYPE##INPUT_TEX_CHANNELS(					\
	TPLDataDst * _dst,											\
	unsigned int _width,										\
	unsigned int _height,										\
	unsigned int _channels,									\
	float4 _scale)												\
{																		\
	GCUDA_KRNL_DBG_FIRST_THREAD("NAME##_##INPUT_TEX_TYPE##INPUT_TEX_CHANNELS", int a=0;)\
	const unsigned int iy = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;		\
	const unsigned int ix = (__mul24(blockIdx.x,blockDim.x) + threadIdx.x);		\
	if (ix < _width && iy < _height)											\
{																			\
	float x = ((float)ix + 0.5f);												\
	float y = ((float)iy + 0.5f);												\
	unsigned int Pos = (iy*_width + ix);								\
	TPLDataSrc Val1 = tex2D(_CUDAG_GET_TEX_NM(A,INPUT_TEX_TYPE##INPUT_TEX_CHANNELS, 2),x, y);			\
	TPLDataSrc Val2 = tex2D(_CUDAG_GET_TEX_NM(B,INPUT_TEX_TYPE##INPUT_TEX_CHANNELS, 2),x, y);			\
	GCUDA_KRNL_DBG(printf("\nPos:%d/%d \tVal1/2:%d + %d\tSum:%d", ix, iy, Val1.x, Val2.x,Val1.x+ Val2.x);)\
	TPLDataDst TempDst;															\
	ARITHMFunction::Do(TempDst, Val1, Val2, _scale);							\
	GCU_OP_MULTIPLEXER<channels,KERNEL_ARITHM_OPER_AFFECT>::Do(_dst[Pos],TempDst);\
	GCUDA_KRNL_DBG_LAST_THREAD("NAME##_##INPUT_TEX_TYPE##INPUT_TEX_CHANNELS", int a=0;)\
}																					\
}

//StAffectFilterTex::Do(_dst[Pos], TempDst);\



//StAffectFilterTex::Do(_dst[Pos], TempDst);\
//

/** Performs arithmetic operation between 2 images using one mask image
*/
#define CUDA_ARITHM_TPL_KERNEL_TEX2D_MASK_IMG2(INPUT_TEX_CHANNELS, INPUT_TEX_TYPE)\
	template <typename ARITHMFunction,int channels, typename TPLDataDst,typename TPLDataSrc>		\
	__GCU_FCT_GLOBAL void GCUDA_KRNL_ARITHM_2TEX_MASK_##INPUT_TEX_TYPE##INPUT_TEX_CHANNELS(					\
	TPLDataDst * _dst,											\
	unsigned int _width,										\
	unsigned int _height,										\
	unsigned int _channels,									\
	float4 _scale)												\
{																		\
	GCUDA_KRNL_DBG_FIRST_THREAD("NAME##_##INPUT_TEX_TYPE##INPUT_TEX_CHANNELS", int a=0;)\
	const unsigned int iy = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;		\
	const unsigned int ix = (__mul24(blockIdx.x,blockDim.x) + threadIdx.x);		\
	if (ix < _width && iy < _height)											\
{																			\
	float x = ((float)ix + 0.5f)*channels;									\
	float y = ((float)iy + 0.5f);											\
	unsigned int Pos = (iy*_width + ix);						\
	TPLDataSrc ValMask = tex2D(_CUDAG_GET_TEX_NM(MASK,INPUT_TEX_TYPE##INPUT_TEX_CHANNELS, 2),x, y);			\
	if(ValMask.x>0)															\
{																		\
	TPLDataSrc Val1 = tex2D(_CUDAG_GET_TEX_NM(A,INPUT_TEX_TYPE##INPUT_TEX_CHANNELS, 2),x, y);			\
	TPLDataSrc Val2 = tex2D(_CUDAG_GET_TEX_NM(B,INPUT_TEX_TYPE##INPUT_TEX_CHANNELS, 2),x, y);			\
	GCUDA_KRNL_DBG(printf("\nPos:%d/%d \tVal1/2:%d + %d\tSum:%d", ix, iy, Val1.x, Val2.x,Val1.x+ Val2.x);)\
	TPLDataDst TempDst;											\
	ARITHMFunction::Do(TempDst, Val1, Val2, _scale);			\
	GCU_OP_MULTIPLEXER<channels,KERNEL_ARITHM_OPER_AFFECT>::Do(_dst[Pos],TempDst);\
}																	\
			else																\
{																	\
	GCU_OP_MULTIPLEXER<channels,KERNEL_ARITHM_OPER_CLEAR>::Do(_dst[Pos]);	\
}																	\
	GCUDA_KRNL_DBG_LAST_THREAD("NAME##_##INPUT_TEX_TYPE##INPUT_TEX_CHANNELS", int a=0;)\
}																								\
}



/** Performs arithmetic operation between 2 images
*/
#define CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG1_1SCALAR(INPUT_TEX_CHANNELS, INPUT_TEX_TYPE)\
	template <typename ARITHMFunction,int channels, typename TPLDataDst,typename TPLDataSrc>		\
	__GCU_FCT_GLOBAL void GCUDA_KRNL_ARITHM_1TEX_1SCALAR_##INPUT_TEX_TYPE##INPUT_TEX_CHANNELS(			\
	TPLDataDst * _dst,											\
	unsigned int _width,										\
	unsigned int _height,										\
	unsigned int _channels,									\
	float4 _scale,												\
	TPLDataSrc _scalarVal)										\
{																		\
	GCUDA_KRNL_DBG_FIRST_THREAD("NAME##_##INPUT_TEX_TYPE##INPUT_TEX_CHANNELS", int a=0;)\
	const unsigned int iy = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;		\
	const unsigned int ix = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;		\
	if (ix < _width && iy < _height)											\
{																			\
	float x = ((float)ix + 0.5f)*channels;									\
	float y = ((float)iy + 0.5f);												\
	unsigned int Pos = (iy*_width + ix);								\
	TPLDataSrc Val1 = tex2D(_CUDAG_GET_TEX_NM(A,INPUT_TEX_TYPE##INPUT_TEX_CHANNELS, 2),x, y);			\
	GCUDA_KRNL_DBG(printf("\nPos:%d/%d \tVal1/2:%d + %d\tSum:%d", ix, iy, Val1.x, _scalarVal.x, Val1.x+ _scalarVal.x);)\
	TPLDataDst TempDst;															\
	ARITHMFunction::Do(TempDst, Val1, _scalarVal, _scale);						\
	GCU_OP_MULTIPLEXER<channels,KERNEL_ARITHM_OPER_AFFECT>::Do(_dst[Pos],TempDst);		\
	GCUDA_KRNL_DBG_LAST_THREAD("NAME##_##INPUT_TEX_TYPE##INPUT_TEX_CHANNELS", int a=0;)\
}																						\
}


/** Performs arithmetic operation between 2 images using one mask image
*/
#define CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG1_1SCALAR_MASK(INPUT_TEX_CHANNELS, INPUT_TEX_TYPE)\
	template <typename ARITHMFunction,int channels, typename TPLDataDst,typename TPLDataSrc>		\
	__GCU_FCT_GLOBAL void GCUDA_KRNL_ARITHM_1TEX_1SCALAR_MASK_##INPUT_TEX_TYPE##INPUT_TEX_CHANNELS(					\
	TPLDataDst * _dst,											\
	unsigned int _width,										\
	unsigned int _height,										\
	unsigned int _channels,									\
	float4 _scale,												\
	TPLDataSrc _scalarVal)											\
{																		\
	GCUDA_KRNL_DBG_FIRST_THREAD("NAME##_##INPUT_TEX_TYPE##INPUT_TEX_CHANNELS", int a=0;)\
	const unsigned int iy = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;		\
	const unsigned int ix = (__mul24(blockIdx.x,blockDim.x) + threadIdx.x);		\
	if (ix < _width && iy < _height)											\
{																			\
	float x = ((float)ix + 0.5f)*channels;									\
	float y = ((float)iy + 0.5f);											\
	unsigned int Pos = (iy*_width + ix);						\
	TPLDataSrc ValMask = tex2D(_CUDAG_GET_TEX_NM(MASK,INPUT_TEX_TYPE##INPUT_TEX_CHANNELS, 2),x, y);			\
	if(ValMask.x>0)															\
{																		\
	TPLDataSrc Val1 = tex2D(_CUDAG_GET_TEX_NM(A,INPUT_TEX_TYPE##INPUT_TEX_CHANNELS, 2),x, y);			\
	TPLDataSrc Val2 = tex2D(_CUDAG_GET_TEX_NM(B,INPUT_TEX_TYPE##INPUT_TEX_CHANNELS, 2),x, y);			\
	GCUDA_KRNL_DBG(printf("\nPos:%d/%d \tVal1/2:%d + %d\tSum:%d", ix, iy, Val1.x, _scalarVal.x,Val1.x+ _scalarVal.x);)\
	TPLDataDst TempDst;											\
	ARITHMFunction::Do(TempDst, Val1, _scalarVal, _scale);			\
	GCU_OP_MULTIPLEXER<channels,KERNEL_ARITHM_OPER_AFFECT>::Do(_dst[Pos],TempDst);\
}																	\
			else																\
{																	\
	GCU_OP_MULTIPLEXER<channels,KERNEL_ARITHM_OPER_CLEAR>::Do(_dst[Pos]);	\
}																	\
	GCUDA_KRNL_DBG_LAST_THREAD("NAME##_##INPUT_TEX_TYPE##INPUT_TEX_CHANNELS", int a=0;)\
}																						\
}

//using CUDA textures
#define DECLARE_ALL_TYPE__CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG2(KERNEL_MACRO_FUNCT, CHANNELS)\
	KERNEL_MACRO_FUNCT(CHANNELS, uchar);\
	KERNEL_MACRO_FUNCT(CHANNELS, char);\
	KERNEL_MACRO_FUNCT(CHANNELS, short);\
	KERNEL_MACRO_FUNCT(CHANNELS, ushort);\
	KERNEL_MACRO_FUNCT(CHANNELS, int);\
	KERNEL_MACRO_FUNCT(CHANNELS, uint);\
	KERNEL_MACRO_FUNCT(CHANNELS, float);

#define DECLARE_ALL_CHANNELS__CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG2(KERNEL_FUNCT)\
	DECLARE_ALL_TYPE__CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG2(KERNEL_FUNCT,1);\
	DECLARE_ALL_TYPE__CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG2(KERNEL_FUNCT,4);
//DECLARE_ALL_TYPE__CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG2(KERNEL_FUNCT,2);\ //GCU_USE_CHANNEL_2


DECLARE_ALL_CHANNELS__CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG2(CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG2);
DECLARE_ALL_CHANNELS__CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG2(CUDA_ARITHM_TPL_KERNEL_TEX2D_MASK_IMG2);
DECLARE_ALL_CHANNELS__CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG2(CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG1_1SCALAR);
DECLARE_ALL_CHANNELS__CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG2(CUDA_ARITHM_TPL_KERNEL_TEX2D_IMG1_1SCALAR_MASK);
#endif
#endif//_GPUCV_CUDA_CXCORE_CU_ARITHM_KERNEL_H
