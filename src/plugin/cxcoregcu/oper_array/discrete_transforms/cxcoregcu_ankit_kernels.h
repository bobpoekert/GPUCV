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

#ifndef __GPUCV_CUDA_CXCOREGCU_ANKIT_H
#define __GPUCV_CUDA_CXCOREGCU_ANKIT_H

#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>
#include <GPUCVCuda/base_kernels/cxcoregcu_arithm_fct.kernel.h>
#if _GPUCV_COMPILE_CUDA

#if 0

template <typename TPLDst,typename TPLSrc>
__device__
TPLDst NewClamp(TPLDst &dst,TPLSrc &value)
{	
	int result1 = value - GetTypeMaxVal(dst);
	result1 = result1 >> 31 ;
	result1 = result1 & 0x01;
	int result2 = GetTypeMinVal(dst)- value ;
	result2 = result2 >> 31 ;
	result2 = result2 & 0x01 ;
	dst = value*result1*result2+(~result1& 0x01)*GetTypeMaxVal(dst)+(~result1& 0x01)*GetTypeMinVal(dst);
	return dst;	
}
template <typename TPLSrc>
__device__
float NewClamp(float &dst,TPLSrc &value)
{
	return dst=value;
}

#endif
#if CUDA_VERSION < 2010
template <int channels>	
struct CSgcuChannelKernel
{
	template <typename DST_TYPE,typename TPLDst,typename TPLDataSrcFULL>
	__device__ static//__GCU_FCT_ALLSIDE_INLINE 
		void ConvertScale(TPLDataSrcFULL  & src , TPLDst & dst, float s1,float s2)
	{
		float4 TempVal;//float is used cause we know the result may be out of the TPLDataDst range
		float4 TempS1;
		float4 TempS2;
		TempS1.x = TempS1.y = TempS1.z = TempS1.w = s1;
		TempS2.x = TempS2.y = TempS2.z = TempS2.w = s2;
		GCU_OP_MULTIPLEXER<channels, KERNEL_ARITHM_OPER_MUL>::Do(TempVal, src, TempS1);
		GCU_OP_MULTIPLEXER<channels, KERNEL_ARITHM_OPER_ADD>::Do(TempVal, TempS2);//, float4, float4
#if 1
		GCU_OP_MULTIPLEXER<channels, KERNEL_ARITHM_OPER_CLAMP>::Do(dst,TempVal);//, TPLDataSrc,float4//clamp result into destination data
#else
		GCU_OP_MULTIPLEXER<channels, KERNEL_ARITHM_OPER_CLAMP>::Do(dst,TempVal);//, TPLDataSrc,float4//clamp result into destination data
#endif
		return;
	}

};
#endif
#if CUDA_VERSION < 2010
/*=====================================================

Add on 4 Channels
*/
template <>	
struct CSgcuChannelKernel<4>
{
	template <typename TPLDst,typename TPLDataSrcFULL>
	__device__ static//__GCU_FCT_ALLSIDE_INLINE
		void ConvertScale( TPLDataSrcFULL *block,TPLDst *dst,float s1,
		float s2)
	{	TPLDataSrcFULL px1 = *((TPLDataSrcFULL*) (&block));
	float4 TempVal;
	TempVal.x = px1.x*s1 + s2;
	TempVal.y = px1.y*s1 + s2;
	TempVal.z = px1.z*s1 + s2;
	TempVal.w = px1.w*s1 + s2;
	//TPLDst TempDestVal;
	GCU_OP_MULTIPLEXER <4,KERNEL_ARITHM_OPER_CLAMP>::Do(dst,TempVal);
	//GCU_OP_MULTIPLEXER <4,KERNEL_ARITHM_OPER_AFFECT>::Do(dst, TempDestVal);		
	}
};
/*=====================================================

Add on 3 Channels
*/
template <>	
struct CSgcuChannelKernel<3>
{
	template <typename TPLDst, typename TPLDataSrcFULL>
	__device__ static//__GCU_FCT_ALLSIDE_INLINE
		void ConvertScale( TPLDataSrcFULL *block,TPLDst *dst,float s1,
		float s2)
	{	TPLDataSrcFULL px1 = *((TPLDataSrcFULL*) (&block));
	float3 TempVal;
	TempVal.x = px1.x*s1 + s2;
	TempVal.y = px1.y*s1 + s2;
	TempVal.z = px1.z*s1 + s2;
	//dst.x = TempVal.x;
	//dst.y = TempVal.y;
	//dst.z = TempVal.z;
	GCU_OP_MULTIPLEXER <3,KERNEL_ARITHM_OPER_CLAMP>::Do(dst,TempVal);
	//MultiPlex<3, KERNEL_ARITHM_OPER_CLAMP>(dst, TempVal);
	}
};

/*=====================================================

Add on 2 Channels
*/
template <>	
struct CSgcuChannelKernel<2>
{
	template <typename TPLDst, typename TPLDataSrcFULL>
	__device__ static//__GCU_FCT_ALLSIDE_INLINE
		void ConvertScale( TPLDataSrcFULL *block,TPLDst *dst,float s1,
		float s2)

	{	TPLDataSrcFULL px1 = *((TPLDataSrcFULL*) (&block));
	float2 TempVal;
	TempVal.x = px1.x*s1 + s2;
	TempVal.y = px1.y*s1 + s2;
	GCU_OP_MULTIPLEXER <2,KERNEL_ARITHM_OPER_CLAMP>::Do(dst,TempVal);
	}
};

/*=====================================================

Add on 1 Channels
*/
template <>	
struct CSgcuChannelKernel<1>
{
	template <typename TPLDst,typename TPLDataSrcFULL>
	__device__ static//__GCU_FCT_ALLSIDE_INLINE
		void ConvertScale( TPLDataSrcFULL *block , TPLDst *dst, float s1 , float s2)
	{	TPLDataSrcFULL px1 = *((TPLDataSrcFULL*) (&block));
	//int px1 = (&block[index_block]));
	float1 TempVal;
	TempVal.x = px1.x*s1 + s2;
	//dst[_IndexOut]=TempVal;
	GCU_OP_MULTIPLEXER <1,KERNEL_ARITHM_OPER_CLAMP>::Do(dst,TempVal);
	//dst[_IndexOut].x = px1.x*sa1 + px2.x*sa2 + gam;
	//dst[_IndexOut].x = px1.x*0.50000 + px2.x*0.50000 + 50;
	}
};
#endif
/*======================================================
main function
kernel using shared memory,but the shared mempry is declared as of source type and the resulting bench marks are better than opencv only for images havng 32f depth.checking todo
*
*/

#define GCU_CONCERT_SHARE 0
template <int TPLChannels,typename TPLDataDst,typename TPLDataSrc,int block_width, int block_height>
__GCU_FCT_GLOBAL 
void gcuConvertScaleKernel(TPLDataSrc *src1,TPLDataDst *dst,float4 scale,float4 shift,
						   uint width,
						   uint height,
						   size_t pitch)

{
	#if CUDA_VERSION < 2010
#if GCU_CONCERT_SHARE
	__shared__ int4 block[block_width*block_height*sizeof(TPLDataSrcFULL)];

	//__shared__ TPLDataSrcFULL block[block_width*block_height];
	int4 *I4src_ptr =(int4*) src1;
	TPLDataSrcFULL * TPL_block_ptr = (TPLDataSrcFULL*)block;
#endif
	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;
	const unsigned int iy = __mul24(blockIdx.y,blockDim.y) + threadIdx.y;		\
	const unsigned int ix = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;		\
	//unsigned int index_in  = __mul24(pitch, yBlock + threadIdx.y) + 
	//	xBlock + threadIdx.x;
	unsigned int index_in = (iy*width + ix);
#if GCU_CONCERT_SHARE
	index_in = (index_in * sizeof(TPLDataSrcFULL))/ sizeof(int4);;
	//unsigned int index_block = __mul24(threadIdx.y, block_width) + threadIdx.x;
	//index_block = (index_block * sizeof(TPLDataSrcFULL))/sizeof(int4);

	unsigned int index_block_B = (__mul24(threadIdx.y, block_width) + threadIdx.x)*sizeof(TPLDataSrcFULL);
	unsigned int index_in_B  = (__mul24(pitch, yBlock + threadIdx.y) + xBlock + threadIdx.x)*sizeof(TPLDataSrcFULL);

	unsigned int index_block_IT4	= index_block_B/sizeof(int4);
	unsigned int index_in_IT4		= index_in_B/sizeof(int4);
	unsigned int index_block_TPL	= index_block_B/sizeof(TPLDataSrcFULL);
	unsigned int index_in_TPL		= index_in_B/sizeof(TPLDataSrcFULL);



	if (xIndex < width && yIndex < height)
	{
		if (index_block_B < block_width*block_height*sizeof(TPLDataSrcFULL))
		{
			//GCU_OP_MULTIPLEXER<4,KERNEL_ARITHM_OPER_AFFECT>::Do(block[index_block],I4src_ptr[index_in]);
			GCU_OP_MULTIPLEXER<4,KERNEL_ARITHM_OPER_AFFECT>::Do(block[index_block_IT4],I4src_ptr[index_in_IT4]);
			//block[index_block_IT4] = I4src_ptr[index_in_IT4];
			//GCU_OP_MULTIPLEXER<4,KERNEL_ARITHM_OPER_AFFECT>::Do(((int4*)dst)[index_in],I4src_ptr[index_in]);
		}
		/*	else
		{
		int4 black;
		black.x = black.y = black.z=black.w=0;
		GCU_OP_MULTIPLEXER<4,KERNEL_ARITHM_OPER_AFFECT>::Do(block[index_block_IT4],black);
		}
		*/

	}
	/*
	__syncthreads();
	if (xIndex < width && yIndex < height)
	{
	//if (index_in*sizeof(int4)<block_width*sizeof(TPLDataSrcFULL))
	{
	int4 *I4dst_ptr =(int4*) dst;
	I4dst_ptr[index_in] = block[index_block];
	}
	}
	*/
	__syncthreads();
#endif

#if 1
	if (xIndex < width && yIndex < height)
	{
		unsigned int _IndexOut	= __mul24(pitch, yBlock + threadIdx.y) + 
			xBlock + threadIdx.x;

#if GCU_CONCERT_SHARE
		CSgcuChannelKernel<TPLChannels>::ConvertScale<TPLDst,TPLDataSrcFULL>(TPL_block_ptr[_IndexOut],dst,scale,shift);
#else
		//CSgcuChannelKernel<TPLChannels>::ConvertScale<TPLDst,TPLDataSrcFULL>(src1[_IndexOut],dst[_IndexOut],scale,shift);

#if 0
		CSgcuChannelKernel<TPLChannels>::ConvertScale<DST_TYPE,TPLDst,TPLDataSrcFULL>(src1[_IndexOut],dst[_IndexOut],scale,shift);
#else
		TPLDataDst TempDest;
		float4 TempFloat;
		//CSgcuChannelKernel<TPLChannels>::ConvertScale<DST_TYPE,TPLDst,TPLDataSrcFULL>(src1[_IndexOut],TempDest,scale,shift);
		GCU_OP_MULTIPLEXER<TPLChannels,KERNEL_ARITHM_OPER_MUL, float4, TPLDataSrc, float4>::Do(TempFloat, src1[index_in], scale);
		GCU_OP_MULTIPLEXER<TPLChannels,KERNEL_ARITHM_OPER_ADD, float4, float4, float4>::Do(TempFloat, TempFloat, shift);
		GCU_OP_MULTIPLEXER<TPLChannels,KERNEL_ARITHM_OPER_CLAMP, TPLDataSrc, float4>::Do(TempDest,TempFloat);
		//GCU_OP_MULTIPLEXER<TPLChannels,KERNEL_ARITHM_OPER_AFFECT>::Do(dst[index_in],TempFloat);
		dst[index_in] = TempDest;
		//dst[index_in].x = 0xFF;
		//dst[_IndexOut].y = 0xFF;
		//dst[_IndexOut].z = 0xFF;
		//StAffectFilterTex::Do(dst[_IndexOut], TempDest);
#endif
		//GCU_OP_MULTIPLEXER<TPLChannels, KERNEL_ARITHM_OPER_AFFECT>::Do(dst[_IndexOut],TempDest);//, TPLDataSrc,float4//clamp result into destination data
#endif
	}
#endif
#endif//CUDA version
}

#endif 
#endif
//__GPUCV_CUDA_CXCOREGCU_ANKIT_H

