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
/** \brief Contains GpuCV-CUDA correspondance of cxcore->array.
\author Yannick Allusse
*/
#ifndef __GPUCV_CUDA_ARITHM_LOGIC_TEMPLATES_H
#define __GPUCV_CUDA_ARITHM_LOGIC_TEMPLATES_H
#include "../../config.h"
#if _GPUCV_COMPILE_CUDA

//#include <GPUCVCuda/base_kernels/tpl_textures.kernel.h>
#include <typeinfo>
#include <GPUCVCuda/gpucv_wrapper_c.h>
//#include <GPUCV/oper_enum.h>
#include <cxcoregcu/cxcoregcu_array_arithm.kernel.h>
#include <GPUCVCuda/base_kernels/cxcoregcu_arithm_fct.kernel.h>
#include <cxcoregcu/cxcoregcu_statistics.kernel.h>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
enum GCU_ALU_TYPE
{
	GCU_ARITHM,
	GCU_LOGIC,
	GCU_NO_RESHAPE //! Optional, disable automatic reshaping of images, might be used when using masks or with scalar
};

#define PROF_ARITHM(FCT)//FCT
#define _GCU_ALLOW_RESHAPE 1
extern GCULogicStruct		varLocalLogic;//!< Used only for template specialization, does not contain any data.
extern GCUArithmStruct		varLocalArithm;//!< Used only for template specialization, does not contain any data.

template <typename TPL_OPERATOR, typename TPL_OPERATORTYPE, int channels, typename TPLSrcType,  typename TPLDstType>
void CudaArithm_SWITCHALL(TPL_OPERATORTYPE* ALUType,
						  void* d_src1,
						  void* d_src2,
						  void* d_dst,
						  void* d_mask,
						  unsigned int _width,
						  unsigned int _height,
						  double _scale=1.,
						  float4 * _Scalar=NULL
						  )
{
	printf("\nCudaArithm_SWITCHALL() -> Unkown operator type, must be GCUArithmStruct or GCULogicStruct!! No kernel executed\n");
}

//Used only for arthmetics
template <typename TPL_OPERATOR, typename TPL_OPERATORTYPE, int channels, typename TPLSrcType,  typename TPLDstType>
void CudaArithm_SWITCHALL(
						  GCUArithmStruct * ALUType,
						  void* d_src1,
						  void* d_src2,
						  void* d_dst,
						  void* d_mask,
						  unsigned int _width,
						  unsigned int _height,
						  double _scale=1.,
						  float4 * _Scalar=NULL
						  )
{
	//=====================
	//prepare parameters
	dim3 threads(16,16,1);
	dim3 blocks = dim3(iDivUp(_width,threads.x),
		iDivUp(_height, threads.y),
		1);

	//update parameters with current hardware settings
#if 0
	cudaDeviceProp * devProperties = gcuGetDeviceProperties();
	if (0)//devProperties) 
	{
		if(devProperties->major ==2)
			threads.x = devProperties->warpSize/2;
		else
			threads.x = devProperties->warpSize;
		

		//threads.y = devProperties->warpSize;
		
		while( (threads.x * threads.y * threads.z) > devProperties->maxThreadsPerBlock)
		{
			threads.y /=2;
		}
		blocks = dim3(iDivUp(_width,threads.x),
					  iDivUp(_height, threads.y),
					  1);
	}
#endif
	//update with for loop unrolling
	blocks.y /= _GCU_KERNEL_PRAGMA_UNROLL_NBR;
	
	
	//dispatch charge
	if(blocks.x <2)
	{
		blocks.x*=4;
		threads.x/=4;
	}
	if(blocks.y <2)
	{
		blocks.y*=4;
		threads.y/=4;
	}
	//=================
#if 0//def _DEBUG
	printf("\nCudaArithm_SWITCHALL===\n");
	printf("- width: %d\n", _width);
	printf("- height: %d\n", _height);
	//printf("- channels: %d\n", _channels);
	printf("- scale factor: %f\n", _scale);
	//printf("- DataN: %d\n", DataN);
	//printf("- DataSize: %d\n", DataSize);
	printf("- threads: %d %d %d\n", threads.x, threads.y, threads.z);
	printf("- blocks: %d %d %d\n", blocks.x, blocks.y, blocks.z);
#endif

#if 1 //simulate filter
	if(1)
	{
		//size_t dst_pitch=0;
		float4 Scale;
		Scale.x = Scale.y =Scale.z =Scale.w = (float)_scale;
		TPLSrcType KernelScalar;
		if(_Scalar)
		{
			for( int i = 0; i < channels; i++)
			{
				(&(KernelScalar.x))[i] = (&(_Scalar->x))[i];
			}
		}

		//call processing operator
		if(_Scalar)
		{
				if(d_mask)
					GCUDA_KRNL_ARITHM_1BUFF_MASK<StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_dst, (TPLSrcType *)d_src1, (uchar1 *)d_mask, _width,_height,Scale, KernelScalar);
				else
					GCUDA_KRNL_ARITHM_1BUFF<StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_dst, (TPLSrcType *)d_src1, _width,_height,Scale, KernelScalar);
		}
		else
		{
				if(d_mask)
					GCUDA_KRNL_ARITHM_2BUFF_MASK<StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_dst, (TPLSrcType*)d_src1, (TPLSrcType*)d_src2, (uchar1*)d_mask, _width,_height,Scale);
				else
					GCUDA_KRNL_ARITHM_2BUFF<StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_dst, (TPLSrcType*)d_src1, (TPLSrcType*)d_src2, _width,_height,Scale);

		}
	}
#endif//simule
}

//Used only for arthmetics
template <typename TPL_OPERATOR, typename TPL_OPERATORTYPE, int channels, typename TPLSrcType,  typename TPLDstType>
void CudaArithm_SWITCHALL(GCULogicStruct * ALUType,
						  void* d_src1,
						  void* d_src2,
						  void* d_dst,
						  void* d_mask,
						  unsigned int _width,
						  unsigned int _height,
						  double _scale=1.,
						  float4 * _Scalar=NULL
						  )
{
	//=====================
	//prepare parameters
	dim3 threads(16,16,1);
	dim3 blocks = dim3(iDivUp(_width,threads.x),
		iDivUp(_height, threads.y),
		1);
	blocks.y /= _GCU_KERNEL_PRAGMA_UNROLL_NBR;
	if(blocks.x <2)
	{
		blocks.x*=4;
		threads.x/=4;
	}
	if(blocks.y <2)
	{
		blocks.y*=4;
		threads.y/=4;
	}
	//=================
#if 0//def _DEBUG
	printf("\nCudaArithm_SWITCHALL===\n");
	printf("- width: %d\n", _width);
	printf("- height: %d\n", _height);
	//printf("- channels: %d\n", _channels);
	printf("- scale factor: %f\n", _scale);
	//printf("- DataN: %d\n", DataN);
	//printf("- DataSize: %d\n", DataSize);
	printf("- threads: %d %d %d\n", threads.x, threads.y, threads.z);
	printf("- blocks: %d %d %d\n", blocks.x, blocks.y, blocks.z);
#endif

#if 1 //simulate filter
	if(1)
	{
		//size_t dst_pitch=0;
		float4 Scale;
		Scale.x = Scale.y =Scale.z =Scale.w = (float)_scale;
		TPLSrcType KernelScalar;
		if(_Scalar)
		{
			for( int i = 0; i < channels; i++)
			{
				(&(KernelScalar.x))[i] = (&(_Scalar->x))[i];
			}
		}

		//call processing operator
		if(_Scalar)
		{
				if(d_mask)
					GCUDA_KRNL_ARITHM_1BUFF_MASK<StLogicFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_dst, (TPLSrcType *)d_src1, (uchar1 *)d_mask, _width,_height,Scale, KernelScalar);
				else
					GCUDA_KRNL_ARITHM_1BUFF<StLogicFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_dst, (TPLSrcType *)d_src1, _width,_height,Scale, KernelScalar);
		}
		else
		{
				if(d_mask)
					GCUDA_KRNL_ARITHM_2BUFF_MASK<StLogicFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_dst, (TPLSrcType *)d_src1, (TPLSrcType *)d_src2, (uchar1 *)d_mask, _width,_height,Scale);
				else
					GCUDA_KRNL_ARITHM_2BUFF<StLogicFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_dst, (TPLSrcType *)d_src1, (TPLSrcType *)d_src2, _width,_height,Scale);//, KernelScalar);
		}
	}

#endif//simule
}
//=========================================================
/** \todo Reshape with mask not supported yet.
\note Can not reshape with 3 channels scalar values.
*/
template <typename TPL_OPERATOR, typename TPL_OPERATORTYPE/*Logic/Arithm*/>
void CudaArithm_SwitchCHANNELS(
							   TPL_OPERATORTYPE * ALUType,
							   CvArr* _src1,
							   CvArr* _src2,
							   CvArr* _dst,
							   CvArr* _mask,
							   double scale=1.,
							   float4* _Scalar=NULL)
{
	void * d_src1 = NULL;
	void * d_src2 = NULL;
	void * d_mask = NULL;
	void * d_dst = NULL;

	unsigned int width = gcuGetWidth(_dst);
	unsigned int height = gcuGetHeight(_dst);
	
	int channels = gcuGetnChannels(_dst);
	int NewChannels = channels;
	if(_mask)//No reshape here yet...due to the mask...
	{
#if 0//_DEBUG
		printf("\nChannels nbr could not be reshaped(%d)", channels);
#endif
	}
	else if(_Scalar && channels==3)//check for scalar values
	{//we can not yet re-affect scalar value with reshaping...
	}
	else
	{
		if(IS_MULTIPLE_OF(width *channels, 4))
			NewChannels = 4;
		else if(IS_MULTIPLE_OF(width *channels, 2))
			NewChannels = 2;
	}

	//load data and try reshaping
	if (NewChannels != channels)
	{//synchro images first
		if(_Scalar)
		{
			if(channels==1 && NewChannels==2)
			{
				_Scalar->y = _Scalar->x;
			}
			else if(channels==1 && NewChannels==4)
			{
				_Scalar->y = _Scalar->x;
				_Scalar->z = _Scalar->x;
				_Scalar->w = _Scalar->x;
			}
			else if(channels==2 && NewChannels==4)
			{
				_Scalar->z = _Scalar->x;
				_Scalar->w = _Scalar->y;
			}
		}
		//refresh size
		width = (width * channels)/NewChannels;// gcuGetWidth(_dst);
	}
	if(_dst)//output is always in CUDA_BUFFER
		d_dst = gcuPreProcess(_dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
	if(_src1)
		d_src1 = gcuPreProcess(_src1, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	if(_src2)
		d_src2 = gcuPreProcess(_src2, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	if(_mask)
		d_mask = gcuPreProcess(_mask, GCU_INPUT, CU_MEMORYTYPE_DEVICE);

	unsigned int depth		= gcuGetGLDepth(_src1);

	//here we use reshaped width and channels when possible
	#define GCU_ARITHM_SWITCH_FCT(CHANNELS, SRC_TYPE, DST_TYPE)\
		CudaArithm_SWITCHALL<TPL_OPERATOR,TPL_OPERATORTYPE, CHANNELS, SRC_TYPE##CHANNELS, DST_TYPE##CHANNELS>(ALUType,d_src1, d_src2, d_dst, d_mask, width, height, scale,_Scalar);

	GCU_MULTIPLEX_ALLCHANNELS_ALLFORMAT(GCU_ARITHM_SWITCH_FCT, NewChannels, depth);

	//kernel executed...
	CUT_CHECK_ERROR("Kernel execution failed");

	//=====================
	//post process
	if(_src1)
		gcuPostProcess(_src1);
	if(_src2)
		gcuPostProcess(_src2);
	if(_dst)
		gcuPostProcess(_dst);
	if(_mask)
		gcuPostProcess(_mask);
	//---------------------
}
#endif//_GPUCV_COMPILE_CUDA
#endif//__GPUCV_CUDA_ARITHM_LOGIC_TEMPLATES_H
