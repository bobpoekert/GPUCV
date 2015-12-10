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
#include <cxcoregcu/config.h>
#if _GPUCV_DEPRECATED//_GPUCV_COMPILE_CUDA

#include <GPUCVCuda/base_kernels/tpl_textures.kernel.h>
#include <typeinfo>
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <GPUCV/oper_enum.h>
#include <cxcoregcu/cxcoregcu_array_arithm.kernel.h>
#include <GPUCVCuda/base_kernels/cxcoregcu_arithm_fct.kernel.h>
#include <cxcoregcu/cxcoregcu_statistics.kernel.h>

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#define CUDA_ARITHM_TPL_USE_TEX 0
#define PROF_ARITHM(FCT)//FCT


#if CUDA_ARITHM_TPL_USE_TEX
/** \brief Structs to call template CUDA kernels depending on TPL_OPERATOR/TPL_CHANNEL/TPL_DST/TPL_SRC.
\note Using texture with CUDA is faster than device array but require to know the exact type of the texture data at compile time.
GpuCV was designed to process any kind of format/channels so we need a specific template mechanism that hide the CUDA texture description.
\author Yannick Allusse
*/
template <typename TPL_OPERATOR, int TPL_CHANNEL, typename TPL_DST,typename TPL_SRC>//, typename TPL_OPERATOR >
struct GCU_StArithmFct
{
	/** \brief Apply the operator TPL_OPERATOR to CUDA textures texA%TYPE% and texB%TYPE% with a scale factor _scale.
	*/
	static __host__ void DoTex2_Fl1(dim3 Blocks, dim3 grids, TPL_DST * _dest, uint _width,uint _height, char _channels, float4 _scale)
	{
		printf("\n\nCUDAArithmOperFctTex=> no template match available functions....\n\n");
	}
	/** \brief Apply the ope	rator TPL_OPERATOR to CUDA textures texA%TYPE% and texB%TYPE% using the texture texMASK%TYPE% and a scale factor _scale.
	*/
	static __host__ void DoTex2_Fl1_Mask(dim3 Blocks, dim3 grids, TPL_DST * _dest, uint _width,uint _height, char _channels, float4 _scale)
	{
		printf("\n\nCUDAArithmOperFctTex=> no template match available functions....\n\n");
	}
	/** \brief Apply the operator TPL_OPERATOR to CUDA texture texA%TYPE% and scalar _scalar using with a scale factor _scale.
	*/
	static __host__ void DoTex1_Scalar1_Fl1(dim3 Blocks, dim3 grids, TPL_DST * _dest, uint _width, uint _height, char _channels, float4 _scale, TPL_SRC & _scalar)
	{
		printf("\n\nCUDAArithmOperFctTex=> no template match available functions....\n\n");
	}
	/** \brief Apply the operator TPL_OPERATOR to CUDA texture texA%TYPE% and scalar _scalar using the texture texMASK%TYPE% and a scale factor _scale.
	*/
	static __host__ void DoTex1_Scalar1_Fl1_Mask(dim3 Blocks, dim3 grids, TPL_DST * _dest, uint _width,uint _height, char _channels, float4 _scale, TPL_SRC & _scalar)
	{
		printf("\n\nCUDAArithmOperFctTex=> no template match available functions....\n\n");
	}
	/** \brief Load given IplImage into CUDA Array and connect it to texture texA_##TPL_SRC##_2.
	*/
	static __host__ void LoadIntoTex_A(IplImage* _IplSrc)
	{
		printf("\nLoadIntoTex_A=> no template match available functions....\n\n");
	}
	/** \brief Load given IplImage into CUDA Array and connect it to texture texB_##TPL_SRC##_2.
	*/
	static __host__ void LoadIntoTex_B(IplImage* _IplSrc)
	{
		printf("\nLoadIntoTex_B=> no template match available functions....\n\n");
	}
	/** \brief Load given IplImage into CUDA Array and connect it to texture texC_##TPL_SRC##_2.
	*/
	static __host__ void LoadIntoTex_C(IplImage* _IplSrc)
	{
		printf("\nLoadIntoTex_C=> no template match available functions....\n\n");
	}
	/** \brief Load given IplImage into CUDA Array and connect it to texture texMASK_##TPL_SRC##_2.
	*/
	static __host__ void LoadIntoTex_Mask(IplImage* _IplSrc)
	{
		printf("\nLoadIntoTex_Mask=> no template match available functions....\n\n");
	}
};

/** \brief Define template specifications to GCU_StArithmFct for all channels/data type possibilities.
*	\sa CUDA_ARITHM_OPER_TEX_FCT__ALL_TYPE, CUDA_ARITHM_OPER_TEX_FCT__ALL_CHANNELS, CUDAArithmOperFctTex
*	\related GCU_StArithmFct
*/
#define CUDA_ARITHM_OPER_TEX_FCT(DST, SRC, CHANNELS)\
	template <typename TPL_OPERATOR>\
struct GCU_StArithmFct<TPL_OPERATOR, CHANNELS, DST, SRC>\
{\
	static __host__ void DoTex2_Fl1(dim3 Blocks, dim3 grids, DST * _dest, uint _width,uint _height, char _channels, float4 _scale)						\
{	GCUDA_KRNL_ARITHM_2TEX_##SRC<TPL_OPERATOR, CHANNELS, DST, SRC> <<<Blocks, grids>>>((DST *)_dest,_width,_height,_channels,_scale); \
	printf("\nCalling operator: GCUDA_KRNL_ARITHM_2TEX_%s\n",#SRC);\
}	\
	static __host__ void DoTex2_Fl1_Mask(dim3 Blocks, dim3 grids, DST * _dest, uint _width,uint _height, char _channels, float4 _scale)						\
{	GCUDA_KRNL_ARITHM_2TEX_MASK_##SRC<TPL_OPERATOR, CHANNELS, DST, SRC> <<<Blocks, grids>>>((DST *)_dest,_width,_height,_channels,_scale); }	\
	static __host__ void DoTex1_Scalar1_Fl1(dim3 Blocks, dim3 grids, DST * _dest, uint _width,uint _height, char _channels, float4 _scale, SRC & _scalar)\
{	GCUDA_KRNL_ARITHM_1TEX_1SCALAR_##SRC<TPL_OPERATOR, CHANNELS, DST, SRC> <<<Blocks, grids>>>((DST *)_dest,_width,_height,_channels,_scale, _scalar); }	\
	static __host__ void DoTex1_Scalar1_Fl1_Mask(dim3 Blocks, dim3 grids, DST * _dest, uint _width,uint _height, char _channels, float4 _scale, SRC & _scalar)\
{	GCUDA_KRNL_ARITHM_1TEX_1SCALAR_MASK_##SRC<TPL_OPERATOR, CHANNELS, DST, SRC> <<<Blocks, grids>>>((DST *)_dest,_width,_height,_channels,_scale, _scalar); }	\
	static __host__ void LoadIntoTex_A(IplImage* _IplSrc)																							\
{	\
	cudaBindTextureToArray (texA_##SRC##_2, (cudaArray*)gcuPreProcess(_IplSrc, GCU_INPUT, CU_MEMORYTYPE_ARRAY, &texA_##SRC##_2.channelDesc));\
	printf("\nDataloaded into textureA:%s%s_2 ", "texA_", #SRC);\
}\
	static __host__ void LoadIntoTex_B(IplImage* _IplSrc)																							\
{	cudaBindTextureToArray (texB_##SRC##_2, (cudaArray*)gcuPreProcess(_IplSrc, GCU_INPUT, CU_MEMORYTYPE_ARRAY, &texB_##SRC##_2.channelDesc));}\
	static __host__ void LoadIntoTex_C(IplImage* _IplSrc)																							\
{	cudaBindTextureToArray (texC_##SRC##_2, (cudaArray*)gcuPreProcess(_IplSrc, GCU_INPUT, CU_MEMORYTYPE_ARRAY, &texC_##SRC##_2.channelDesc));}\
	static __host__ void LoadIntoTex_Mask(IplImage* _IplSrc)																							\
{	cudaBindTextureToArray (texMASK_##SRC##_2, (cudaArray*)gcuPreProcess(_IplSrc, GCU_INPUT, CU_MEMORYTYPE_ARRAY, &texMASK_##SRC##_2.channelDesc));}\
};

/**	*	\brief Defines operators {Add} for the given I/O types and channels.
*	\author Yannick Allusse
*	\sa CUDA_ARITHM_OPER_TEX_FCT__ALL_TYPE, CUDA_ARITHM_OPER_TEX_FCT__ALL_CHANNELS, CUDA_ARITHM_OPER_TEX_FCT
*/
#define CUDA_ARITHM_OPER_TEX_FCT__ALL_OPER(DST, SRC, CHANNELS)\
	CUDA_ARITHM_OPER_TEX_FCT(DST, SRC,CHANNELS);

/**	*	\brief Defines all operators(CUDA_ARITHM_OPER_TEX_FCT__ALL_OPER) for all channels number(CUDA_ARITHM_OPER_TEX_FCT__ALL_CHANNELS) and for the given types DST and SRC.
*	\author Yannick Allusse
*	\sa CUDA_ARITHM_OPER_TEX_FCT__ALL_TYPE, CUDA_ARITHM_OPER_TEX_FCT__ALL_OPER, CUDA_ARITHM_OPER_TEX_FCT
*/
#define CUDA_ARITHM_OPER_TEX_FCT__ALL_CHANNELS(DST, SRC)\
	CUDA_ARITHM_OPER_TEX_FCT__ALL_OPER(DST##1,SRC##1,1);\
	CUDA_ARITHM_OPER_TEX_FCT__ALL_OPER(DST##4,SRC##4,4);
//CUDA_ARITHM_OPER_TEX_FCT__ALL_OPER(DST##2,SRC##2,2); //GCU_USE_CHANNEL_2

/**	*	\brief Defines all operators for all channels number(see CUDA_ARITHM_OPER_TEX_FCT__ALL_CHANNELS) and for the following types {uchar, char, uint, int, float, double}
*	\author Yannick Allusse
*	\sa CUDA_ARITHM_OPER_TEX_FCT__ALL_CHANNELS, CUDA_ARITHM_OPER_TEX_FCT__ALL_OPER, CUDA_ARITHM_OPER_TEX_FCT
*/
#if GCU_USE_CHAR
CUDA_ARITHM_OPER_TEX_FCT__ALL_CHANNELS(uchar,uchar);
CUDA_ARITHM_OPER_TEX_FCT__ALL_CHANNELS(char,char);
#endif
#if GCU_USE_SHORT
CUDA_ARITHM_OPER_TEX_FCT__ALL_CHANNELS(ushort,ushort);
CUDA_ARITHM_OPER_TEX_FCT__ALL_CHANNELS(short,short);
#endif
#if GCU_USE_INT
CUDA_ARITHM_OPER_TEX_FCT__ALL_CHANNELS(int,int);
CUDA_ARITHM_OPER_TEX_FCT__ALL_CHANNELS(uint,uint);
#endif
#if GCU_USE_FLOAT
CUDA_ARITHM_OPER_TEX_FCT__ALL_CHANNELS(float,float);
#endif

#endif







enum GCU_ALU_TYPE
{
	GCU_ARITHM,
	GCU_LOGIC
};

#define GCU_FORCE_BUFFER_USE 1

template <typename TPL_OPERATOR, int channels, typename TPLSrcType,  typename TPLDstType>
void CudaArithm_SWITCHALL(
						  GCU_ALU_TYPE ALUType,
						  CvArr* src1,
						  CvArr* src2,
						  CvArr* dst,
						  CvArr* mask,
						  float _scale=1.,
						  float4 * _Scalar=NULL,
						  int PixelPackageRatio=1

						  )
{
	//prepare settings	

	//unsigned int channels = dst->nChannels;
	//const unsigned int DataN = width * height * channels;
	//const unsigned int DataSize = DataN * sizeof(unsigned char);
	//used for 3 channels images
	TPLSrcType * d_src1 = NULL;
	TPLSrcType * d_src2 = NULL;
	TPLSrcType * d_mask = NULL;

	//Check inputs is done in the cv_cu.cpp file, to manage exceptions
	PROF_ARITHM(CUT_SAFE_CALL(cutCreateTimer(&hTimer)););	
	//=====================
	//prepare source


	unsigned int width	= gcuGetWidth(dst);
	unsigned int height = gcuGetHeight(dst);

	if(channels==3 || GCU_FORCE_BUFFER_USE)
	{
		if(src1)
		{
			d_src1		= (TPLSrcType*)gcuPreProcess(src1, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
			CvSize Size = gcuGetDataDscSize(src1,CU_MEMORYTYPE_DEVICE);
			width		= Size.width;
			height		= Size.height;
		}
		if(src2)
			d_src2 = (TPLSrcType*)gcuPreProcess(src2, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
		if(mask)
			d_mask = (TPLSrcType*)gcuPreProcess(mask, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	}
	else
	{
#if CUDA_ARITHM_TPL_USE_TEX
		if(src1)
		{
			GCU_StArithmFct < StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType>::LoadIntoTex_A(src1);
			CvSize Size = gcuGetDataDscSize(src1,CU_MEMORYTYPE_DEVICE);
			width = Size.width;
			height = Size.height;
		}
		if(src2)
			GCU_StArithmFct < StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType>::LoadIntoTex_B(src2);
		if(mask)
			GCU_StArithmFct< StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType >::LoadIntoTex_Mask(mask);
#endif
	}

	PROF_ARITHM(
		GCU_CUDA_SAFE_CALL( cudaThreadSynchronize());
	CUT_SAFE_CALL(  cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer);
	printf("gcuPreProcess() time (average) : %f msec\n",timerValue);
	);
	//=====================

	//prepare ouput========
	//output is always in CUDA_BUFFER
	TPLDstType * d_result = (TPLDstType *)gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
	//=====================



	//prepare parameters
	//=================
	gcudaThreadSynchronize();

	dim3 threads(16,16,1);//default
	dim3 blocks = dim3(iDivUp(width,threads.x),
		iDivUp(height, threads.y),
		1);

	

	#if _DEBUG
	printf("DilateSharedKernel===\n");
	printf("- width: %d\n", width);
	printf("- height: %d\n", height);
	printf("- channels: %d\n", channels);
	printf("- height: %d\n", height);
	//printf("- DataN: %d\n", DataN);
	//printf("- DataSize: %d\n", DataSize);
	printf("- threads: %d %d %d\n", threads.x, threads.y, threads.z);
	printf("- blocks: %d %d %d\n", blocks.x, blocks.y, blocks.z);
#endif

#if 1 //simule filter
	if(1)
	{

		gcudaThreadSynchronize();
		//size_t dst_pitch=0;
		float4 Scale;
		Scale.x = Scale.y=Scale.z=Scale.w=_scale;
		TPLSrcType KernelScalar;
		if(_Scalar)
		{
			for( int i = 0; i < channels; i++)
			{
				(&(KernelScalar.x))[i] = (&(_Scalar->x))[i];
			}
		}

		switch(ALUType)
		{
		case GCU_ARITHM: //ARITHMETIC OPS
			//call processing operator
			if(_Scalar)
			{
				if(channels==3|| GCU_FORCE_BUFFER_USE)
				{
					if(mask)
						GCUDA_KRNL_ARITHM_1BUFF_MASK<StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_result, (TPLSrcType *)d_src1, (TPLSrcType *)d_mask, width,height,Scale, KernelScalar);
					else
						GCUDA_KRNL_ARITHM_1BUFF<StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_result, (TPLSrcType *)d_src1, width,height,Scale, KernelScalar);
				}
#if CUDA_ARITHM_TPL_USE_TEX
				else
				{
					if(mask)
						GCU_StArithmFct<StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType>::DoTex1_Scalar1_Fl1_Mask(blocks, threads,(TPLDstType *)d_result,width,height,channels,Scale, KernelScalar);
					else
						GCU_StArithmFct<StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType>::DoTex1_Scalar1_Fl1(blocks, threads,(TPLDstType *)d_result,width,height,channels,Scale, KernelScalar);
				}
#endif
			}
			else
			{
				if(channels==3|| GCU_FORCE_BUFFER_USE)
				{
					if(mask)
						GCUDA_KRNL_ARITHM_2BUFF_MASK<StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_result, d_src1, d_src2, d_mask, width,height,Scale);
					else
						GCUDA_KRNL_ARITHM_2BUFF<StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_result, d_src1, d_src2, width,height,Scale);
				}
#if CUDA_ARITHM_TPL_USE_TEX					
				else
				{
					if(mask)
						GCU_StArithmFct<StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType>::DoTex2_Fl1_Mask(blocks, threads,(TPLDstType *)d_result,width,height,channels,Scale);
					else
						GCU_StArithmFct<StArithmFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType>::DoTex2_Fl1(blocks, threads,(TPLDstType *)d_result,width,height,channels,Scale);
				}
#endif
			}
			break;
		case GCU_LOGIC:
			//call processing operator
			if(_Scalar)
			{
				if(channels==3|| GCU_FORCE_BUFFER_USE)
				{
					if(mask)
						GCUDA_KRNL_ARITHM_1BUFF_MASK<StLogicFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_result, (TPLSrcType *)d_src1, (TPLSrcType *)d_mask, width,height,Scale, KernelScalar);
					else
						GCUDA_KRNL_ARITHM_1BUFF<StLogicFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_result, (TPLSrcType *)d_src1, width,height,Scale, KernelScalar);
				}
#if CUDA_ARITHM_TPL_USE_TEX
				else
				{
					/*		if(mask)
					GCU_StArithmFct<StLogicFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType>::DoTex1_Scalar1_Fl1_Mask(blocks, threads,(TPLDstType *)d_result,width,height,channels,Scale, KernelScalar);
					else
					GCU_StArithmFct<StLogicFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType>::DoTex1_Scalar1_Fl1(blocks, threads,(TPLDstType *)d_result,width,height,channels,Scale, KernelScalar);
					*/				
				}
#endif
			}
			else
			{
				if(channels==3|| GCU_FORCE_BUFFER_USE)
				{
					if(mask)
						GCUDA_KRNL_ARITHM_2BUFF_MASK<StLogicFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_result, (TPLSrcType *)d_src1, (TPLSrcType *)d_src2, (TPLSrcType *)d_mask, width,height,Scale);
					else
						GCUDA_KRNL_ARITHM_2BUFF<StLogicFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType> <<<blocks, threads>>>((TPLDstType *)d_result, (TPLSrcType *)d_src1, (TPLSrcType *)d_src2, width,height,Scale);//, KernelScalar);
				}
#if CUDA_ARITHM_TPL_USE_TEX
				else
				{
					/*
					if(mask)
					GCU_StArithmFct<StLogicFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType>::DoTex2_Fl1_Mask(blocks, threads,(TPLDstType *)d_result,width,height,channels,Scale);
					else
					GCU_StArithmFct<StLogicFilterTex<channels, TPL_OPERATOR>, channels, TPLDstType,TPLSrcType>::DoTex2_Fl1(blocks, threads,(TPLDstType *)d_result,width,height,channels,Scale);
					*/
				}
#endif
			}
			break;
		}
	}


	gcudaThreadSynchronize();
#endif//simule
	CUT_CHECK_ERROR("Kernel execution failed");

	PROF_ARITHM(
		CUT_SAFE_CALL(  cutStopTimer(hTimer));
	timerValue = cutGetTimerValue(hTimer) / NUM_ITERATIONS;
	printf("CudaAdd() time (average) : %f msec //%f MB/sec\n", timerValue, DataSize / (1048576.0 * timerValue * 0.001));
	);    


	//clean output
	gcuPostProcess(dst);
	//=====================

	//=====================
	//clean source
	if(src1)
		gcuPostProcess(src1);
	if(src2)
		gcuPostProcess(src2);
	if(mask)
		gcuPostProcess(mask);

	//close operator
	PROF_ARITHM(CUT_SAFE_CALL(cutDeleteTimer(hTimer)););
}

#endif//_GPUCV_COMPILE_CUDA
