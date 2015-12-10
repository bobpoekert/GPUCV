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
/** \brief Compute the cvIntegral (Sum Array table) operator
\author Source version come from CUDPP library, updated by Yannick Allusse
*/
#include <cvgcu/config.h>
#include <cxcoregcu/oper_array/transform_permut/split.filter.h>
#include <cxcoregcu/oper_array/transform_permut/merge.filter.h>
#include <cxcoregcu/oper_array/linear_algebra/transpose.filter.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>



//Linux MakefileCuda can not find CUDPP argument, so set it to true by default.
#ifdef  _GPUCV_CUDA_SUPPORT_CUDPP
#include <cudpp/cudpp.h>
#include <cutil.h>
#include <cuda_gl_interop.h>

//int width = 0;
//int height = 0;
size_t d_satPitch = 0;
size_t d_satPitchInElements = 0;


#if 0//_DEBUG
#define GCU_INTEGRAL_DEBUG(FCT)FCT	
#else
#define GCU_INTEGRAL_DEBUG(FCT)	
#endif


CUDPPConfiguration config = { 
	CUDPP_SCAN, 
	CUDPP_ADD, 
	CUDPP_FLOAT, 
	CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE 
};
CUDPPHandle scanPlan;

void *d_SATs[2][3] = {
	NULL, NULL,
	NULL, NULL,
	NULL, NULL
};
unsigned int timer[3];

_GPUCV_CVGCU_EXPORT_CU
__host__ void initializeSAT(int width, int height, int Channels, unsigned int glDepth)
{
	GCU_INTEGRAL_DEBUG(printf("\nInitializeSAT: width=%d, height=%d, channels=%d", width, height, Channels));

	//if(d_SATs[0][1]==NULL)//we init
	{
		//config.maxNumElements = width;
		//config.maxNumRows = height;

		int elemSize = gcuGetGLTypeSize(glDepth);

		size_t dpitch = width * elemSize;

		d_SATs[0][1] = d_SATs[0][2] = d_SATs[1][1] = d_SATs[1][2] = NULL;

		switch(Channels)
		{
		case 4: //not done yet...
		case 3:
			GCU_CUDA_SAFE_CALL( cudaMallocPitch( (void**) &d_SATs[0][2], &d_satPitch, dpitch, height));
			GCU_CUDA_SAFE_CALL( cudaMallocPitch( (void**) &d_SATs[1][2], &d_satPitch, dpitch, height));
		case 2:
			GCU_CUDA_SAFE_CALL( cudaMallocPitch( (void**) &d_SATs[0][1], &d_satPitch, dpitch, height));
			GCU_CUDA_SAFE_CALL( cudaMallocPitch( (void**) &d_SATs[1][1], &d_satPitch, dpitch, height));
		case 1:
			GCU_CUDA_SAFE_CALL( cudaMallocPitch( (void**) &d_SATs[0][0], &d_satPitch, dpitch, height));
			GCU_CUDA_SAFE_CALL( cudaMallocPitch( (void**) &d_SATs[1][0], &d_satPitch, dpitch, height));
		}
		d_satPitchInElements = d_satPitch / elemSize;

		if(glDepth==GL_FLOAT)
			config.datatype = CUDPP_FLOAT;
		//	else if(glDepth==GL_DOUBLE)
		//		config.datatype = CUDPP_DOUBLE;
		else if(glDepth==GL_INT)
			config.datatype = CUDPP_INT;
		else if(glDepth==GL_UNSIGNED_INT)
			config.datatype = CUDPP_UINT;
		else
		{
			printf("initializeSAT()=> wrong destination format.\n");
		}

		if (CUDPP_SUCCESS != cudppPlan(&scanPlan, config, width, height, d_satPitchInElements))
		{
			printf("Error creating CUDPPPlan.\n");
		}

		//CUT_SAFE_CALL(cutCreateTimer(&timer[0]));
		//CUT_SAFE_CALL(cutCreateTimer(&timer[1]));
		//CUT_SAFE_CALL(cutCreateTimer(&timer[2]));
	}
}

_GPUCV_CVGCU_EXPORT_CU
__host__ void finalizeSAT(bool _preserveMemory)
{
	if (CUDPP_SUCCESS != cudppDestroyPlan(scanPlan))
	{
		printf("Error creating CUDPPPlan.\n");
	}
	if(_preserveMemory)
	{
		for(int i =0; i < 3; i++)
		{
			for(int j =0; j < 3; j++)
				if(d_SATs[j][i])
				{
					gcudaFree(d_SATs[j][i]);
				}
				//cutDeleteTimer(timer[i]);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
/**http://forums.nvidia.com/lofiversion/index.php?t168384.html
\brief Sum Array Table fuction based on CUDPP library.
\bug Does not work on GTX 480, see http://forums.nvidia.com/lofiversion/index.php?t168384.html
*/
_GPUCV_CVGCU_EXPORT_CU
void gcuSAT(CvArr * _src, CvArr * _dst)//, int radius) 
{
	void *in_data=NULL;
	void *out_data=NULL;

	int src_width		= gcuGetWidth(_src);
	int src_height		= gcuGetHeight(_src);
	int src_Depth		= gcuGetGLDepth(_src);
	int src_nChannels	= gcuGetnChannels(_src);

	int dst_width		= src_width;//gcuGetWidth(_dst);
	int dst_height		= src_height;//gcuGetHeight(_dst);
	int dst_Depth		= gcuGetGLDepth(_dst);
	int dst_nChannels	= gcuGetnChannels(_dst);
	GCU_INTEGRAL_DEBUG(
		printf("\ngcuSAT source: width=%d, height=%d, channels=%d",  src_width,  src_height,  src_nChannels);
		printf("\ngcuSAT destination: width=%d, height=%d, channels=%d", dst_width, dst_height, dst_nChannels);
	);

	in_data		= gcuPreProcess(_src, GCU_INPUT, CU_MEMORYTYPE_DEVICE, NULL);
	out_data	= gcuPreProcess(_dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE, NULL);

	dim3 threads(16,16,1);
	dim3 blocks = dim3(iDivUp(dst_width,threads.x), iDivUp(dst_height,threads.y), 1);

	//cutResetTimer(timer[0]);
	//cutResetTimer(timer[1]);
	//cutResetTimer(timer[2]);


	//CUT_SAFE_CALL(cutStartTimer(timer[0]));

	if(src_Depth != GL_FLOAT && src_Depth != GL_UNSIGNED_BYTE)
	{
		printf("gcuSAT() Error format must be GL_FLOAT or GL_UNSIGNED_BYTE");
	}
	//=======================================
	//split input image if requierd
	//=======================================
	//result is d_SATs[0][?]
	/*	if(src_Depth == GL_FLOAT && src_nChannels==1)
	{
	TmpFloatData = d_SATs[0][0];
	d_SATs[0][0] = (float *)in_data;
	}
	else
	*/	{
#define GCU_INTEGRAL_SPLIT_SWITCH_FCT(CHANNELS, SRC_TYPE, DST_TYPE)\
	gcuSplitKernel_SubImage<CHANNELS,DST_TYPE, SRC_TYPE, SRC_TYPE##CHANNELS> <<<blocks, threads>>> \
	((SRC_TYPE*)	in_data, (DST_TYPE*)d_SATs[0][0], (DST_TYPE*)d_SATs[0][1], (DST_TYPE*)d_SATs[0][2],(DST_TYPE*)NULL,\
	src_width, src_height, dst_width, dst_height, 0.,1.);

		GCU_MULTIPLEX_CONVERT_ALLCHANNELS_ALLFORMAT(GCU_INTEGRAL_SPLIT_SWITCH_FCT, src_nChannels, src_Depth, dst_Depth);
	}
	CUT_CHECK_ERROR("de-interleave");
	GCU_INTEGRAL_DEBUG(gcuShowImage("src_image",	src_width, src_height, gcuGetCVDepth(_src), src_nChannels, in_data, gcuGetGLTypeSize(gcuGetGLDepth(_src)),1.));
	GCU_INTEGRAL_DEBUG(if(d_SATs[0][0])gcuShowImage("de-interleave-0",dst_width, dst_height, gcuGetCVDepth(_dst), 1			 , d_SATs[0][0], gcuGetGLTypeSize(gcuGetGLDepth(_dst)),1./64.));
	GCU_INTEGRAL_DEBUG(if(d_SATs[0][1])gcuShowImage("de-interleave-1",dst_width, dst_height, gcuGetCVDepth(_dst), 1			 , d_SATs[0][1], gcuGetGLTypeSize(gcuGetGLDepth(_dst)),1./16.));
	GCU_INTEGRAL_DEBUG(if(d_SATs[0][2])gcuShowImage("de-interleave-2",dst_width, dst_height, gcuGetCVDepth(_dst), 1			 , d_SATs[0][2], gcuGetGLTypeSize(gcuGetGLDepth(_dst)),1./16.));
	//CUT_SAFE_CALL(cutStopTimer(timer[0]));

	//=======================================
	//process scanning of rows
	//=======================================
	//result is d_SATs[1][?]
	//CUT_SAFE_CALL(cutStartTimer(timer[1]));
	// scan rows
	//result is d_SATs[1][?]
	switch(src_nChannels)
	{
	case 4: 
		//not done yet ????
	case 3:
		//cudppMultiScan(scanPlan,d_SATs[1][2], d_SATs[0][2], dst_width, dst_height);
		cudppScan(scanPlan,d_SATs[1][2], d_SATs[0][2], dst_width* dst_height);
		CUT_CHECK_ERROR("scan 3");
	case 2:
		//cudppMultiScan(scanPlan,d_SATs[1][1], d_SATs[0][1], dst_width, dst_height);
		cudppScan(scanPlan,d_SATs[1][1], d_SATs[0][1], dst_width* dst_height);
		CUT_CHECK_ERROR("scan 2");
	case 1:
		//cudppMultiScan(scanPlan,d_SATs[1][0], d_SATs[0][0], dst_width, dst_height);
		cudppScan(scanPlan,d_SATs[1][0], d_SATs[0][0], dst_width* dst_height);
		CUT_CHECK_ERROR("scan 1");
	}
	//CUT_SAFE_CALL(cutStopTimer(timer[1]));
	GCU_INTEGRAL_DEBUG(gcuShowImage("process scanning of rows", dst_width, dst_height, gcuGetCVDepth(_dst), 1, d_SATs[1][0], sizeof(float),1.));

	//+++++++++++++++++++++++++++++++++++++++


	//=======================================
	//transpose so columns become rows
	//=======================================
	//result is d_SATs[0][?]
	//CUT_SAFE_CALL(cutStartTimer(timer[2]));
	switch(src_nChannels)
	{
	case 4: 
		//not done yet
	case 3:	
		 gcuTransposeKernel_Shared<float1, float1, 1, 16,16><<<blocks, threads, 0>>>((float1*)d_SATs[1][2], (float1*)d_SATs[0][2], /*d_satPitchInElements,*/ dst_width, dst_height);
		CUT_CHECK_ERROR("transpose 3");
	case 2: 
		 gcuTransposeKernel_Shared<float1, float1, 1, 16,16><<<blocks, threads, 0>>>((float1*)d_SATs[1][1], (float1*)d_SATs[0][1], /*d_satPitchInElements,*/ dst_width, dst_height);
		CUT_CHECK_ERROR("transpose 2");
	case 1: 
		 gcuTransposeKernel_Shared<float1, float1, 1, 16,16><<<blocks, threads, 0>>>((float1*)d_SATs[1][0], (float1*)d_SATs[0][0], /*d_satPitchInElements,*/ dst_width, dst_height);
		CUT_CHECK_ERROR("transpose 1");
	}
	//CUT_SAFE_CALL(cutStopTimer(timer[2]));
	GCU_INTEGRAL_DEBUG(gcuShowImage("transpose so columns become rows", dst_width, dst_height, gcuGetCVDepth(_dst), 1, d_SATs[0][0], gcuGetGLTypeSize(gcuGetGLDepth(_dst)),1.));
	//+++++++++++++++++++++++++++++++++++++++


	//=======================================
	//process scanning of columns
	//=======================================
	//result is d_SATs[1][?]
	//CUT_SAFE_CALL(cutStartTimer(timer[1]));
	/*	if(src_Depth == GL_FLOAT && src_nChannels==1)
	{
	cudppMultiScan(scanPlan, out_data, d_SATs[0][0], dst_width, dst_height);
	CUT_CHECK_ERROR("scan 1");
	}
	else
	*/	{
		switch(src_nChannels)
		{
		case 4: 
			//not done yet ????
		case 3:
			cudppMultiScan(scanPlan,d_SATs[1][2], d_SATs[0][2], dst_width, dst_height);
			CUT_CHECK_ERROR("scan 3");
		case 2:
			cudppMultiScan(scanPlan,d_SATs[1][1], d_SATs[0][1], dst_width, dst_height);
			CUT_CHECK_ERROR("scan 2");
		case 1:
			cudppMultiScan(scanPlan,d_SATs[1][0], d_SATs[0][0], dst_width, dst_height);
			CUT_CHECK_ERROR("scan 1");
		}
	}
	//CUT_SAFE_CALL(cutStopTimer(timer[1]));
	GCU_INTEGRAL_DEBUG(gcuShowImage("process scanning of columns", dst_width, dst_height, gcuGetCVDepth(_dst), 1, d_SATs[1][0], gcuGetGLTypeSize(gcuGetGLDepth(_dst)),1.));
	//++++++++++++++++++++++++++++++++++++++++


	//=======================================
	//merge channels
	//=======================================
	//check if we need to merge images
	//result is out_data
	//CUT_SAFE_CALL(cutStartTimer(timer[0]));
	/*	if(src_Depth == GL_FLOAT && src_nChannels==1)
	{
	// = d_SATs[0][0];
	d_SATs[0][0] = TmpFloatData;//restore pointer
	}
	else
	*/	{
#define GCU_INTEGRAL_MERGE_SWITCH_FCT(CHANNELS, SRC_TYPE, DST_TYPE)\
	gcuMergeKernel_SubImage<CHANNELS,DST_TYPE, SRC_TYPE,DST_TYPE##CHANNELS> <<<blocks, threads>>> \
	((DST_TYPE##CHANNELS*)	out_data, (SRC_TYPE*)d_SATs[1][0], (SRC_TYPE*)d_SATs[1][1], (SRC_TYPE*)d_SATs[1][2],(SRC_TYPE*)NULL,\
	src_width, src_height, dst_width+1, dst_height+1, 0, 1.);

		GCU_MULTIPLEX_ALLCHANNELS_ALLFORMAT(GCU_INTEGRAL_MERGE_SWITCH_FCT, src_nChannels, dst_Depth);//, dst_Depth);
	}
	CUT_CHECK_ERROR("interleave");
	//CUT_SAFE_CALL(q(timer[0]));
	GCU_INTEGRAL_DEBUG(gcuShowImage("merge channels", dst_width+1, dst_height+1, gcuGetCVDepth(_dst), src_nChannels, out_data, gcuGetGLTypeSize(gcuGetGLDepth(_dst)), 1.));
	//+++++++++++++++++++++++++++++++++++++++++

#if 0//_DEBUG
	printf("Total: %0.2f ms | (de)Interleave: %0.2f ms | Multiscan: %0.2f ms | Transpose: %0.2f ms\n",
		cutGetTimerValue(timer[0])+cutGetTimerValue(timer[1])+cutGetTimerValue(timer[2]),
		cutGetTimerValue(timer[0]), cutGetTimerValue(timer[1]), cutGetTimerValue(timer[2]));
#endif
	gcuPostProcess(_dst);
	gcuPostProcess(_src);
}
#endif//_GPUCV_CUDA_SUPPORT_CUDPP
