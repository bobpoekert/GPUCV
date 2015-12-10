//CVG_LicenseBegin========================================== ====================
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
#ifndef __GPUCV_CUDA_SET_KERNEL_CU
#define __GPUCV_CUDA_SET_KERNEL_CU

#include <cxcoregcu/config.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <GPUCVCuda/gcu_runtime_api_wrapper.h>
#include <cxcoregcu/oper_array/copy_fill/set.filter.h>

#if _GPUCV_COMPILE_CUDA

_GPUCV_CXCOREGCU_EXPORT_CU
void gcuSet(CvArr* src, gcuScalar &cudavalue, CvArr* mask/*=NULL*/)
{
	unsigned int width		= gcuGetWidth(src);
	unsigned int height		= gcuGetHeight(src);
	unsigned int src_depth	= gcuGetGLDepth(src);
	unsigned int src_depth_size	= gcuGetGLTypeSize (src_depth);
	unsigned int channels	= gcuGetnChannels(src);

	void* d_src = gcuPreProcess(src, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
	
	if(mask==NULL && 
	(cudavalue.val[0]==cudavalue.val[1] && cudavalue.val[1]==cudavalue.val[2] && cudavalue.val[2]==cudavalue.val[3]))
	{//no mask, 'single channel' set
		gcudaMemset(d_src, cudavalue.val[0], width * height* src_depth_size * channels);
	}
	else 
	{//use a kernel...
		float4 TempScalarMask;
		TempScalarMask.x = cudavalue.val[0];
		TempScalarMask.y = cudavalue.val[1];
		TempScalarMask.z = cudavalue.val[2];
		TempScalarMask.w = cudavalue.val[3];

		float4 TempScalarClear;
		TempScalarClear.x = TempScalarClear.y = cudavalue.val[1] = TempScalarClear.z =	TempScalarClear.w = (float)0;

		dim3 threads(16,16,1);
		dim3 blocks = dim3(iDivUp(width,threads.x),iDivUp(height,threads.y), 1);  

		//kernel start...
		CUT_CHECK_ERROR("Kernel execution could not start");
	
		if(mask)
		{		
			void* d_mask = gcuPreProcess(mask, GCU_INPUT, CU_MEMORYTYPE_DEVICE);

		#define GCUSET_MASK_SWITCH_FCT(CHANNELS, DST_TYPE, SRC_TYPE)\
			gcuSetKernel_Mask<CHANNELS, DST_TYPE##CHANNELS><<<blocks, threads>>>((DST_TYPE##CHANNELS *)d_src, (char1*)d_mask,width, height,TempScalarMask,TempScalarClear);
		 
			GCU_MULTIPLEX_ALLCHANNELS_ALLFORMAT(GCUSET_MASK_SWITCH_FCT, channels, src_depth);
		}
		else
		{
		#define GCUSET_SWITCH_FCT(CHANNELS, DST_TYPE, SRC_TYPE)\
			gcuSetKernel<CHANNELS, DST_TYPE##CHANNELS><<<blocks, threads>>>((DST_TYPE##CHANNELS *)d_src, width, height,TempScalarMask);
		 
			GCU_MULTIPLEX_ALLCHANNELS_ALLFORMAT(GCUSET_SWITCH_FCT, channels, src_depth);
		}
		//kernel executed...
		CUT_CHECK_ERROR("Kernel execution failed");
	}

	gcuPostProcess(src);
	if(mask)
		gcuPostProcess(mask);
}
#endif
#endif
