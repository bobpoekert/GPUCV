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
#include <cxcoregcu/config.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>


#if _GPUCV_COMPILE_CUDA

#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCV/oper_enum.h>
#include <cxcoregcu/cxcoregcu_array_arithm.kernel.h>
#include <cxcoregcu/cxcoregcu_statistics.kernel.h>
#include <cxcoregcu/oper_array/discrete_transforms/cxcoregcu_ankit_kernels.h>



#define _GCU_ALLOW_RESHAPE 1
_GPUCV_CXCOREGCU_EXPORT_CU  
void gcuConvertScale(CvArr* src1,CvArr* dst,double s1,double s2)
{
	unsigned int width		= gcuGetWidth(dst);
	unsigned int height		= gcuGetHeight(dst);
	unsigned int src_depth	= gcuGetGLDepth(src1);
	unsigned int dst_depth	= gcuGetGLDepth(dst);
	unsigned int channels	= gcuGetnChannels(dst);

	unsigned int NewChannels = channels;
	if(IS_MULTIPLE_OF(width *channels, 4))
		NewChannels = 4;
	else if(IS_MULTIPLE_OF(width *channels, 2))
		NewChannels = 2;

	if(NewChannels != channels)
	{
		width *=  channels / NewChannels;
	}

	void* d_dst= gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE,NULL);
	void* d_src= gcuPreProcess(src1, GCU_INPUT, CU_MEMORYTYPE_DEVICE,NULL);

	dim3 threads(16, 16,1);
	dim3 blocks =  dim3(iDivUp(width*  channels,threads.x), iDivUp(height,threads.y), 1);

	float4 shift_fl;
	float4 scale_fl;
	scale_fl.x = scale_fl.y = scale_fl.z = scale_fl.w = s1;
	shift_fl.x = shift_fl.y = shift_fl.z = shift_fl.w = s2;

#define GCUCS_SWITCH_FCT(CHANNELS,DST_TYPE,SRC_TYPE)\
	{\
		size_t pitch = gcuGetPitch(dst)*  channels / NewChannels;\
		gcuConvertScaleKernel<CHANNELS,DST_TYPE##CHANNELS, SRC_TYPE##CHANNELS,16,16> <<<blocks, threads>>> \
		((SRC_TYPE##CHANNELS*) d_src,(DST_TYPE##CHANNELS*)d_dst,scale_fl,shift_fl,width,height,pitch);\
	}

	GCU_MULTIPLEX_CONVERT_ALLCHANNELS_ALLFORMAT(GCUCS_SWITCH_FCT, NewChannels , src_depth, dst_depth);

	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	gcuPostProcess(dst);
	gcuPostProcess(src1);
}
#endif
