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
#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>
#include <cxcoregcu/oper_array/transform_permut/split.filter.h>
    

_GPUCV_CXCOREGCU_EXPORT_CU
void gcuSplit(CvArr* src, CvArr* dst0 = NULL, CvArr* dst1 = NULL, CvArr* dst2= NULL, CvArr* dst3= NULL)
{
	//device pointers
	void* d_src  = NULL;
	void* d_dst0 = NULL;
	void* d_dst1 = NULL;
	void* d_dst2 = NULL;
	void* d_dst3 = NULL;
	//------------------

	d_src = gcuPreProcess(src, GCU_INPUT, CU_MEMORYTYPE_DEVICE,NULL);
	if(dst0)
		d_dst0 = gcuPreProcess(dst0, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE,NULL);
	if(dst1)
		d_dst1 = gcuPreProcess(dst1, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE,NULL);
	if(dst2)
		d_dst2 = gcuPreProcess(dst2, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE,NULL);
	if(dst3)
		d_dst3 = gcuPreProcess(dst3, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE,NULL);

	//params
	unsigned int width		= gcuGetWidth(src);
	unsigned int height		= gcuGetHeight(src);
	unsigned int channels	= gcuGetnChannels(src);
	unsigned int depth		= gcuGetGLDepth(src);
	//------------------

	unsigned int pitch = 0;
	dim3 threads(16,16,1);
	dim3 blocks = dim3(iDivUp(width,threads.x), iDivUp(height,threads.y), 1);

#if !GCU_EMULATION_MODE
#define GCUSPLIT_SWITCH_FCT(CHANNELS, DST_TYPE, SRC_TYPE)\
	pitch = (unsigned int)gcuGetPitch(dst0)/sizeof(DST_TYPE);\
	gcuSplitKernel<CHANNELS,DST_TYPE, SRC_TYPE, SRC_TYPE##CHANNELS> <<<blocks, threads>>> \
	((SRC_TYPE*)	d_src, (DST_TYPE*)d_dst0, (DST_TYPE*)d_dst1, (DST_TYPE*)d_dst2,(DST_TYPE*)d_dst3,width, height, pitch, 1.);

	GCU_MULTIPLEX_ALLCHANNELS_ALLFORMAT(GCUSPLIT_SWITCH_FCT, channels, depth);
#endif

	CUT_CHECK_ERROR("Kernel execution failed");

	gcudaThreadSynchronize();
	if(dst0)
		gcuPostProcess(dst0);
	if(dst1)
		gcuPostProcess(dst1);
	if(dst2)
		gcuPostProcess(dst2);
	if(dst3)
		gcuPostProcess(dst3);

	gcuPostProcess(src);
}

