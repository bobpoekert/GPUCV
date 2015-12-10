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
#include <cxcoregcu/oper_array/transform_permut/merge.filter.h>


_GPUCV_CXCOREGCU_EXPORT_CU 
void gcuMerge(CvArr* src0, CvArr* src1, CvArr* src2, CvArr* src3, CvArr* dst)
{
#if 1
	void* d_dst  = NULL;
	void* d_src0 = NULL;
	void* d_src1 = NULL;
	void* d_src2 = NULL;
	void* d_src3 = NULL;

	//get properties...
	unsigned int width		= gcuGetWidth(dst);
	unsigned int height		= gcuGetHeight(dst);
	unsigned int channels	= gcuGetnChannels(dst);
	unsigned int depth		= gcuGetGLDepth(dst);
	//==================

	if(src0)
		d_src0 = gcuPreProcess(src0, GCU_INPUT, CU_MEMORYTYPE_DEVICE,NULL);
	if(src1)
		d_src1 = gcuPreProcess(src1, GCU_INPUT, CU_MEMORYTYPE_DEVICE,NULL);
	if(src2)
		d_src2 = gcuPreProcess(src2, GCU_INPUT, CU_MEMORYTYPE_DEVICE,NULL);
	if(src3)
		d_src3 = gcuPreProcess(src3, GCU_INPUT, CU_MEMORYTYPE_DEVICE,NULL);

	d_dst = gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE,NULL);

	unsigned int pitch = gcuGetPitch(dst)/sizeof(uchar);
	dim3 threads(16,16,1);
	dim3 blocks = dim3(iDivUp(width,threads.x), iDivUp(height,threads.y), 1);

#if !GCU_EMULATION_MODE
#define GCUMERGE_SWITCH_FCT(CHANNELS, DST_TYPE, SRC_TYPE)\
	gcuMergeKernel<CHANNELS,DST_TYPE, SRC_TYPE,DST_TYPE##CHANNELS> <<<blocks, threads>>> ((DST_TYPE##CHANNELS*)		d_dst,\
	(SRC_TYPE*)d_src0,\
	(SRC_TYPE*)d_src1,\
	(SRC_TYPE*)d_src2,\
	(SRC_TYPE*)d_src3,\
	width, height, \
	pitch, 1.)

	GCU_MULTIPLEX_ALLCHANNELS_ALLFORMAT(GCUMERGE_SWITCH_FCT, channels, depth);
	CUT_CHECK_ERROR("Kernel execution failed");
#endif
	gcudaThreadSynchronize();
	if(dst)
		gcuPostProcess(dst);
	if(src0)
		gcuPostProcess(src0);
	if(src1)
		gcuPostProcess(src1);
	if(src2)
		gcuPostProcess(src2);
	if(src3)
		gcuPostProcess(src3);
#endif//0
}



