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
#include <cxcoregcu/oper_array/linear_algebra/transpose.filter.h>


_GPUCV_CXCOREGCU_EXPORT_CU
void gcuTranspose(CvArr* src, CvArr* dst)
{
	char * d_dst = (char*)gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE, NULL);
	char * d_src = (char*)gcuPreProcess(src, GCU_INPUT, CU_MEMORYTYPE_DEVICE, NULL);
//	size_t Pitch = 0;// is calculated 
	//gcuGetPitch(src)/ sizeof(uchar1); 

	unsigned int width		= gcuGetWidth(dst);
	unsigned int height		= gcuGetHeight(dst);
	unsigned int channels	= gcuGetnChannels(dst);
	unsigned int depth		= gcuGetGLDepth(dst);

	gcudaThreadSynchronize();
	dim3 threads(16,16);
	//???channels.
	dim3 blocks = dim3(iDivUp(width*channels,threads.x), iDivUp(height,threads.y), 1);

#define GCUSPLIT_SWITCH_FCT(CHANNELS, DST_TYPE, SRC_TYPE)\
	gcuTransposeKernel_Shared<SRC_TYPE##CHANNELS, DST_TYPE##CHANNELS,CHANNELS,16,16> <<<blocks, threads>>> \
	((SRC_TYPE##CHANNELS*)d_src, (DST_TYPE##CHANNELS*)d_dst, width, height);


	GCU_MULTIPLEX_ALLCHANNELS_ALLFORMAT(GCUSPLIT_SWITCH_FCT, channels, depth);

	//kernel executed...
	CUT_CHECK_ERROR("Kernel execution failed"); 
	
	gcudaThreadSynchronize();
	
	gcuPostProcess(src);
	gcuPostProcess(dst);
}  


