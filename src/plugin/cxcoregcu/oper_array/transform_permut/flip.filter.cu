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

/*
* The function cvFlip flips the array in one of different 3 ways (row and column indices are 0-based) 
*/
#include <cxcoregcu/config.h>
#include <typeinfo>
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>
#include <cxcoregcu/oper_array/transform_permut/flip.filter.h>

_GPUCV_CXCOREGCU_EXPORT_CU 
 
void gcuFlip(CvArr* src, CvArr* dst, int flip_mode=0)
{ 
	unsigned int width		= gcuGetWidth(src);
	unsigned int height		= gcuGetHeight(src);
	unsigned int channels	= gcuGetnChannels(src);
	unsigned int depth		= gcuGetGLDepth(src);
	unsigned char * d_result	= (unsigned char *)gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
	unsigned char * d_src		= (unsigned char *)gcuPreProcess(src, GCU_INPUT, CU_MEMORYTYPE_DEVICE);

	dim3 threads(16,16,1);
	dim3 blocks = dim3(iDivUp(width,threads.x),iDivUp(height,threads.y), 1);  

	//kernel executed...
	CUT_CHECK_ERROR("Kernel execution could not start");

#define GCUFLIP_SWITCH_FCT(CHANNELS, DST_TYPE, SRC_TYPE)\
	gcudaKernel_Flip<DST_TYPE##CHANNELS, SRC_TYPE##CHANNELS><<<blocks, threads>>>((SRC_TYPE##CHANNELS*)d_src, (DST_TYPE##CHANNELS *)d_result, width, height,flip_mode);
 
	GCU_MULTIPLEX_ALLCHANNELS_ALLFORMAT(GCUFLIP_SWITCH_FCT, channels, depth);

	//kernel executed...
	CUT_CHECK_ERROR("Kernel execution failed");

	gcuPostProcess(src);
	gcuPostProcess(dst);
}

