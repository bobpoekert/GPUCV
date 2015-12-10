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
#ifndef __GPUCV_CUDA_LUT_KERNEL_CU
#define __GPUCV_CUDA_LUT_KERNEL_CU

#include <cxcoregcu/config.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <cxcoregcu/oper_array/lut.filter.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>


#if 1//_GPUCV_COMPILE_CUDA

_GPUCV_CXCOREGCU_EXPORT_CU
void gcuLut(CvArr* src,CvArr* dst,CvArr* lut)
{
#if 1//_GPUCV_DEVELOP_BETA
	unsigned int width		= gcuGetWidth(dst);
	unsigned int height		= gcuGetHeight(dst);
	unsigned int src_depth	= gcuGetGLDepth(src);
	unsigned int dst_depth	= gcuGetGLDepth(dst);
	unsigned int ch			= gcuGetnChannels(dst);
	unsigned int pitch		= 0;
	int delta = 0;

	void* d_dst = gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE,NULL);
	void* d_src = gcuPreProcess(src, GCU_INPUT, CU_MEMORYTYPE_DEVICE,NULL);
	void* d_lut = gcuPreProcess(lut, GCU_INPUT, CU_MEMORYTYPE_DEVICE,NULL);

	dim3 threads(16,16,1);   

	dim3 blocks = dim3(iDivUp(width,threads.x), iDivUp(height,threads.y), 1);

#define GCULUT_SWITCH_FCT(CHANNELS,SRC_TYPE,DST_TYPE)\
	{pitch = gcuGetPitch(src)/(sizeof(SRC_TYPE)*CHANNELS);\
	gcuLutKernel<CHANNELS,SRC_TYPE,DST_TYPE,SRC_TYPE##CHANNELS,DST_TYPE##CHANNELS,16,16><<<blocks, threads>>>((SRC_TYPE##CHANNELS*)d_src,(DST_TYPE##CHANNELS*)d_dst,(DST_TYPE*)d_lut,width, height, pitch,delta);}
#if 1
	if (src_depth == GL_UNSIGNED_BYTE) 
	{
		delta = 0;
	}
	else if(src_depth == GL_BYTE)
	{ 
		delta = 128;
	}
	else     
	{ 
		printf("Unkown type!!");
	}  
  
	GCU_MULTIPLEX_ALLCHANNELS_ALLFORMAT(GCULUT_SWITCH_FCT, ch, dst_depth);

#else
	if (src_depth == GL_UNSIGNED_BYTE)
	{delta = 0 ; 
	switch(ch)  
	{          
	case 1: 
		switch(dst_depth) 																					\
		{ 																										\
		case IPL_DEPTH_8U:  GCULUT_SWITCH_FCT(1, uchar, uchar); break;
		case IPL_DEPTH_8S:  GCULUT_SWITCH_FCT(1, uchar, char); break;
		case IPL_DEPTH_16U:  GCULUT_SWITCH_FCT(1, uchar, uint); break;
		case IPL_DEPTH_16S:  GCULUT_SWITCH_FCT(1, uchar, int); break;
		case IPL_DEPTH_32F:  GCULUT_SWITCH_FCT(1, uchar, float); break; 
		}
		break;
	case 2:
		switch(dst_depth)  																					\
		{  																									\
		case IPL_DEPTH_8U:  GCULUT_SWITCH_FCT(2, uchar, uchar); break;
		case IPL_DEPTH_8S:  GCULUT_SWITCH_FCT(2, uchar, char); break;
		case IPL_DEPTH_16U: GCULUT_SWITCH_FCT(2, uchar, uint); break;
		case IPL_DEPTH_16S:  GCULUT_SWITCH_FCT(2, uchar, int); break;
		case IPL_DEPTH_32F:  GCULUT_SWITCH_FCT(2, uchar, float); break;
		}
		break;
	case 3: 
		switch(dst_depth) 																					\
		{ 		 																								\
		case IPL_DEPTH_8U:  GCULUT_SWITCH_FCT(3, uchar, uchar); break;
		case IPL_DEPTH_8S:  GCULUT_SWITCH_FCT(3, uchar, char); break;
		case IPL_DEPTH_16U:  GCULUT_SWITCH_FCT(3, uchar, uint); break;
		case IPL_DEPTH_16S:  GCULUT_SWITCH_FCT(3, uchar, int); break;
		case IPL_DEPTH_32F:  GCULUT_SWITCH_FCT(3, uchar, float); break;
		} 
		break;  
	case 4: 
		switch(dst_depth) 																					\
		{ 																										\
		case IPL_DEPTH_8U:  GCULUT_SWITCH_FCT(4, uchar, uchar); break;
		case IPL_DEPTH_8S:  GCULUT_SWITCH_FCT(4, uchar, char); break;
		case IPL_DEPTH_16U:  GCULUT_SWITCH_FCT(4, uchar, uint); break;
		case IPL_DEPTH_16S:  GCULUT_SWITCH_FCT(4, uchar, int); break;
		case IPL_DEPTH_32F:  GCULUT_SWITCH_FCT(4, uchar, float); break;
		}
		break;  
	} 
	}
	else
	{delta = 128 ;
	switch(ch)
	{        
	case 1: 
		switch(dst_depth) 																					\
		{ 					 																					\
		case IPL_DEPTH_8U:  GCULUT_SWITCH_FCT(1, char, uchar); break;
		case IPL_DEPTH_8S:  GCULUT_SWITCH_FCT(1, char, char); break;
		case IPL_DEPTH_16U:  GCULUT_SWITCH_FCT(1, char, uint); break;
		case IPL_DEPTH_16S:  GCULUT_SWITCH_FCT(1, char, int); break;
		case IPL_DEPTH_32F:  GCULUT_SWITCH_FCT(1, char, float); break; 
		}
		break;   
	case 2:
		switch(dst_depth)  																					\
		{  																									\
		case IPL_DEPTH_8U:  GCULUT_SWITCH_FCT(2, char, uchar); break;
		case IPL_DEPTH_8S:  GCULUT_SWITCH_FCT(2, char, char); break;
		case IPL_DEPTH_16U: GCULUT_SWITCH_FCT(2, char, uint); break;
		case IPL_DEPTH_16S:  GCULUT_SWITCH_FCT(2, char, int); break;
		case IPL_DEPTH_32F:  GCULUT_SWITCH_FCT(2, char, float); break;
		}
		break;
	case 3: 
		switch(dst_depth) 																					\
		{ 																										\
		case IPL_DEPTH_8U:  GCULUT_SWITCH_FCT(3, char, uchar); break;
		case IPL_DEPTH_8S:  GCULUT_SWITCH_FCT(3, char, char); break;
		case IPL_DEPTH_16U:  GCULUT_SWITCH_FCT(3, char, uint); break;
		case IPL_DEPTH_16S:  GCULUT_SWITCH_FCT(3, char, int); break;
		case IPL_DEPTH_32F:  GCULUT_SWITCH_FCT(3, char, float); break;
		}
		break; 
	case 4: 
		switch(dst_depth) 																					\
		{ 																										\
		case IPL_DEPTH_8U:  GCULUT_SWITCH_FCT(4, char, uchar); break;
		case IPL_DEPTH_8S:  GCULUT_SWITCH_FCT(4, char, char); break;
		case IPL_DEPTH_16U:  GCULUT_SWITCH_FCT(4, char, uint); break;
		case IPL_DEPTH_16S:  GCULUT_SWITCH_FCT(4, char, int); break;
		case IPL_DEPTH_32F:  GCULUT_SWITCH_FCT(4, char, float); break;
		}
		break;
	}

	}

	gcudaThreadSynchronize();

	gcuPostProcess(dst);
	gcuPostProcess(src);
#endif
#endif//beta
}

#endif
#endif
