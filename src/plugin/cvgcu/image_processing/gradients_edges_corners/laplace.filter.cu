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
#include <cvgcu/config.h>
#include <cvgcu/image_processing/gradients_edges_corners/sobelFilter.kernel.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>

#if 0//PUCV_COMPILE_CUDA
#include <GPUCVCuda/base_kernels/tpl_convolutions.kernels.h>
#include <assert.h>
#include <cutil_inline.h>


/**
\return false if convolution kernel has not been calculated
\param src => Kernel will be faster if type is 32S/32U/32F
\param dst => type must be 16S/16U/32S/32U/32F.
\note In fact, all type are supported but results and performances are unknown for 8U/8S/...
*/
_GPUCV_CVGCU_EXPORT_CU
bool gcuLaplace(void* src, void* dst, int xorder, int yorder, int aperture_size)
{
	unsigned int width		= gcuGetWidth(dst);
	unsigned int height		= gcuGetHeight(dst);
	
	//prepare input/ouput========
	void* d_src	= gcuPreProcess(src, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	void* d_dst = gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
	
    float *h_Kernel_Horiz = (float *)malloc(KERNEL_LENGTH * sizeof(float));
	float *h_Kernel_Vert = (float *)malloc(KERNEL_LENGTH * sizeof(float));
	
	//use float temporary image
	void *d_Buffer = NULL;
	cudaMalloc((void **)&d_Buffer , width * height * sizeof(float));
	//=====================

	if(aperture_size==-1)//Scharr filter
	{
		h_Kernel_Vert[0] = 3;
		h_Kernel_Vert[1] = 10;
		h_Kernel_Vert[2] = 3;

		h_Kernel_Horiz[0] = -1;
		h_Kernel_Horiz[1] = 0;
		h_Kernel_Horiz[2] = 1;	
	}
	else if ((xorder==0) && (yorder==1) && (aperture_size==3))
	{
		h_Kernel_Vert[0] = 1;
		h_Kernel_Vert[1] = 2;
		h_Kernel_Vert[2] = 1;

		h_Kernel_Horiz[0] = -1;
		h_Kernel_Horiz[1] = 0;
		h_Kernel_Horiz[2] = 1;
	}
	else if ((xorder==1) && (yorder==0) && (aperture_size==3))
	{
		h_Kernel_Vert[0] = 1;
		h_Kernel_Vert[1] = 0;
		h_Kernel_Vert[2] = -1;

		h_Kernel_Horiz[0] = 1;
		h_Kernel_Horiz[1] = 2;
		h_Kernel_Horiz[2] = 1;
	}
	else
	{
		cutilSafeCall( cudaFree(d_Buffer ) );
		free(h_Kernel_Horiz);
		free(h_Kernel_Vert);

		//clean input/output
		gcuPostProcess(dst);
		gcuPostProcess(src);
		return false;
	}

	cudaMemcpyToSymbol(c_Kernel_H, h_Kernel_Horiz, KERNEL_LENGTH * sizeof(float));
	cudaMemcpyToSymbol(c_Kernel_V, h_Kernel_Vert, KERNEL_LENGTH * sizeof(float));


	#define GCU_SOBEL_SWITCH_FCT(CHANNELS, SRC_TYPE, DST_TYPE)\
		convolutionRowsGPU<SRC_TYPE, float>((SRC_TYPE*)d_src,(float*)d_Buffer,width,height);\
		convolutionColumnsGPU<float, DST_TYPE>((float*)d_Buffer,(DST_TYPE*)d_dst,width,height);

	//run kernels for any kind of format...some format might not be used, but it is easier to mange all of them with 
	//this macro than set them manually...
	GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLSRCFORMAT(GCU_SOBEL_SWITCH_FCT, 1, gcuGetGLDepth(src),gcuGetGLDepth(dst));
	
	
	gcudaThreadSynchronize();

    gcudaFree(d_Buffer );
    free(h_Kernel_Horiz);
	free(h_Kernel_Vert);

	//clean input/output
	gcuPostProcess(dst);
	gcuPostProcess(src);
	return true;
}
//=========================================================
#endif//_GPUCV_COMPILE_CUDA