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
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <GPUCVCuda/base_kernels/tpl_convolutions.kernels.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>


#if _GPUCV_COMPILE_CUDA

//template can't be extern, cause they need to be compiled by CUDA...
template <typename TPLSrcType,  typename TPLDstType>
void CudaSmouthIpl(CvArr* src,CvArr* dst, int smoothtype/*=CV_GAUSSIAN*/,
				   int param1/*=3*/, int param2/*=0*/, double param3/*=0*/, double param4/*=0*/)
{
	unsigned int height	= gcuGetHeight(src);
	unsigned int width	= gcuGetWidth(src);
	unsigned int depth	= gcuGetGLDepth(src);
	unsigned int channels	= gcuGetnChannels(src);

	//Check inputs is done in the cv_cu.cpp file, to manage exceptions

	//=====================
	//prepare source
	void * d_src = gcuPreProcess(src, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	//prepare ouput========
	void * d_result = gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
	//=====================


	//prepare parameters
	//================
	dim3 threads(16,4);
	unsigned int BlockWidth = 80; // must be divisible by 16 for coalescing
	//unsigned int BlockHeight= 1;

#if 1
	if(width == 32 && height == 32)
	{
		threads = dim3(4,4,1);
		BlockWidth = 80*32/512;
	}
	if(width == 1024 && height == 1024)
	{
		threads = dim3(16,8,1);
		BlockWidth = 80;
	}
	else if (width == 2048 && height == 2048)
	{
		threads = dim3(16,16,1);
		BlockWidth = 80;
	}
	dim3 blocks =  dim3(iDivUp(width,(4*BlockWidth)),
		iDivUp(height,threads.y),
		1);
#endif

	//prepare parameters, should stay NULL....
	unsigned char * h_params = NULL;
	unsigned char * d_params = NULL;
	int ParamsNbr = param1*param2;
	h_params = (unsigned char *)malloc(ParamsNbr * sizeof(unsigned char));
	for(int i=0;i< ParamsNbr;i++)
	{
		h_params[i] = 1;
	}
	gcudaMalloc((void**)&d_params, ParamsNbr * sizeof(unsigned char));
	gcudaMemCopyHostToDevice(d_params, h_params, ParamsNbr * sizeof(unsigned char));
	//else all element are ==1, so we don't need to apply specific filter
	//printf("\nElemParam size:%d\n", ParamsNbr * sizeof(unsigned char));
	//=================

	int LocalRadius = (param1-1)/2;//must take care of param2 when not square!!!
	int SharedPitch = ~0x3f&(4*(BlockWidth+2*LocalRadius)+0x3f);
	//int sharedMem = SharedPitch*(threads.y+2*LocalRadius);
	// for the shared kernel, width must be divisible by 4
	width &= ~3;
#if 0
	printf("DilateSharedKernel===\n");
	printf("- width: %d\n", width);
	printf("- height: %d\n", height);
	printf("- channels: %d\n", channels);
	printf("- height: %d\n", height);
	printf("- DataN: %d\n", DataN);
	printf("- DataSize: %d\n", DataSize);
	printf("- threads: %d %d %d\n", threads.x, threads.y, threads.z);
	printf("- BlockWidth: %d\n", BlockWidth);
	printf("- blocks: %d %d %d\n", blocks.x, blocks.y, blocks.z);
	printf("- SharedPitch: %d\n", SharedPitch);
	printf("- sharedMem: %d\n", sharedMem);
	printf("- ParamNbr: %d\n", ParamsNbr);
#endif


#if 0 //simule filter
	//process operator
	float Scale = 1;
	if(channels==1)				
	{//
		switch(smoothtype)
		{
		case CV_BLUR: Scale = 1. / (param1 * param2);
			//printf("\nScale : %f", Scale);
			Scale /=2;
		case CV_BLUR_NO_SCALE:
			if(param1 == param2 && param1 == 3)
				CudaConvKernel3<TPLSrcType, 4, false,3,3,StLpFilter_Smooth_BLUR<TPLSrcType,false,0,0,3,3> ><<<blocks, threads, sharedMem>>>
				((TPLSrcType*)d_src, (uchar4*)d_result, 
				width, BlockWidth, SharedPitch,	width, height, Scale
				,ParamsNbr,d_params);
			else if (param1 == param2 && param1 == 5)
				CudaConvKernel5<TPLSrcType, 4, false,5,5,StLpFilter_Smooth_BLUR<TPLSrcType,false,0,0,5,5> ><<<blocks, threads, sharedMem>>>
				((TPLSrcType*)d_src, (uchar4*)d_result, 
				width, BlockWidth, SharedPitch,	width, height, Scale
				,ParamsNbr,d_params);
			break;


		case CV_GAUSSIAN:
			break;
		case CV_MEDIAN:
			break;
		case CV_BILATERAL:
			break;
		}
	}
#endif
	CUT_CHECK_ERROR("Kernel execution failed");
	//clean output
	gcuPostProcess(dst);
	//clean source
	gcuPostProcess(src);
}

_GPUCV_CVGCU_EXPORT_CU
void CudaSmouthIplTPL(CvArr* src,CvArr* dst, int smoothtype/*=CV_GAUSSIAN*/,
					  int param1/*=3*/, int param2/*=0*/, double param3/*=0*/, double param4/*=0*/)
{
	CudaSmouthIpl<unsigned char, unsigned char>(src, dst, smoothtype,param1, param2, param3, param4);
}

#endif//_GPUCV_COMPILE_CUDA
