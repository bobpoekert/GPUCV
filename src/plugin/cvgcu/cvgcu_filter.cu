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
//Other Licenses:
//Some operators are inspired from the CUDA SDK, see corresponding license terms.
//	CUDA SDK:
//	- CudaSobelIpl() based on Sobel.
//==============================================================================
#include "config.h"
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <GPUCVCuda/base_kernels/tpl_convolutions.kernels.h>


#if 0//_GPUCV_COMPILE_CUDA

#define PROF_FCT(FCT)//FCT


template <bool CustumConvKernel, typename TPLSrcType,  typename TPLDstType, int TPLApertureSize>
void CudaConvolutionIplTPLSwithOper(int OPER_TYPE, TPLSrcType* _src,TPLDstType* _dst, GCU_CONV_Kernel* element,int iterations, GCU_CONVOL_KERNEL_SETTINGS_ST & OpSettings)
{
#if 1
#if 0
	CudaConvKernel<TPLSrcType, 4,	CustumConvKernel,TPLApertureSize,TPLApertureSize, StLpFilter_Erode<TPLSrcType,CustumConvKernel,0,0,TPLApertureSize,TPLApertureSize>,TPLApertureSize > <<<OpSettings.KSize.Blocks, OpSettings.KSize.Threads, OpSettings.KSize.SharedMem>>>
		((TPLSrcType*)_src, (uchar4*)_dst, 
		OpSettings.DSize.Size.x, OpSettings.KSize.BlockSize.x, OpSettings.KSize.SharedPitch,	
		OpSettings.DSize.Size.x, OpSettings.DSize.Size.y, 1.,
		OpSettings.ParamsNbr,OpSettings.Params);
#else
	switch(OPER_TYPE)
	{
	case CONVOLUTION_KERNEL_ERODE://FCT_PTR_COMP_DILATE_TPL(ComputeDilate,unsigned char,CustumConvKernel,unsigned char)
		CudaConvKernel3<TPLSrcType, 4,	CustumConvKernel,TPLApertureSize,TPLApertureSize, /*StLpFilter_Erode*/StLpFilter_Dilate<TPLSrcType,CustumConvKernel,0,0,TPLApertureSize,TPLApertureSize> > <<<OpSettings.KSize.Blocks, OpSettings.KSize.Threads, OpSettings.KSize.SharedMem>>>
			((TPLSrcType*)_src, (uchar4*)_dst, 
			OpSettings.DSize.Size.x, OpSettings.KSize.BlockSize.x, OpSettings.KSize.SharedPitch,	OpSettings.DSize.Size.x, OpSettings.DSize.Size.y, 1.
			,OpSettings.ParamsNbr,OpSettings.Params);
		break;
		//case CONVOLUTION_KERNEL_ERODE_NOSHARE://FCT_PTR_COMP_DILATE_TPL(ComputeDilate,unsigned char,CustumConvKernel,unsigned char)
		//		ErodeTexKernel<<<blocks, threads, sharedMem>>>
		//											((uchar*)d_tempDest, 
		//											SharedPitch,	width, height, 1.);
		//	break;
	case CONVOLUTION_KERNEL_DILATE:
		CudaConvKernel3<TPLSrcType, 4,	CustumConvKernel,TPLApertureSize,TPLApertureSize,StLpFilter_Dilate<TPLSrcType,CustumConvKernel,0,0,TPLApertureSize,TPLApertureSize> > <<<OpSettings.KSize.Blocks, OpSettings.KSize.Threads, OpSettings.KSize.SharedMem>>>
			((TPLSrcType*)_src, (uchar4*)_dst, 
			OpSettings.DSize.Size.x, OpSettings.KSize.BlockSize.x, OpSettings.KSize.SharedPitch,	OpSettings.DSize.Size.x, OpSettings.DSize.Size.y, 1.
			,OpSettings.ParamsNbr,OpSettings.Params);
		break;
#if 1
	case CONVOLUTION_KERNEL_LAPLACE:
		CudaConvKernel3<TPLSrcType, 4, CustumConvKernel,TPLApertureSize,TPLApertureSize,StLpFilter_Laplace<TPLSrcType,CustumConvKernel,0,0,TPLApertureSize,TPLApertureSize> > <<<OpSettings.KSize.Blocks, OpSettings.KSize.Threads, OpSettings.KSize.SharedMem>>>
			((TPLSrcType*)_src, (uchar4*)_dst, 
			OpSettings.DSize.Size.x, OpSettings.KSize.BlockSize.x, OpSettings.KSize.SharedPitch,	OpSettings.DSize.Size.x, OpSettings.DSize.Size.y, 1.
			,OpSettings.ParamsNbr,OpSettings.Params);

		break;
	case CONVOLUTION_KERNEL_SOBEL:
		CudaConvKernel3<TPLSrcType, 4, CustumConvKernel,TPLApertureSize,TPLApertureSize,StLpFilter_Sobel<TPLSrcType,CustumConvKernel,0,0,TPLApertureSize,TPLApertureSize> > <<<OpSettings.KSize.Blocks, OpSettings.KSize.Threads, OpSettings.KSize.SharedMem>>>
			((TPLSrcType*)_src, (uchar4*)_dst, 
			OpSettings.DSize.Size.x, OpSettings.KSize.BlockSize.x, OpSettings.KSize.SharedPitch,	OpSettings.DSize.Size.x, OpSettings.DSize.Size.y, 1.
			,OpSettings.ParamsNbr,OpSettings.Params);
		break;
#endif
	}
#endif
#endif
}







template <bool CustumConvKernel, typename TPLSrcType,  typename TPLDstType>
void CudaConvolutionIplTPL(int OPER_TYPE, CvArr* src,CvArr* dst, GCU_CONV_Kernel* element,int iterations,int aperture_size=1)
{

	//prepare settings	
	GCU_CONVOL_KERNEL_SETTINGS_ST OpSettings;
	OpSettings.DSize.Size.x = gcuGetWidth(src);
	OpSettings.DSize.Size.y = gcuGetHeight(src);
	OpSettings.DSize.Channels = gcuGetnChannels(src);;
	const unsigned int DataN = OpSettings.DSize.Size.x * OpSettings.DSize.Size.y * OpSettings.DSize.Channels;
	const unsigned int DataSize = DataN * sizeof(unsigned char);

	//Check inputs is done in the cv_cu.cpp file, to manage exceptions

	//=====================
	//prepare source
	//Src1
	TPLSrcType * d_src = (TPLSrcType *)gcuPreProcess(src, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	//=====================

	//prepare ouput========
	TPLDstType * d_result = (TPLDstType *)gcuPreProcess(dst, GCU_OUTPUT,CU_MEMORYTYPE_DEVICE);
	//temp image:
	TPLDstType * d_Temp = NULL;
	if(iterations!=1)
	{
		size_t Pitch;
		gcudaMallocPitch((void **)&d_Temp, &Pitch, OpSettings.DSize.Size.x * OpSettings.DSize.Channels*sizeof(TPLDstType), OpSettings.DSize.Size.y);
		gcudaMemset(d_Temp, 0, DataSize);
		gcudaCheckError("DataDsc_CUDA::_DeviceAllocate() execution failed\n");
	}
	//=====================


	//prepare parameters
	unsigned char * h_params = NULL;
	void * d_params = NULL;
	int ParamsNbr = 0;
	if(element && CustumConvKernel)
	{
		ParamsNbr = element->nCols * element->nRows;
		h_params = (unsigned char*)malloc(ParamsNbr * sizeof(unsigned char));
		for(int i=0;i< ParamsNbr;i++)
		{
			h_params[i] = (element->values[i])?1:0;
		}
		gcudaMalloc((void**)&d_params, ParamsNbr * sizeof(unsigned char));
		gcudaMemCopyHostToDevice(d_params, h_params, ParamsNbr * sizeof(unsigned char));
		OpSettings.ParamsNbr=ParamsNbr;
		OpSettings.Params=(unsigned char*)d_params;
	}
	//else all element are ==1, so we don't need to apply specific filter
	//printf("\nElemParam size:%d\n", ParamsNbr * sizeof(unsigned char));
	//=================

	
	gcudaThreadSynchronize();

	OpSettings.KSize.Threads.x = 16;
	OpSettings.KSize.Threads.y = 4;

	OpSettings.KSize.BlockSize.x = 80; // must be divisible by 16 for coalescing
	OpSettings.KSize.BlockSize.y = 1;//updated later
	//BlockWidth must be at least (4*)*2 time inferior to width...
	/*while(width<(BlockWidth)*4*2)
	{
	if(BlockWidth>16)
	{
	BlockWidth-=16;
	}
	else
	{
	BlockWidth /=2;
	}
	}

	//width should be multiple of BlockWidth??
	if (!IS_MULTIPLE_OF(width,BlockWidth))
	{
	BlockWidth = (int)width/((int)width/BlockWidth);
	printf("new BlockWidth = %d\n",BlockWidth);
	}*/	



	// (256,256,1);

	OpSettings.KSize.Blocks.x = OpSettings.DSize.Size.x/(4*OpSettings.KSize.BlockSize.x)
		+(0!=OpSettings.DSize.Size.x%(4*OpSettings.KSize.BlockSize.x));

	OpSettings.KSize.Blocks.y = OpSettings.DSize.Size.y/OpSettings.KSize.Threads.y
		+(0!=OpSettings.DSize.Size.y%OpSettings.KSize.Threads.y);

	//dim3 blocks = dim3(
	//width/(threads.x*BlockWidth/4)+(0!=width%(threads.x/4*BlockWidth)),
	//height/threads.y+(0!=height%threads.y));

#if 0
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
	blocks =  dim3(	iDivUp(width,(4*BlockWidth)),
		iDivUp(height,threads.y),
		1);
#endif

	//OpSettings.KSize.BlockSize.y = OpSettings.KSize.Threads.y;

	int LocalRadius = (aperture_size)?(aperture_size-1)/2:1;
	OpSettings.KSize.SharedPitch = ~0x3f&(4*(OpSettings.KSize.BlockSize.x+2*LocalRadius)+0x3f);
	OpSettings.KSize.SharedMem = OpSettings.KSize.SharedPitch *(OpSettings.KSize.Threads.y+2*LocalRadius);
	// for the shared kernel, width must be divisible by 4
	OpSettings.DSize.Size.x &= ~3;
#if 1//_DEBUG
	printf("DilateSharedKernel===\n");
	printf("- width: %d\n", OpSettings.DSize.Size.x);
	printf("- height: %d\n", OpSettings.DSize.Size.y);
	printf("- channels: %d\n", OpSettings.DSize.Channels);
	printf("- DataN: %d\n", DataN);
	printf("- DataSize: %d\n", DataSize);
	printf("- threads: %d %d %d\n", OpSettings.KSize.Threads.x, OpSettings.KSize.Threads.y, OpSettings.KSize.Threads.z);
	printf("- blocks: %d %d %d\n", OpSettings.KSize.Blocks.x, OpSettings.KSize.Blocks.y, OpSettings.KSize.Blocks.z);
	printf("- Blockdim: %d %d\n",	OpSettings.KSize.BlockSize.x, OpSettings.KSize.BlockSize.y);		
	printf("- SharedPitch: %d\n",	OpSettings.KSize.SharedPitch);
	printf("- sharedMem: %d\n",		OpSettings.KSize.SharedMem);
#endif

	void * d_tempSrc	= NULL;
	void * d_tempDest	= NULL;

	if(iterations==1)
	{
		d_tempSrc	= d_src;
		d_tempDest	= d_result;
	}
	else
	{
		d_tempSrc	= d_src;
		d_tempDest	= IS_MULTIPLE_OF(iterations,2)? d_Temp:d_result;
	}
	//printf("Begin:\n tempSrc:%d\ntempDst:%d\n",d_tempSrc, d_tempDest);
#if 1 //simule filter
	for(int i = 0; i < iterations; i++)
	{
		//	printf("iterations:%d/%d\n", i, iterations);
		//	printf("Processing tempSrc:%d to tempDst:%d\n",d_tempSrc, d_tempDest);
		//process operator
		if(OpSettings.DSize.Channels==1)				
		{//
			switch(aperture_size)
			{
			case 3:
				CudaConvolutionIplTPLSwithOper<CustumConvKernel, TPLSrcType, TPLDstType, 3>(OPER_TYPE, (TPLSrcType*)d_tempSrc,(TPLDstType*)d_tempDest, element,iterations,OpSettings);
				break;
			case 5:
				CudaConvolutionIplTPLSwithOper<CustumConvKernel, TPLSrcType, TPLDstType, 5>(OPER_TYPE, (TPLSrcType*)d_tempSrc,(TPLDstType*)d_tempDest, element,iterations,OpSettings);
				break;
#if 0 
			case 7:
				CudaConvolutionIplTPLSwithOper<CustumConvKernel, TPLSrcType, TPLDstType, 7>(OPER_TYPE, (TPLSrcType*)d_tempSrc,(TPLDstType*)d_tempDest, element,iterations,OpSettings);
				break;
#endif
			}
		}

		if(iterations>1)
		{
			if(i == 0)//we preserve to write into the source object
			{
				//		printf("Changing tempsrc :%d to %d\n",d_tempSrc, IS_MULTIPLE_OF(iterations,2)? d_result:d_Temp);
				d_tempSrc = (IS_MULTIPLE_OF(iterations,2)? d_result:d_Temp);
			}
			//printf("tempSrc:%d\ntempDst:%d\n",d_tempSrc, d_tempDest);
			//SWITCH_VAL(TPLSrcType*, (TPLSrcType*)d_tempSrc, (TPLSrcType*)d_tempDest);
			void* SwitchTemp = d_tempSrc;
			d_tempSrc = d_tempDest;
			d_tempDest =  SwitchTemp;
		}			
	}
#endif
	//if(d_tempDest!=d_src && d_tempDest!=d_result)
	gcudaFree(d_Temp);

	//update output
	gcuPostProcess(dst);
	//update source
	gcuPostProcess(src);
	//free variables
	free(h_params);
	gcudaFree(d_params);
	//=======================
}



template <bool CustumConvKernel, typename TPLSrcType>
inline
void CudaConvolutionIplTPL_Dest(int _ConvolType, CvArr* src,CvArr* dst, GCU_CONV_Kernel* element,int iterations,int aperture_size=0)
{
	switch(gcuGetGLDepth(src))
	{
		//case IPL_DEPTH_1U:

		//	case IPL_DEPTH_8U:	CudaConvolutionIplTPL<CustumConvKernel,TPLSrcType,unsigned char>(_ConvolType,src, dst, NULL, iterations, aperture_size);break;
		//	case GL_BYTE:	CudaConvolutionIplTPL<CustumConvKernel,TPLSrcType,char>(_ConvolType,src, dst, NULL, iterations, aperture_size);break;
		//	case GL_UNSIGNED_SHORT: CudaConvolutionIplTPL<CustumConvKernel,TPLSrcType,unsigned int>(_ConvolType,src, dst, NULL, iterations, aperture_size);break;
		//	case GL_SHORT: CudaConvolutionIplTPL<CustumConvKernel,TPLSrcType,int>(_ConvolType,src, dst, NULL, iterations, aperture_size);break;
		//	case IPL_DEPTH_32F: 
		//	case IPL_DEPTH_32S:
		//						CudaConvolutionIplTPL<CustumConvKernel,TPLSrcType,float>(_ConvolType,src, dst, NULL, iterations, aperture_size);break;
	default:
		CudaConvolutionIplTPL<CustumConvKernel,TPLSrcType,unsigned char>(_ConvolType,src, dst, NULL, iterations, aperture_size);break;
	}
}

template <bool CustumConvKernel>//, typename TPLSrcType,  typename TPLDstType>
inline
void CudaConvolutionIplTPL_Source(int _ConvolType, CvArr* src,CvArr* dst, GCU_CONV_Kernel* element,int iterations,int aperture_size=0)
{
	switch(gcuGetGLDepth(src))
	{
		//case IPL_DEPTH_1U:

		//case IPL_DEPTH_8U:	CudaConvolutionIplTPL_Dest<CustumConvKernel,unsigned char>(_ConvolType,src, dst, NULL, iterations, aperture_size);break;
		//case GL_BYTE:	CudaConvolutionIplTPL_Dest<CustumConvKernel,char>(_ConvolType,src, dst, NULL, iterations, aperture_size);break;
		//	case GL_UNSIGNED_SHORT: CudaConvolutionIplTPL_Dest<CustumConvKernel,unsigned int>(_ConvolType,src, dst, NULL, iterations, aperture_size);break;
		//	case GL_SHORT: CudaConvolutionIplTPL_Dest<CustumConvKernel,int>(_ConvolType,src, dst, NULL, iterations, aperture_size);break;
		//	case IPL_DEPTH_32F: 
		//	case IPL_DEPTH_32S:
		//						CudaConvolutionIplTPL_Dest<CustumConvKernel,float>(_ConvolType,src, dst, NULL, iterations, aperture_size);break;
	default:
		CudaConvolutionIplTPL_Dest<CustumConvKernel,unsigned char>(_ConvolType,src, dst, NULL, iterations, aperture_size);break;
	}
}



_GPUCV_CVGCU_EXPORT_CU
void gcuConvolutionTPL_OptimizedKernel(int _ConvolType, CvArr* src,CvArr* dst, GCU_CONV_Kernel* element,int iterations,int aperture_size=0)
{
	//check is all kernel

	int ParamsNbr = 0;
	int TotalMul = 1;
	bool UseOptimizedDefaultKernel = true;
	int LocalApertureSize = aperture_size;

	if(element && !aperture_size)
		LocalApertureSize = element->nCols;

	switch(_ConvolType)
	{
	case CONVOLUTION_KERNEL_ERODE:
	case CONVOLUTION_KERNEL_DILATE:
		//val is used as iteration number
		{//check is all kernel element are 1 
			// if true we can use optimized cuda kernel
			if(element)
			{
				ParamsNbr = element->nCols * element->nRows;
				for(int i=0;i< ParamsNbr;i++)
				{
					TotalMul *= element->values[i];
				}
				if(TotalMul==1)
					UseOptimizedDefaultKernel=true;
				else
					UseOptimizedDefaultKernel=false;
			}	
			break;
		}
	case CONVOLUTION_KERNEL_LAPLACE:
		{//val is used as aperture_size
			if(aperture_size==1)
			{	/*laplace filter is 
				|0  1  0|
				|1 -4  1|
				|0  1  0|
				Wich is default optimized filter
				*/
				//CudaConvolutionIplTPL<false>(_ConvolType,src, dst, NULL,1);
				UseOptimizedDefaultKernel=true;
			}
			else
			{
				//...generate kernel Elements...
				UseOptimizedDefaultKernel=false;
			}
			break;
		}
#if 0//Done externaly
	case CONVOLUTION_KERNEL_SOBEL:
		{
			if(aperture_size==CV_SCHARR)
			{//structuring element is:
				/*	| -3 0  3|
				|-10 0 10|
				| -3 0  3|
				*/
				if(!element)
				{
					const int element_shape = CV_SHAPE_RECT;
					float pos = (3 -1)/2;
					element = cvCreateStructuringElementEx( pos*2+1, pos*2+1, pos, pos, element_shape, 0 );
					element->values[0] = element->values[6]= -3;
					element->values[1] = element->values[4] = element->values[7] = 0;
					element->values[2] = element->values[8] = 3;
					element->values[3] = element->values[5] = -10;
				}
				UseOptimizedDefaultKernel=false;
			}
			else
				UseOptimizedDefaultKernel=true;
			break;
		}
#endif
	}

	if(UseOptimizedDefaultKernel)
		CudaConvolutionIplTPL_Source<false>(_ConvolType,src, dst, NULL, iterations, LocalApertureSize);
	else
		CudaConvolutionIplTPL_Source<true>(_ConvolType,src, dst, element,iterations, LocalApertureSize);
}

#endif//_GPUCV_COMPILE_CUDA
