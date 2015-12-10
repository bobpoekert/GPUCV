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
#ifndef __GPUCV_CUDA_CXCOREGCU_FFT_CU
#define __GPUCV_CUDA_CXCOREGCU_FFT_CU

#include <cxcoregcu/config.h>  
#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <cufft.h>
#include <cutil.h>

#define GCU_Complex float2
	/** \	DFT - Discrete time fourier transform implemented using FFT algorithm and cudaFFT library,
	gcuDFT - kernel for DFT operator.
	\param flags -> not used yet, could be a combination of: CV_DXT_FORWARD/CV_DXT_INVERSE/CV_DXT_SCALE/CV_DXT_ROWS
	\param nr -> not used yet

	*/
_GPUCV_CXCOREGCU_EXPORT_CU
	void gcuDFT(CvArr* src1,CvArr* dst,int flags,int nr)
	{		
		int NY = gcuGetHeight(src1);
		int NX = gcuGetWidth(src1);
		int channelSrc = gcuGetnChannels(src1);
		int channelDst = gcuGetnChannels(dst);

		GCU_Complex* h_signal = (GCU_Complex*)malloc(sizeof(GCU_Complex) * NX);
		cufftHandle plan;
		//int mem_size = sizeof(cufftGCU_Complex) * NX;

		// Allocate device memory for signal
		GCU_Complex* di_signal = (GCU_Complex*)gcuPreProcess(src1, GCU_INPUT, CU_MEMORYTYPE_DEVICE,NULL);
		GCU_Complex* do_signal = (GCU_Complex*)gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE,NULL);


		//get conversion type REAL/COMPLEX to REAL/COMPLEX
		cufftType ConversionType=CUFFT_R2C;
		if(channelSrc==1)
			if(channelDst==1)
			{
				//ConversionType = CUFFT_R2R;
				//not possible
			}
			else
				ConversionType = CUFFT_R2C;
		if(channelSrc==2)
			if(channelDst==1)
				ConversionType = CUFFT_C2R;
			else
				ConversionType = CUFFT_C2C;

		ConversionType = CUFFT_C2C;
			

		if (NY==1)
		{	/* Create a 1D FFT plan. */
			CUFFT_SAFE_CALL(cufftPlan1d(&plan, NX, ConversionType, 1));
		}
		else
		{	/* Create a 2D FFT plan. */
			CUFFT_SAFE_CALL(cufftPlan2d(&plan, NX , NY, ConversionType));
		}

		switch(ConversionType)
		{
			case CUFFT_R2C:CUFFT_SAFE_CALL(cufftExecR2C(plan, (cufftReal *)di_signal,(cufftComplex *)do_signal));break;
			case CUFFT_C2R:CUFFT_SAFE_CALL(cufftExecC2R(plan, (cufftComplex *)di_signal,(cufftReal *)do_signal));break;
			case CUFFT_C2C:CUFFT_SAFE_CALL(cufftExecC2C(plan, (cufftComplex *)di_signal,(cufftComplex *)do_signal,CUFFT_FORWARD));break; 
		}		

		CUFFT_SAFE_CALL(cufftDestroy(plan));
		gcuPostProcess(dst);
		gcuPostProcess(src1);
	}

#endif// __GPUCV_CUDA_CXCOREGCU_FFT_CU	
