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
//	- CudaCalcHistIpl() based on histogram64 and histogram256.
//==============================================================================
#include <cvgcu/config.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>

#if _GPUCV_COMPILE_CUDA

#define CUDA_HISTO256_SUPPORT 1

#include <cvgcu/image_processing/histograms/histo64.filter.h>
#if CUDA_HISTO256_SUPPORT 
#include <cvgcu/image_processing/histograms/histo256.filter.h>
#endif


///////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int NUM_ITERATIONS = 1;

_GPUCV_CVGCU_EXPORT_CU
void gcuCalcHist(CvArr ** _src, 
					 CvArr * histImage,//store the data from the histogram
					 int accumulate/*=0*/, 
					 const CvArr* mask/*=NULL*/)
{//Check inputs is done in the cv_cu.cpp file, to manage exceptions
	//prepare settings
	CvArr * mainSrc = *_src;
	unsigned int width 		= gcuGetWidth(mainSrc);
	unsigned int height 	= gcuGetHeight(mainSrc);
	unsigned int channels 	= gcuGetnChannels(mainSrc);
	unsigned int DataN 		= width * height * channels;
//	unsigned int DataSize 	= DataN * sizeof(unsigned char);
	int i;
	int iBins = gcuGetWidth(histImage);
	
	//=====================
	unsigned int * d_src 	= (unsigned int*)gcuPreProcess(mainSrc, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	int	* h_bins 	= (int*)gcuSyncToCPU(histImage, false);
	//memset(h_bins, sizeof(int)*iBins, 0);
	//int * d_bins 	= (int*)gcuPreProcess(histImage, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
	//=====================

	//init Histogram
#if CUDA_HISTO256_SUPPORT 
	if(iBins == 256)
		initHistogram256();
	else
#endif
		initHistogram64();

#if CUDA_HISTO256_SUPPORT
	if(iBins == 256)
	{
		for(i = 0; i < NUM_ITERATIONS; i++)
			histogram256GPU(d_src,h_bins,DataN);
		closeHistogram256();
	}
	else
#endif
	{
		for(i = 0; i < NUM_ITERATIONS; i++)
			histogram64GPU(d_src,h_bins,DataN);
		closeHistogram64();			
	}

#if 0//def _DEBUG
	//check results 
			int sum = 0;
			for(i = 0; i < iBins; i++)
			{
				sum   += h_bins[i];
				printf("bin %d, val %d, Sum %d\n", i, h_bins[i], sum);
			}
			printf("Total sum of histogram elements: %i\n", sum);
#endif

	//=====================
	//clean source
	gcuPostProcess(mainSrc);
	//gcuPostProcess(histImage);
	//=====================
}
#endif//_GPUCV_COMPILE_CUDA
