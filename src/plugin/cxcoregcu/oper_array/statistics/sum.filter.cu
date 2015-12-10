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
/** 	\brief Contains GpuCV-CUDA correspondance of cxcore->array.
*	\author Yannick Allusse
*/
#include <cxcoregcu/config.h>
#if _GPUCV_COMPILE_CUDA

#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <cxcoregcu/cxcoregcu_array_arithm.kernel.h>
#include <cxcoregcu/cxcoregcu_statistics.kernel.h>

void ReduceGetNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
	if (whichKernel < 3)
	{
		threads = (n < maxThreads) ? n : maxThreads;
		blocks = n / threads;
	}
	else
	{
		if (n == 1) 
			threads = 1;
		else
			threads = (n < maxThreads*2) ? n / 2 : maxThreads;
		blocks = n / (threads * 2);

		if (whichKernel == 6)
			blocks = min(maxBlocks, blocks);
	}
}
//=============================================================
#define PROF_REDUCE(FCT)

/**
*\bug Don't know why yet, but the value of the ponter is changing when arriving into the function...???
*/
_GPUCV_CXCOREGCU_EXPORT_CU 
void gcuSumArr(CvArr* _arr)
{
#if 0
	//gcudaDeviceInit();
	//prepare settings
	unsigned int width		= gcuGetWidth(_arr);
	unsigned int height		= gcuGetHeight(_arr);
	unsigned int depth		= gcuGetGLDepth(_arr);
	unsigned int channels	= gcuGetnChannels(_arr);
	//const unsigned int DataSize = DataN * sizeof(unsigned char);
	int maxThreads = 128;  // number of threads per block
	int whichKernel = 6;
	int maxBlocks = 64;


	//Check inputs is done in the cv_cu.cpp file, to manage exceptions
	PROF_REDUCE(CUT_SAFE_CALL(cutCreateTimer(&hTimer)););	

	//=====================
	//prepare source
	unsigned char * d_src = (unsigned char *)gcuPreProcess(_arr, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	//=====================

	int numBlocks = 0;
	int numThreads = 0;
	int maxNumBlocks = DataN / maxThreads;
	ReduceGetNumBlocksAndThreads(whichKernel, DataN, maxBlocks, maxThreads, numBlocks, numThreads);
	//if (numBlocks == 1) cpuFinalThreshold = 1;

	//prepare ouput========
	int* d_odata = NULL;
	int* h_odata = (int*) malloc(maxNumBlocks*sizeof(int));
	GCU_CUDA_SAFE_CALL( cudaMalloc((void**) &d_odata, numBlocks*sizeof(int)) );
	//=====================
	
	gcudaThreadSynchronize();

	dim3 dimBlock(numThreads, 1, 1);
	dim3 dimGrid(numBlocks, 1, 1);
	int smemSize = numThreads * sizeof(int);
	switch (numThreads)
	{
	case 512:
		CudaReduceSum<512><<< dimGrid, dimBlock, smemSize >>>(d_src, d_odata, DataN); break;
	case 256:
		CudaReduceSum<256><<< dimGrid, dimBlock, smemSize >>>(d_src, d_odata, DataN); break;
	case 128:
		CudaReduceSum<128><<< dimGrid, dimBlock, smemSize >>>(d_src, d_odata, DataN); break;
	case 64:
		CudaReduceSum< 64><<< dimGrid, dimBlock, smemSize >>>(d_src, d_odata, DataN); break;
	case 32:
		CudaReduceSum< 32><<< dimGrid, dimBlock, smemSize >>>(d_src, d_odata, DataN); break;
	case 16:
		CudaReduceSum< 16><<< dimGrid, dimBlock, smemSize >>>(d_src, d_odata, DataN); break;
	case  8:
		CudaReduceSum<  8><<< dimGrid, dimBlock, smemSize >>>(d_src, d_odata, DataN); break;
	case  4:
		CudaReduceSum<  4><<< dimGrid, dimBlock, smemSize >>>(d_src, d_odata, DataN); break;
	case  2:
		CudaReduceSum<  2><<< dimGrid, dimBlock, smemSize >>>(d_src, d_odata, DataN); break;
	case  1:
		CudaReduceSum<  1><<< dimGrid, dimBlock, smemSize >>>(d_src, d_odata, DataN); break;
	}
	// sum partial sums from each block on CPU        
	// copy result from device to host
	GCU_CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, numBlocks*sizeof(int), cudaMemcpyDeviceToHost) );

	gcuScalar gpu_result;
	for(int i=0; i<numBlocks; i++) 
	{
		gpu_result.val[0] += h_odata[i];
	}

	//GCU_CUDA_SAFE_CALL (cudaUnBindTexture(texSobel));

	//check results

	//clean output
	GCU_CUDA_SAFE_CALL( cudaFree(d_odata));
	//=====================

	//=====================
	//clean source
	gcuPostProcess(_arr);	

	//close operator
	PROF_REDUCE(CUT_SAFE_CALL(cutDeleteTimer(hTimer)););
#endif
}
//==========================================================
#endif//CUDA
