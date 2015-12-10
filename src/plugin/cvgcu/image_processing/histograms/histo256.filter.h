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
* Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.  Users and possessors of this source code
* are hereby granted a nonexclusive, royalty-free license to use this code
* in individual and commercial software.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.   This source code is a "commercial item" as
* that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer  software"  and "commercial computer software
* documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*
* Any use of this source code in individual and commercial software must
* include, in the user documentation and internal comments to the code,
* the above Disclaimer and U.S. Government End Users Notice.
*/
#ifndef __GPUCV_CUDA_CV_CU_HISTO256_KERNEL_H
#define __GPUCV_CUDA_CV_CU_HISTO256_KERNEL_H

#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>
#include <GPUCVCuda/gcu_runtime_api_wrapper.h>

//Total number of possible data values
#define     HISTO_256_BIN_COUNT		256
#define		HISTOGRAM_SIZE			(HISTO_256_BIN_COUNT * sizeof(unsigned int))

//Machine warp size
#ifndef __DEVICE_EMULATION__
//G80's warp size is 32 threads
#define WARP_LOG_SIZE 5
#else
//Emulation currently doesn't execute threads in coherent groups of 32 threads,
//which effectively means warp size of 1 thread for emulation modes
#define WARP_LOG_SIZE 0
#endif


//Warps in thread block
#define  WARP_N 6
//Threads per block count
#define     HISTO_256_THREAD_N (WARP_N << WARP_LOG_SIZE)
//Per-block number of elements in histograms
#define		HISTO_256_BLOCK_MEMORY (WARP_N * HISTO_256_BIN_COUNT)


#define IMUL(a, b) __mul24(a, b)


__GCU_FCT_DEVICE void addData256(volatile unsigned int *s_WarpHist, unsigned int data, unsigned int threadTag){
	unsigned int count;

	do{
		count = s_WarpHist[data] & 0x07FFFFFFU;
		count = threadTag | (count + 1);
		s_WarpHist[data] = count;
	}while(s_WarpHist[data] != count);
}

__GCU_FCT_GLOBAL void histogram256Kernel(unsigned int *d_Result, unsigned int *d_Data, int dataN){

	GCUDA_KRNL_DBG_FIRST_THREAD("histogram256Kernel",
		printf("- dataN:\t%d\n",	dataN);
	)


		//Current global thread index
		const int    globalTid = IMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	//Total number of threads in the compute grid
	const int   numThreads = IMUL(blockDim.x, gridDim.x);
	//Thread tag for addPixel()
#ifndef __DEVICE_EMULATION__
	//WARP_LOG_SIZE higher bits of counter values are tagged
	//by lower WARP_LOG_SIZE threadID bits
	const unsigned int threadTag = threadIdx.x << (32 - WARP_LOG_SIZE);
#else
	//Explicitly set to zero to avoid potential troubles
	const unsigned int threadTag = 0;
#endif
	//Shared memory cache for each warp in current thread block
	//Declare as volatile to prevent incorrect compiler optimizations in addPixel()
	volatile __shared__ unsigned int s_Hist[HISTO_256_BLOCK_MEMORY];
	//Current warp shared memory frame
	const int warpBase = IMUL(threadIdx.x >> WARP_LOG_SIZE, HISTO_256_BIN_COUNT);

	//Clear shared memory buffer for current thread block before processing
	for(int pos = threadIdx.x; pos < HISTO_256_BLOCK_MEMORY; pos += blockDim.x)
		s_Hist[pos] = 0;

	__syncthreads();
	//Cycle through the entire data set, update subhistograms for each warp
	//Since threads in warps always execute the same instruction,
	//we are safe with the addPixel trick
	GCUDA_KRNL_DBG_SHOW_THREAD_POS();
	for(int pos = globalTid; pos < dataN; pos += numThreads){
		unsigned int data4 = d_Data[pos];

		GCUDA_KRNL_DBG(
			printf("Pos:%d\tVal:%d", pos, data4);
		printf("Char data:{%d,%d,%d,%d}", (data4 >>  2) & 0x3FU,
			(data4 >>  10) & 0x3FU,
			(data4 >>  18) & 0x3FU,
			(data4 >>  26) & 0x3FU);
		printf("\n");
		)
			addData256(s_Hist + warpBase, (data4 >>  0) & 0xFFU, threadTag);
		addData256(s_Hist + warpBase, (data4 >>  8) & 0xFFU, threadTag);
		addData256(s_Hist + warpBase, (data4 >> 16) & 0xFFU, threadTag);
		addData256(s_Hist + warpBase, (data4 >> 24) & 0xFFU, threadTag);
	}

	__syncthreads();
	//Merge per-warp histograms into per-block and write to global memory
	GCUDA_KRNL_DBG_SHOW_THREAD_POS();
	for(int pos = threadIdx.x; pos < HISTO_256_BIN_COUNT; pos += blockDim.x){
		unsigned int sum = 0;

		for(int base = 0; base < HISTO_256_BLOCK_MEMORY; base += HISTO_256_BIN_COUNT)
			sum += s_Hist[base + pos] & 0x07FFFFFFU;

#if ATOMICS
		atomicAdd(d_Result + pos, sum);
#else
		d_Result[IMUL(HISTO_256_BIN_COUNT, blockIdx.x) + pos] = sum;
#endif

	}
}


//Thread block (== subhistogram) count
#define BLOCK_N 64


///////////////////////////////////////////////////////////////////////////////
// Merge BLOCK_N subhistograms of HISTO_256_BIN_COUNT bins into final histogram
////////////////////////////////////////////////////////////////////////////////
__GCU_FCT_GLOBAL void mergeHistogram256Kernel(unsigned int *d_Result){
	//gridDim.x   == HISTO_256_BIN_COUNT
	//blockDim.x  == BLOCK_N
	//blockIdx.x  == bin counter processed by current block
	//threadIdx.x == subhistogram index
	__shared__ unsigned int data[BLOCK_N];

	//Reads are uncoalesced, but this final stage takes
	//only a fraction of total processing time
	data[threadIdx.x] = d_Result[IMUL(threadIdx.x, HISTO_256_BIN_COUNT) + blockIdx.x];

	for(int stride = BLOCK_N / 2; stride > 0; stride >>= 1){
		__syncthreads();
		if(threadIdx.x < stride)
			data[threadIdx.x] += data[threadIdx.x + stride];
	}

	if(threadIdx.x == 0)
		d_Result[blockIdx.x] = data[0];
}



////////////////////////////////////////////////////////////////////////////////
// Put all kernels together
////////////////////////////////////////////////////////////////////////////////
//histogram256kernel() results buffer
unsigned int *d_ResultHisto256;

//Internal memory allocation
void initHistogram256(void){
#if ATOMICS
	GCU_CUDA_SAFE_CALL( cudaMalloc((void **)&d_ResultHisto256, HISTOGRAM_SIZE ) );
#else
	GCU_CUDA_SAFE_CALL( cudaMalloc((void **)&d_ResultHisto256, BLOCK_N * HISTOGRAM_SIZE) );
#endif
}

//Internal memory deallocation
void closeHistogram256(void){
	GCU_CUDA_SAFE_CALL( cudaFree(d_ResultHisto256) );
}

//histogram256 CPU front-end
void histogram256GPU(
					 unsigned int *d_Data,
					int *h_Result,//OpenCV store results in 'int' not 'uint'
					//int *d_Result,//OpenCV store results in 'int' not 'uint'
					int dataN
					 ){
#if 0//_GPUCV_DEBUG_MODE
						 printf("histogram256GPU()\n");
						 printf("- dataN: %d\n", dataN);
						 printf("- HISTO_256_THREAD_N: %d\n", HISTO_256_THREAD_N);
						 printf("- BLOCK_N: %d\n", BLOCK_N);
						 printf("- MAX_BLOCK_N: %d\n", MAX_BLOCK_N);
						 printf("- WARP_N: %d\n", WARP_N);
						 printf("- HISTO_256_BLOCK_MEMORY: %d\n", HISTO_256_BLOCK_MEMORY);
#endif
#if ATOMICS
						 GCU_CUDA_SAFE_CALL( cudaMemset(d_ResultHisto256, 0, HISTOGRAM_SIZE) );

						 histogram256Kernel<<<BLOCK_N, HISTO_256_THREAD_N>>>(
							 d_ResultHisto256,
							 (unsigned int *)d_Data,
							 dataN / 4
							 );
						 CUT_CHECK_ERROR("histogram256Kernel() execution failed\n");

#else
						 histogram256Kernel<<<BLOCK_N, HISTO_256_THREAD_N>>>(
							 d_ResultHisto256,
							 (unsigned int *)d_Data,
							 dataN / 4
							 );
						 CUT_CHECK_ERROR("histogram256Kernel() execution failed\n");

						 mergeHistogram256Kernel<<<HISTO_256_BIN_COUNT, BLOCK_N>>>(d_ResultHisto256);
						 CUT_CHECK_ERROR("mergeHistogram256Kernel() execution failed\n");
#endif

						 GCU_CUDA_SAFE_CALL( cudaMemcpy(h_Result, d_ResultHisto256, HISTOGRAM_SIZE, cudaMemcpyDeviceToHost) );
}

#endif //_GPUCV_CUDA_CV_CU_HISTO256_KERNEL_H
