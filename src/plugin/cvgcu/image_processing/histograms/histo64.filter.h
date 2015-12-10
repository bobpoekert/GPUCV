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

// Set atomics to 1 and add '-arch sm_11' to the command line to enable atomic operations.
/*! Atomic operation can be used for cuda processor from compute capability version 1.1 and higher(8800 GTS is 1.0)
*	\sa http://forums.nvidia.com/index.php?showtopic=36286  What GPUs does CUDA run on?
*/
#ifndef __GPUCV_CUDA_CV_CU_HISTO64_KERNEL_H
#define __GPUCV_CUDA_CV_CU_HISTO64_KERNEL_H
#define ATOMICS 0
#if ATOMICS
#include <sm_11_atomic_functions.h>
#endif
/*
#include "device_types.h"
#include "host_defines.h"
#include "builtin_types.h"
*/
#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>
#include <GPUCVCuda/gcu_runtime_api_wrapper.h>

//Total number of possible data values
#define     HISTO_64_BIN_COUNT	64
#define		HISTOGRAM_SIZE		(HISTO_64_BIN_COUNT * sizeof(unsigned int))

//Threads per block count, preferred to be a multiple of 64 (see below)
#define     THREAD_N 192
//Per-block number of elements in all per-thread histograms
#define BLOCK_MEMORY (THREAD_N * HISTO_64_BIN_COUNT)
//Dwords processed by single block count
//Since counters are one-byte, in order to avoid overflow(wrap around)
//this size must be limited by the number
#define   BLOCK_DATA (THREAD_N * 63)

#define IMUL(a, b) __mul24(a, b)

////////////////////////////////////////////////////////////////////////////////
// If threadPos == threadIdx.x, there are always  4-way bank conflicts,
// since each group of 16 threads (half-warp) accesses different bytes,
// but only within 4 shared memory banks. Having shuffled bits of threadIdx.x
// as in histogram64GPU(), each half-warp accesses different shared memory banks
// avoiding any bank conflicts at all.
// See supplied whitepaper for detailed explanations.
////////////////////////////////////////////////////////////////////////////////
__GCU_FCT_DEVICE void addData64(unsigned char *s_Hist, int threadPos, unsigned int data){
	s_Hist[threadPos + IMUL(data, THREAD_N)]++;
}

////////////////////////////////////////////////////////////////////////////////
// Main computation pass: compute gridDim.x partial histograms
////////////////////////////////////////////////////////////////////////////////
__GCU_FCT_GLOBAL void histogram64Kernel(unsigned int *d_Result, unsigned int *d_Data, int dataN){

	GCUDA_KRNL_DBG_FIRST_THREAD("histogram64Kernel",
		printf("- dataN:\t%d\n",	dataN);
	)


		//Global base index in input data for current block
		const int  baseIndex = IMUL(BLOCK_DATA, blockIdx.x);
	//Current block size, clamp by array border
	const int   dataSize = min(dataN - baseIndex, BLOCK_DATA);
	//Encode thread index in order to avoid bank conflicts in s_Hist[] access:
	//each half-warp accesses consecutive shared memory banks
	//and the same bytes within the banks
	const int threadPos =
		//[31 : 6] <== [31 : 6]
		((threadIdx.x & (~63)) >> 0) |
		//[5  : 2] <== [3  : 0]
		((threadIdx.x &    15) << 2) |
		//[1  : 0] <== [5  : 4]
		((threadIdx.x &    48) >> 4);

	//Per-thread histogram storage
	__shared__ unsigned char s_Hist[BLOCK_MEMORY];

	//Don't forget to clear shared memory before utilization
	//No need in per-byte access at this stage
	for(int pos = threadIdx.x; pos < (BLOCK_MEMORY / 4); pos += blockDim.x)
		((unsigned int *)s_Hist)[pos] = 0;

	__syncthreads();
	////////////////////////////////////////////////////////////////////////////
	// Cycle through current block, update per-thread histograms
	// Since only 64-bit histogram of 8-bit input data array is calculated,
	// only highest 6 bits of each 8-bit data element are extracted,
	// leaving out 2 lower bits.
	////////////////////////////////////////////////////////////////////////////
	GCUDA_KRNL_DBG_SHOW_THREAD_POS();
	for(int pos = threadIdx.x; pos < dataSize; pos += blockDim.x){
		unsigned int data4 = d_Data[baseIndex + pos];

		GCUDA_KRNL_DBG(
			printf("Pos:%d\tVal:%d", pos, data4);
		printf("Char data:{%d,%d,%d,%d}", (data4 >>  2) & 0x3FU,
			(data4 >>  10) & 0x3FU,
			(data4 >>  18) & 0x3FU,
			(data4 >>  26) & 0x3FU);
		printf("\n");
		)
			addData64(s_Hist, threadPos, (data4 >>  2) & 0x3FU);
		addData64(s_Hist, threadPos, (data4 >> 10) & 0x3FU);
		addData64(s_Hist, threadPos, (data4 >> 18) & 0x3FU);
		addData64(s_Hist, threadPos, (data4 >> 26) & 0x3FU);
	}

	__syncthreads();

	////////////////////////////////////////////////////////////////////////////
	// Merge per-thread histograms into per-block and write to global memory.
	// Start accumulation positions for half-warp each thread are shifted
	// in order to avoid bank conflicts.
	// See supplied whitepaper for detailed explanations.
	////////////////////////////////////////////////////////////////////////////

	if(threadIdx.x < HISTO_64_BIN_COUNT){
		GCUDA_KRNL_DBG_SHOW_THREAD_POS();
		unsigned int sum = 0;
		const int value = threadIdx.x;

		const int valueBase = IMUL(value, THREAD_N);
		const int  startPos = IMUL(threadIdx.x & 15, 4);

		//Threads with non-zero start positions wrap around the THREAD_N border
		for(int i = 0, accumPos = startPos; i < THREAD_N; i++){
			sum += s_Hist[valueBase + accumPos];
			if(++accumPos == THREAD_N) accumPos = 0;
		}

#if ATOMICS
		atomicAdd(d_Result + value, sum);
#else
		d_Result[IMUL(HISTO_64_BIN_COUNT, blockIdx.x) + value] = sum;
#endif
	}
	GCUDA_KRNL_DBG_LAST_THREAD("histogram64Kernel",int i=0;)
}



////////////////////////////////////////////////////////////////////////////////
// Merge blockN histograms into gridDim.x histograms
// blockDim.x == HISTO_64_BIN_COUNT
// gridDim.x  == BLOCK_N2
////////////////////////////////////////////////////////////////////////////////
__GCU_FCT_GLOBAL void mergeHistogram64Kernel(unsigned int *d_Result, int blockN){
	const int  globalTid = IMUL(blockIdx.x, HISTO_64_BIN_COUNT) + threadIdx.x;
	const int numThreads = IMUL(gridDim.x,  HISTO_64_BIN_COUNT);
	const int   dataSize = IMUL(blockN,     HISTO_64_BIN_COUNT);

	int sum = 0;
	for(int pos = globalTid; pos < dataSize; pos += numThreads)
		sum += d_Result[pos];

	d_Result[globalTid] = sum;
}


////////////////////////////////////////////////////////////////////////////////
// CPU interface to GPU histogram calculator
////////////////////////////////////////////////////////////////////////////////
//histogram64kernel() results buffer
unsigned int *d_ResultHisto64;
unsigned int *h_ResultHisto64GPU;
//Maximum block count for histogram64kernel()
//Limits input data size to 756MB
const int MAX_BLOCK_N = 16384;

//Internal memory allocation
const int BLOCK_N2 = 32;

void initHistogram64(void){
	h_ResultHisto64GPU = (unsigned int *)malloc(HISTOGRAM_SIZE * BLOCK_N2);
#if ATOMICS
	GCU_CUDA_SAFE_CALL( cudaMalloc((void **)&d_ResultHisto64, HISTOGRAM_SIZE) );
#else
	GCU_CUDA_SAFE_CALL( cudaMalloc((void **)&d_ResultHisto64, MAX_BLOCK_N * HISTOGRAM_SIZE) );
#endif
}

//Internal memory deallocation
void closeHistogram64(void){
	GCU_CUDA_SAFE_CALL( cudaFree(d_ResultHisto64) );

}
//////////////////////////////////////////////////////////////////////////////////


void histogram64GPU(
					unsigned int *d_Data,
					int *h_Result,//OpenCV store results in 'int' not 'uint'
					//int *d_Result,//OpenCV store results in 'int' not 'uint'
					int dataN
					)
{
						const int  blockN = iDivUp(dataN / 4, THREAD_N * 63);
#if 0//def _DEBUG
						printf("histogram64GPU()\n");
						printf("- dataN: %d\n", dataN);
						printf("- THREAD_N: %d\n", THREAD_N);
						printf("- blockN: %d\n", blockN);
						printf("- MAX_BLOCK_N: %d\n", MAX_BLOCK_N);

						if(blockN > MAX_BLOCK_N){
							printf("histogram64gpu(): data size exceeds maximum\n");
							return;
						}
#endif
#if ATOMICS
						GCU_CUDA_SAFE_CALL( cudaMemset(d_ResultHisto64, 0, HISTOGRAM_SIZE) );
						histogram64Kernel<<<blockN, THREAD_N>>>(
							d_Result,
							(unsigned int *)d_Data,
							dataN / 4
							);
						CUT_CHECK_ERROR("histogram64Kernel() execution failed\n");
						GCU_CUDA_SAFE_CALL( cudaMemcpy(h_Result, d_ResultHisto64, HISTOGRAM_SIZE, cudaMemcpyDeviceToHost) );
#else
						histogram64Kernel<<<blockN, THREAD_N>>>(
							d_ResultHisto64,
							(unsigned int *)d_Data,
							dataN / 4
							);
						CUT_CHECK_ERROR("histogram64Kernel() execution failed\n");


						const int FINAL_BLOCK_N = min(BLOCK_N2, blockN);
						if(BLOCK_N2 < blockN){
							mergeHistogram64Kernel<<<BLOCK_N2, HISTO_64_BIN_COUNT>>>(
								d_ResultHisto64,
								blockN
								);
							CUT_CHECK_ERROR("mergeHistogram64Kernel() execution failed\n");
						}

						GCU_CUDA_SAFE_CALL( cudaMemcpy(h_ResultHisto64GPU, d_ResultHisto64, FINAL_BLOCK_N * HISTOGRAM_SIZE, cudaMemcpyDeviceToHost) );
						for(int i = 1; i < FINAL_BLOCK_N; i++){
							for(int j = 0; j < HISTO_64_BIN_COUNT; j++)
								h_ResultHisto64GPU[j] += h_ResultHisto64GPU[i * HISTO_64_BIN_COUNT + j];
						}
						for(int i = 0; i < HISTO_64_BIN_COUNT; i++)
							h_Result[i] = (int)h_ResultHisto64GPU[i];

						free(h_ResultHisto64GPU);

#endif
}

#endif//_GPUCV_CUDA_CV_CU_HISTO64_KERNEL_H
