/*
* Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:   
*
* This source code is subject to NVIDIA ownership rights under U.S. and 
* international Copyright laws.  
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
* OR PERFORMANCE OF THIS SOURCE CODE.  
*
* U.S. Government End Users.  This source code is a "commercial item" as 
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
* "commercial computer software" and "commercial computer software 
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
* and is provided to the U.S. Government only as a commercial end item.  
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
* source code with only those rights set forth herein.
*/

/*
Parallel reduction kernels
*/
#if 0//
#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <stdio.h>
#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>
#include <GPUCVCuda/gcu_runtime_api_wrapper.h>

#ifdef __DEVICE_EMULATION__
#define SYNC __syncthreads()
#else
#define SYNC
#endif

/*
Parallel sum reduction using shared memory
- takes log(n) steps for n input elements
- uses n threads
- only works for power-of-2 arrays
*/

/* This version is very inefficient -- interleaved addressing results
in divergent branching within warps */
__GCU_FCT_GLOBAL void
reduce0(int *g_idata, int *g_odata)
{
	extern __shared__ int sdata[];

	// load shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/* This version still uses interleaved addressing, but without divergent 
branching.  However this results in many shared memory bank conflicts. */
__GCU_FCT_GLOBAL void
reduce1(int *g_idata, int *g_odata)
{
	extern __shared__ int sdata[];

	// load shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;

		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
This version uses sequential addressing -- no divergence or bank conflicts.
*/
__GCU_FCT_GLOBAL void
reduce2(int *g_idata, int *g_odata)
{
	extern __shared__ int sdata[];

	// load shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
This version uses n/2 threads --
it performs the first level of reduction when reading from global memory
*/
__GCU_FCT_GLOBAL void
reduce3(int *g_idata, int *g_odata)
{
	extern __shared__ int sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem 
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
This version unrolls the last warp to avoid synchronization where it 
isn't needed
*/
__GCU_FCT_GLOBAL void
reduce4(int *g_idata, int *g_odata)
{
	extern __shared__ int sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();

	// do reduction in shared mem
	for(unsigned int s=blockDim.x/2; s>32; s>>=1) 
	{
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}

#ifndef __DEVICE_EMULATION__
	if (tid < 32)
#endif
	{
		sdata[tid] += sdata[tid + 32]; SYNC;
		sdata[tid] += sdata[tid + 16]; SYNC;
		sdata[tid] += sdata[tid +  8]; SYNC;
		sdata[tid] += sdata[tid +  4]; SYNC;
		sdata[tid] += sdata[tid +  2]; SYNC;
		sdata[tid] += sdata[tid +  1]; SYNC;
	}

	// write result for this block to global mem 
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
This version is completely unrolled.  It uses a template parameter to achieve 
optimal code for any (power of 2) number of threads.  This requires a switch 
statement in the host code to handle all the different thread block sizes at 
compile time.
*/
template <unsigned int blockSize>
__GCU_FCT_GLOBAL void
reduce5(int *g_idata, int *g_odata)
{
	extern __shared__ int sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i+blockSize];
	__syncthreads();

	// do reduction in shared mem
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
	if (tid < 32)
#endif
	{
		if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; SYNC; }
		if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; SYNC; }
		if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; SYNC; }
		if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; SYNC; }
		if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; SYNC; }
		if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; SYNC; }
	}

	// write result for this block to global mem 
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
This version adds multiple elements per thread sequentially.  This reduces the overall
cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
(Brent's Theorem optimization)
*/
template <unsigned int blockSize>
__GCU_FCT_GLOBAL void
CudaReduceSum(unsigned char *g_idata, int *g_odata, unsigned int n)
{
	extern __shared__ int sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;

	// we reduce multiple elements per thread.  The number is determined by the 
	// number of active thread blocks (via gridSize).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		sdata[tid] += g_idata[i] + g_idata[i+blockSize];  
		i += gridSize;
	} 
	__syncthreads();

	// do reduction in shared mem
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

#ifndef __DEVICE_EMULATION__
	if (tid < 32)
#endif
	{
		if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; SYNC; }
		if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; SYNC; }
		if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; SYNC; }
		if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; SYNC; }
		if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; SYNC; }
		if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; SYNC; }
	}

	// write result for this block to global mem 
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

#endif // #ifndef _REDUCE_KERNEL_H_
#endif//