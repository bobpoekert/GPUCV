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
#ifndef __GPUCV_CVGCU_CVTCOLOR_H
#define __GPUCV_CVGCU_CVTCOLOR_H
#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>

/* CvtColor with Cuda
* Device code.
*/

// This kernel is optimized to ensure all global reads and writes are coalesced,
// and to avoid bank conflicts in shared memory.  This kernel is up to 11x faster
// than the naive kernel below.  Note that the shared memory array is sized to 
// (BLOCK_DIM+1)*BLOCK_DIM.  This pads each row of the 2D block in shared memory 
// so that bank conflicts do not occur when threads address the array column-wise.

template < typename TPLDataDst,typename TPLDataSrc>
__global__ void gcudaKernel_CvtColor_BGR2GRAY(TPLDataSrc* srcArr,TPLDataDst* dstArr, int width, int height, float scale=1., float shift=0.)
{
	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;
	unsigned int index = yIndex * width + xIndex;

	if(xIndex < width && yIndex < height)
	{
		TPLDataSrc input=srcArr[index];
		float Y = input.x * .212671 + input.y* .715160 + input.z * .072169;
		Y = Y*scale+shift;
		_Clamp(dstArr[index], Y);
	}
}


template < typename TPLDataDst,typename TPLDataSrc>
__global__ void gcudaKernel_CvtColor_RGB2GRAY(TPLDataSrc* srcArr,TPLDataDst* dstArr, int width, int height, float scale=1., float shift=0.)
{
	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;
	unsigned int index = yIndex * width + xIndex;

	if(xIndex < width && yIndex < height)
	{
		TPLDataSrc input=srcArr[index];
		float Y = input.z * .212671 + input.y * .715160 + input.x * .072169;
		Y = Y*scale+shift;
		_Clamp(dstArr[index], Y);
	}
}  


template < typename TPLDataDst,typename TPLDataSrc>
__global__ void gcudaKernel_CvtColor_BGR2YCrCb(TPLDataSrc* srcArr,TPLDataDst* dstArr, int width, int height, float scale=1., float shift=0.)
{
	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;
	unsigned int index = yIndex * width + xIndex;

	if(xIndex < width && yIndex < height)
	{
		TPLDataSrc input=srcArr[index];
		TPLDataDst output;
		float Y = (.299*input.x + .587*input.y + .114*input.z)*scale+shift;
		float Cr =((input.z - Y)*.713 + .5)*scale+shift;
		float Cb =((input.y - Y)*.564 + .5)*scale+shift;
		_Clamp(output.z, Y);
		_Clamp(output.y, Cr);
		_Clamp(output.x, Cb);
		dstArr[index]=output;
	}

}


template < typename TPLDataDst,typename TPLDataSrc>
__global__ void gcudaKernel_CvtColor_RGB2YCrCb(TPLDataSrc* srcArr,TPLDataDst* dstArr, int width, int height, float scale=1., float shift=0.)
{
	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;
	unsigned int index = yIndex * width + xIndex;

	if(xIndex < width && yIndex < height)
	{
		TPLDataSrc input=srcArr[index];
		TPLDataDst output;
		float Y		= (.299*input.z + .587*input.y + .114*input.x)*scale+shift;
		float Cr	= ((input.z - Y)*.713 + .5)*scale+shift;
		float Cb	= ((input.y - Y)*.564 + .5)*scale+shift;
		_Clamp(output.z, Y);
		_Clamp(output.y, Cr);
		_Clamp(output.x, Cb);
		dstArr[index]=output;
	}
}


#endif	
