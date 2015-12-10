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

#ifndef __GPUCV_CUDA_CXCOREGCU_LOCALSUM_H
#define __GPUCV_CUDA_CXCOREGCU_LOCALSUM_H

#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>
#include <GPUCVCuda/base_kernels/cxcoregcu_arithm_fct.kernel.h>

typedef int1	InputType;
typedef int1	OutputType;


__GCU_FCT_GLOBAL 
void LocalSumKernel(InputType *src1,OutputType *dst,int h,int w,int width,int height,
					size_t pitch)

{	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
unsigned int xIndex = xBlock + threadIdx.x;
unsigned int yIndex = yBlock + threadIdx.y;
//h=h-1;
//w=w-1;

if (xIndex < width && yIndex < height)
{
	unsigned int _IndexOut  = __mul24(width/*__mul24(blockDim.x, gridDim.x)*/, yIndex) + xIndex;

	int _Indexru =	_IndexOut + w;
	int _Indexll =	__mul24(width/*__mul24(blockDim.x, gridDim.x)*/, yIndex + h) + xIndex;
	int _Indexlr =	__mul24(width/*__mul24(blockDim.x, gridDim.x)*/, yIndex + h) + xIndex + w;




	// right upper corner
	if ( xIndex + w > width - 1 ) 
	{		_Indexru = __mul24(width/*__mul24(blockDim.x, gridDim.x)*/, yBlock + threadIdx.y) + width-1;
	_Indexlr = __mul24(width/*__mul24(blockDim.x, gridDim.x)*/, yBlock + threadIdx.y + h) + width-1; 
	}


	//lower left corner
	if ( (yIndex + h ) > height-1)
	{			_Indexll = __mul24(width/*__mul24(blockDim.x, gridDim.x)*/, (height-1)) +  xBlock + threadIdx.x;
	_Indexlr = __mul24(width/*__mul24(blockDim.x, gridDim.x)*/, (height-1)) +  xBlock + threadIdx.x + w;

	}


	//right lower corner 
	if ((yIndex + h ) >= height-1)
	{	if ( xIndex + w >= width-1 ) 
	{
		_Indexlr = __mul24(width/*__mul24(blockDim.x, gridDim.x)*/, (height-1))+ width-1;
	}

	}

	//reading the pixel value at all corners

	OutputType lu, ru, rl, ll, result;
	lu.x = src1[_IndexOut].x;
	ru.x = src1[_Indexru].x;
	rl.x = src1[_Indexlr].x;
	ll.x = src1[_Indexll].x;
	result.x = (rl.x+lu.x-ll.x-ru.x)/w*h;

	//writing out the outputll1].x;//src1[].x
	dst[_IndexOut].x = result.x;
	//GCU_OP_MULTIPLEXER< 1 ,KERNEL_ARITHM_OPER_CLAMP>::Do(dst[_IndexOut], result);//, TPLDataSrc,float4//clamp result into destination data

}
}


#endif

