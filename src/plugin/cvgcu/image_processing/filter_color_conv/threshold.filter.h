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
#ifndef __GPUCV_CVGCU_THRESHOLDFILTER_H
#define __GPUCV_CVGCU_THRESHOLDFILTER_H
#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>
#include <GPUCVCuda/base_kernels/cxcoregcu_arithm_fct.kernel.h>

template <typename TPLDataSrc,typename TPLDataDst>
__global__ void gcuThresholdKernel_1(TPLDataSrc* in,TPLDataDst* out, float threshold, float max_val, uchar threshold_type, unsigned int width, unsigned int height)
{
	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;
	unsigned int index = yIndex * width + xIndex;

	if(xIndex < width && yIndex < height)
	{			
		if(threshold_type == 0)
		{/*threshold_type=CV_THRESH_BINARY: dst(x,y) = max_value, if src(x,y)>threshold else 0*/				
			if(in[index]> threshold)
				out[index]=max_val;
			else
				out[index]=0;	
		}
		else if(threshold_type == 1)
		{/*threshold_type=CV_THRESH_BINARY_INV: dst(x,y) = 0, if src(x,y)> else max_value*/
			if(in[index]> threshold)
				out[index]=0;
			else
				out[index]=max_val;
		}
		else if(threshold_type == 2)
		{/*threshold_type=CV_THRESH_TRUNC:dst(x,y) = threshold, if src(x,y)>threshold else src(x,y)*/
			if(in[index]> threshold)
				out[index]=threshold;
			else
				out[index]=in[index];
		}
		else if(threshold_type == 3)
		{/*threshold_type=CV_THRESH_TOZERO: dst(x,y) = src(x,y), if src(x,y)>threshold else 0*/
			if(in[index]> threshold)
				out[index]=in[index];
			else
				out[index]=0;
		}
		else if(threshold_type == 4)
		{/*threshold_type=CV_THRESH_TOZERO_INV:dst(x,y) = 0, if src(x,y)>threshold else src(x,y)*/
			if(in[index]< threshold)
				out[index]=in[index];
			else
				out[index]=0;
		}
	}
}

#if 0//multiple channels support might improve performances by using the reshape feature

template < int TPLChannels, typename TPLDataSrcFULL,typename TPLDataDstFULL>
__global__ void gcuThresholdKernel(TPLDataSrcFULL* in,TPLDataDstFULL* out, float4 threshold, float4 max_val, uchar threshold_type, unsigned int width, unsigned int height)
{
	unsigned int xBlock = __mul24(blockDim.x, blockIdx.x);
	unsigned int yBlock = __mul24(blockDim.y, blockIdx.y);
	unsigned int xIndex = xBlock + threadIdx.x;
	unsigned int yIndex = yBlock + threadIdx.y;
	unsigned int index = yIndex * width + xIndex;

	if(xIndex < width && yIndex < height)
	{
		TPLDataSrcFULL inputVal=in[index];
		TPLDataSrcFULL threshVal; 	
		TPLDataSrcFULL tempVal;
		TPLDataSrcFULL maxVal;
		TPLDataSrcFULL maskVal;
		GCU_OP_MULTIPLEXER <TPLChannels,KERNEL_ARITHM_OPER_AFFECT, TPLDataSrcFULL, float4>::Do(threshVal, threshold);			
		GCU_OP_MULTIPLEXER <TPLChannels,KERNEL_ARITHM_OPER_AFFECT, TPLDataSrcFULL, float4>::Do(maxVal, max_val);					
		//create mask
		GCU_OP_MULTIPLEXER <TPLChannels,KERNEL_LOGIC_OPER_GREATER, TPLDataSrcFULL, TPLDataSrcFULL, TPLDataSrcFULL>::Do(maskVal, inputVal, threshVal);
		uchar4 uc4Factor;
		uc4Factor.x=uc4Factor.y=uc4Factor.z= uc4Factor.w = 255;
		GCU_OP_MULTIPLEXER <TPLChannels,KERNEL_ARITHM_OPER_DIV, TPLDataDstFULL, TPLDataSrcFULL, TPLDataSrcFULL>::Do(maskVal, maskVal, uc4Factor);		
			
		if(threshold_type == 0)
		{/*threshold_type=CV_THRESH_BINARY: dst(x,y) = max_value, if src(x,y)>threshold else 0*/				
			//apply mask...			
			GCU_OP_MULTIPLEXER <TPLChannels,KERNEL_ARITHM_OPER_MUL, TPLDataDstFULL, TPLDataSrcFULL, TPLDataSrcFULL>::Do(tempVal, maskVal, maxVal);	
		}
		else if(threshold_type == 1)
		{/*threshold_type=CV_THRESH_BINARY_INV: dst(x,y) = 0, if src(x,y)> else max_value*/
			//apply mask...			
			GCU_OP_MULTIPLEXER <TPLChannels,KERNEL_LOGIC_OPER_NOTAND, TPLDataDstFULL, TPLDataSrcFULL, TPLDataSrcFULL>::Do(tempVal, maskVal, maxVal);
			TPLDataSrcFULL factor;		
			GCU_OP_MULTIPLEXER <TPLChannels,KERNEL_ARITHM_OPER_MUL, TPLDataDstFULL, TPLDataSrcFULL, TPLDataSrcFULL>::Do(tempVal, maskVal, maxVal);	
		}

/*
threshold_type=CV_THRESH_TRUNC:
dst(x,y) = threshold, if src(x,y)>threshold
           src(x,y), otherwise

threshold_type=CV_THRESH_TOZERO:
dst(x,y) = src(x,y), if src(x,y)>threshold
           0, otherwise

threshold_type=CV_THRESH_TOZERO_INV:
dst(x,y) = 0, if src(x,y)>threshold
           src(x,y), otherwise


threshold_type=CV_THRESH_BINARY:
dst(x,y) = max_value, if src(x,y)>threshold
           0, otherwise

threshold_type=CV_THRESH_BINARY_INV:
dst(x,y) = 0, if src(x,y)>threshold
           max_value, otherwise

threshold_type=CV_THRESH_TRUNC:
dst(x,y) = threshold, if src(x,y)>threshold
           src(x,y), otherwise

threshold_type=CV_THRESH_TOZERO:
dst(x,y) = src(x,y), if src(x,y)>threshold
           0, otherwise

threshold_type=CV_THRESH_TOZERO_INV:
dst(x,y) = 0, if src(x,y)>threshold
           src(x,y), otherwise
*/

	/*	if(blockDim.x < 4)
			out[index].x = fInputVal.x;
		else if(blockDim.x < 8)
			out[index].x = fMaskVal.x;
		else if(blockDim.x < 12)
	*/		out[index] = tempVal;
	/*	else 
			out[index].x = inputVal.x;
	*/	//out[index] = in[index];
	}
}
#endif

#endif	
