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

#include <stdio.h>
#include <stdlib.h>
#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>
#include <GPUCVCuda/gcu_runtime_api_wrapper.h>


#define KERNEL_RADIUS 1
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)
 __constant__ float c_Kernel_H[KERNEL_LENGTH];
 __constant__ float c_Kernel_V[KERNEL_LENGTH];
 __constant__ float c_Kernel_SQ[KERNEL_LENGTH*KERNEL_LENGTH];

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 16
#define   ROWS_BLOCKDIM_Y 4
#define ROWS_RESULT_STEPS 4
#define   ROWS_HALO_STEPS 1

template <typename TPLSrc, typename TPLDst>
__global__ void convolutionRowsKernel(
     TPLSrc *d_Src,
    TPLDst *d_Dst,
    int imageW,
    int imageH,
    int pitch
){
    __shared__ TPLSrc s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Main data
    #pragma unroll
    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];

    //Left halo
    for(int i = 0; i < ROWS_HALO_STEPS; i++){
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = 
            (baseX >= -i * ROWS_BLOCKDIM_X ) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Right halo
    for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++){
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] =
            (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
        float sum = 0;

        #pragma unroll
        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            sum += c_Kernel_V[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];

        //_Clamp(d_Dst[i * ROWS_BLOCKDIM_X],sum);
		d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   COLUMNS_BLOCKDIM_X 16
#define   COLUMNS_BLOCKDIM_Y 8
#define COLUMNS_RESULT_STEPS 4
#define   COLUMNS_HALO_STEPS 1

template <typename TPLSrc, typename TPLDst>
__global__ void convolutionColumnsKernel(
    TPLSrc *d_Src,
    TPLDst *d_Dst,
    int imageW,
    int imageH,
    int pitch
){
    __shared__  TPLSrc s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];
	
    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Main data
    #pragma unroll
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];

    //Upper halo
    for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 
            (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    //Lower halo
    for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
        float sum = 0;
        #pragma unroll
        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            sum += c_Kernel_H[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];

		 //_Clamp(d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch],sum);
		d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}


//work the same has both previous kernels by use a square filter of size 3x3
template <typename TPLSrc, typename TPLDst>
__global__ void convolutionKernel_3x3(
    TPLSrc *d_Src,
    TPLDst *d_Dst,
    int imageW,
    int imageH,
    int pitch
){
    __shared__  TPLSrc s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];
	
    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Main data
    #pragma unroll
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];

    //Upper halo
    for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = 
            (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    //Lower halo
    for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] =
            (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    //Compute and store results
    __syncthreads();
    #pragma unroll
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
        float sum = 0;
        #pragma unroll
		/*
        for(int j = 0; j <= KERNEL_LENGTH; j++)
			for(int k = 0; k <= KERNEL_LENGTH; k++)
            sum += c_Kernel_SQ[j + k* KERNEL_LENGTH] * s_Data[threadIdx.x][threadIdx.y + (i (j-KERNEL_RADIUS) ) * COLUMNS_BLOCKDIM_Y + (j-KERNEL_RADIUS)];
		*/
		 //_Clamp(d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch],sum);
		d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}



#if 0//old code..??

/** \brief Contains convolution and neighborhoods filters for CUDA.
\author Yannick Allusse
*/
#ifndef _GCU_CVGCU_CONVOL_FILTER_KERNEL_H
#define _GCU_CVGCU_CONVOL_FILTER_KERNEL_H
#define __STLPFILTER_KERNEL_DEVICE_FCT			__device__ static
#ifdef _MSC_VER
#define __STLPFILTER_KERNEL_DEVICE_FCT_INLINE	inline __STLPFILTER_KERNEL_DEVICE_FCT
#else
#define __STLPFILTER_KERNEL_DEVICE_FCT_INLINE	__STLPFILTER_KERNEL_DEVICE_FCT
#endif


//! Used to end looping when beginY = ENDY
//used to end looping when beginX = ENDX
#define __STLPFILTER_CREATE_MAX_XY(FCTNAME, TEMPDATA_TYPE)\
	template <typename TPLData,bool TPLCutomKernel, int X, int Y, int EndX>						\
struct FCTNAME<TPLData,TPLCutomKernel, X, Y, EndX, Y>							\
		{																						\
		__STLPFILTER_KERNEL_DEVICE_FCT_INLINE void Do(TEMPDATA_TYPE * Dest,TPLData **col, unsigned char *Params, float fScale){}	\
		};																						\
		template <typename TPLData,bool TPLCutomKernel, int X, int Y, int EndY> 					\
struct FCTNAME<TPLData,TPLCutomKernel, X, Y, X, EndY>							\
		{																						\
		__STLPFILTER_KERNEL_DEVICE_FCT_INLINE void Do(TEMPDATA_TYPE * Dest,TPLData **col, unsigned char *Params, float fScale){}	\
		};

//! beginY = BeginX = 0=> first element of loop.
#define __STLPFILTER_CREATE_INIT_DST_VAL(FCTNAME, TEMPDATA_TYPE, VAL)\
	template <typename TPLData,bool TPLCutomKernel, int EndX, int EndY> \
struct FCTNAME<TPLData,TPLCutomKernel, 0, 0, EndX, EndY>\
	{\
	__STLPFILTER_KERNEL_DEVICE_FCT_INLINE void Do(TEMPDATA_TYPE * Dest,TPLData **col, unsigned char *Params, float fScale)\
		{\
		*Dest=VAL;\
		FCTNAME<TPLData, TPLCutomKernel, 0,1, EndX, EndY>::Do(Dest, col, Params,fScale);\
		FCTNAME<TPLData, TPLCutomKernel, 1,0, EndX, EndY>::Do(Dest, col, Params,fScale);\
		}\
	};

//=============================================================
//
//	Erode Filter (nxm)
//
//=============================================================

#define TEST_ERODE 1
//! Used to loop from begin to end
template <typename TPLData, bool TPLCutomKernel, int BeginX, int BeginY, int EndX, int EndY>
struct StLpFilter_Erode
{
	__STLPFILTER_KERNEL_DEVICE_FCT void Do(TPLData * Dest,TPLData **col, unsigned char *Params, float fScale)
	{
		DoSomething(Dest, col,Params,fScale);
#if !TEST_ERODE
		StLpFilter_Erode<TPLData, TPLCutomKernel, BeginX,BeginY + 1,  EndX, EndY>::Do(Dest, col, Params,fScale);
		StLpFilter_Erode<TPLData, TPLCutomKernel, BeginX + 1, BeginY, EndX, EndY>::Do(Dest, col, Params,fScale);
#endif
	}
	__STLPFILTER_KERNEL_DEVICE_FCT void DoSomething(TPLData * Dest, TPLData **col,unsigned char *Params, float fScale)
	{
#if TPLCutomKernel
#define __LOCAL_CUSTOM_ELEM(Col, Row) *Params[Col+3*Row]
#else
#define __LOCAL_CUSTOM_ELEM(Col, Row)
#endif
#if TEST_ERODE
		*Dest = 255;
		for(int i =0; i< BeginX; i++)
		{
			for(int j =0; j< BeginY; j++)
			{
				*Dest = min(*Dest, col[i][j] __LOCAL_CUSTOM_ELEM(j,i));
				//*Dest += col[i][j] __LOCAL_CUSTOM_ELEM(j,i);
			}
		}
		//*Dest = 12;
#else
		*Dest = min(*Dest, col[BeginX][BeginY] __LOCAL_CUSTOM_ELEM(BeginY,BeginX));
#endif

	}
};
#if !TEST_ERODE
__STLPFILTER_CREATE_MAX_XY		(StLpFilter_Erode, TPLData);
__STLPFILTER_CREATE_INIT_DST_VAL(StLpFilter_Erode, TPLData, 255);
#endif

//=============================================================
//
//	Dilate Filter (nxm)
//
//=============================================================

//! Used to loop from begin to end
template <typename TPLData, bool TPLCutomKernel, int BeginX, int BeginY, int EndX, int EndY>
struct StLpFilter_Dilate
{
	__STLPFILTER_KERNEL_DEVICE_FCT void Do(TPLData * Dest,TPLData **col, unsigned char *Params, float fScale)
	{
		DoSomething(Dest, col,Params,fScale);
		StLpFilter_Dilate<TPLData, TPLCutomKernel, BeginX,BeginY + 1,  EndX, EndY>::Do(Dest, col, Params,fScale);
		StLpFilter_Dilate<TPLData, TPLCutomKernel, BeginX + 1, BeginY, EndX, EndY>::Do(Dest, col, Params,fScale);
	}
	__STLPFILTER_KERNEL_DEVICE_FCT void DoSomething(TPLData * Dest, TPLData **col,unsigned char *Params, float fScale)
	{
#if TPLCutomKernel
#define __LOCAL_CUSTOM_ELEM(Col, Row) *Params[Col+3*Row]
#else
#define __LOCAL_CUSTOM_ELEM(Col, Row)
#endif
		*Dest = max(*Dest, col[BeginX][BeginY] __LOCAL_CUSTOM_ELEM(BeginY,BeginX));
	}
};
__STLPFILTER_CREATE_MAX_XY		(StLpFilter_Dilate, TPLData);
__STLPFILTER_CREATE_INIT_DST_VAL(StLpFilter_Dilate, TPLData, 0);


//=============================================================
//
//	Sobel Filter (3x3)
//
//=============================================================

//! Used to loop from begin to end
template <typename TPLData, bool TPLCutomKernel, int BeginX, int BeginY, int EndX, int EndY>
struct StLpFilter_Sobel
{
	__STLPFILTER_KERNEL_DEVICE_FCT_INLINE void Do(TPLData * Dest,TPLData **col, unsigned char *Params, float fScale)
	{}
	__STLPFILTER_KERNEL_DEVICE_FCT_INLINE void DoSomething(TPLData * Dest, TPLData **col,unsigned char *Params, float fScale)
	{}
};
//__STLPFILTER_CREATE_MAX_XY(StLpFilter_Sobel);
template <typename TPLData,int BeginX, int BeginY>
struct StLpFilter_Sobel<TPLData, false, BeginX, BeginY,3,3>
{
	__STLPFILTER_KERNEL_DEVICE_FCT_INLINE void Do(TPLData * Dest,TPLData **col, unsigned char *Params, float fScale)
	{
		short Horz = col[2][0] + 2*col[2][1] + col[2][2] - col[0][0] - 2*col[0][1] - col[0][2];//ur + 2*mr + lr - ul - 2*ml - ll;
		short Vert = col[2][0] + 2*col[2][1] + col[2][2] - col[0][0] - 2*col[0][1] - col[0][2];//ul + 2*um + ur - ll - 2*lm - lr;
		short Sum = (short) (fScale*(abs(Horz)+abs(Vert)));
		if ( Sum < 0 ) *Dest=0;
		else if ( Sum > 0xff ) *Dest=0xff;
		else *Dest=Sum;
	}
};

//=============================================================
//
//	Laplace Filter (3x3)
/*	optimized Laplace for :
|0  1  0|
|1 -4  1|
|0  1  0|
else we use custom kernel elements
*/
//=============================================================

//! Used to loop from begin to end
template <typename TPLData, bool TPLCutomKernel, int BeginX, int BeginY, int EndX, int EndY>
struct StLpFilter_Laplace
{
	__STLPFILTER_KERNEL_DEVICE_FCT_INLINE void Do(TPLData * Dest,TPLData **col, unsigned char *Params, float fScale)
	{}
	__STLPFILTER_KERNEL_DEVICE_FCT_INLINE void DoSomething(TPLData * Dest, TPLData **col,unsigned char *Params, float fScale)
	{}
};
//__STLPFILTER_CREATE_MAX_XY(StLpFilter_Sobel);
template <typename TPLData, int BeginX, int BeginY>
struct StLpFilter_Laplace<TPLData, false, BeginX, BeginY,3,3>
{
	__STLPFILTER_KERNEL_DEVICE_FCT_INLINE void Do(TPLData * Dest,TPLData **col, unsigned char *Params, float fScale)
	{
		*Dest = col[1][0] + col[0][1] + col[2][1] + col[1][2] - 4*col[1][1];
		if ( *Dest < 0 ) *Dest=0;
		else if ( *Dest > 0xff ) *Dest=0xff;
	}
};




//=============================================================
//
//	Smooth Filter (nxm) CV_BLUR & CV_BLUR_NO_SCALE
//
//=============================================================

//! Used to loop from begin to end
typedef float BLUR_TEMP_TYPE;
template <typename TPLData, bool TPLCutomKernel, int BeginX, int BeginY, int EndX, int EndY>
struct StLpFilter_Smooth_BLUR
{
	__STLPFILTER_KERNEL_DEVICE_FCT void Do(BLUR_TEMP_TYPE * Dest,TPLData **col, unsigned char *Params, float fScale)
	{
		DoSomething(Dest, col,Params,fScale);
		StLpFilter_Smooth_BLUR<TPLData, TPLCutomKernel, BeginX,BeginY + 1,  EndX, EndY>::Do(Dest, col, Params,fScale);
		StLpFilter_Smooth_BLUR<TPLData, TPLCutomKernel, BeginX + 1, BeginY, EndX, EndY>::Do(Dest, col, Params,fScale);
	}

	__STLPFILTER_KERNEL_DEVICE_FCT void DoSomething(BLUR_TEMP_TYPE * Dest, TPLData **col,unsigned char *Params, float fScale)
	{
		*Dest += col[BeginX][ BeginY];
	}
};
__STLPFILTER_CREATE_MAX_XY(StLpFilter_Smooth_BLUR, BLUR_TEMP_TYPE);
//__STLPFILTER_CREATE_INIT_DST_VAL(StLpFilter_Smooth_BLUR,0);

//First operations is also the last oneLast operations...we multiply by float
template <typename TPLData, bool TPLCutomKernel, int EndX, int EndY>
struct StLpFilter_Smooth_BLUR<TPLData,TPLCutomKernel, 0 , 0, EndX, EndY>
{
	__STLPFILTER_KERNEL_DEVICE_FCT void Do(TPLData * Dest,TPLData **col, unsigned char *Params, float fScale)
	{
		//*Dest =0;
		BLUR_TEMP_TYPE TempVal = 0;
		DoSomething(&TempVal, col,Params,fScale);
		StLpFilter_Smooth_BLUR<TPLData, TPLCutomKernel, 0,1,  EndX, EndY>::Do(&TempVal, col, Params,fScale);
		StLpFilter_Smooth_BLUR<TPLData, TPLCutomKernel, 1,0, EndX, EndY>::Do(&TempVal, col, Params,fScale);
		if(TempVal*fScale > 255)
			*Dest = 255;// - 128;
		else
			*Dest = TempVal*fScale;// - 128;
	}

	__STLPFILTER_KERNEL_DEVICE_FCT void DoSomething(BLUR_TEMP_TYPE * Dest, TPLData **col,unsigned char *Params, float fScale)
	{
		*Dest += col[0][0];
	}
};

/*
//! Used to end looping when beginY = BeginX = 0=> first element of loop.
template <typename TPLData,bool TPLCutomKernel, int X, int Y> 
struct StLpFilter_Erode<TPLData,TPLCutomKernel, X, Y, 0, 0>
{
__STLPFILTER_KERNEL_DEVICE_FCT void Do(TPLData * Dest,TPLData **col, unsigned char *Params)
{//init value to default
*Dest=0;
}
};*/

/*
//used to end looping when All Begin = end
template <typename TPLData,bool TPLCutomKernel, int X, int Y> 
struct StLpFilter_Erode <TPLData,TPLCutomKernel, X, Y, X, Y>
{
__STLPFILTER_KERNEL_DEVICE_FCT void Do(TPLData * Dest,TPLData **col, unsigned char *Params)
{}
};
*/

#endif
#endif//0