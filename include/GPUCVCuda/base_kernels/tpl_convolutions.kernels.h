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
#ifndef __GPUCV_CUDA_BASE_CONVOL_KERNEL_H
#define __GPUCV_CUDA_BASE_CONVOL_KERNEL_H

#include <stdio.h>
#include <stdlib.h>
#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>
#include <GPUCVCuda/gcu_runtime_api_wrapper.h>
//#include <cvgcu/image_processing/filter_color_conv/cvgcu_convol_filter.kernel.h>

#define SV 0.003921f
#define IV 255.f

// Texture reference for reading image
texture<unsigned char, 2> texSobel;
extern __shared__ unsigned char LocalBlock[];


#define Radius 1
#if 0//to remove.???
__GCU_FCT_DEVICE unsigned char
ComputeSobel(unsigned char ul, // upper left
			 unsigned char um, // upper middle
			 unsigned char ur, // upper right
			 unsigned char ml, // middle left
			 unsigned char mm, // middle (unused)
			 unsigned char mr, // middle right
			 unsigned char ll, // lower left
			 unsigned char lm, // lower middle
			 unsigned char lr, // lower right
			 float fScale )
{
	short Horz = ur + 2*mr + lr - ul - 2*ml - ll;
	short Vert = ul + 2*um + ur - ll - 2*lm - lr;
	short Sum = (short) (fScale*(abs(Horz)+abs(Vert)));
	if ( Sum < 0 ) return 0; else if ( Sum > 0xff ) return 0xff;
	return (unsigned char) Sum;
}


__GCU_FCT_GLOBAL void 
SobelSharedKernel( uchar4 *pSobelOriginal, unsigned short SobelPitch, 
				  short BlockWidth, short SharedPitch,
				  short w, short h, float fScale )
{ 
	short u = 4*blockIdx.x*BlockWidth;
	short v = blockIdx.y*blockDim.y + threadIdx.y;
	short ib;

	int SharedIdx = threadIdx.y * SharedPitch;

	for ( ib = threadIdx.x; ib < BlockWidth+2*Radius; ib += blockDim.x ) {
		LocalBlock[SharedIdx+4*ib+0] = tex2D( texSobel, 
			(float) (u+4*ib-Radius+0), (float) (v-Radius) );
		LocalBlock[SharedIdx+4*ib+1] = tex2D( texSobel, 
			(float) (u+4*ib-Radius+1), (float) (v-Radius) );
		LocalBlock[SharedIdx+4*ib+2] = tex2D( texSobel, 
			(float) (u+4*ib-Radius+2), (float) (v-Radius) );
		LocalBlock[SharedIdx+4*ib+3] = tex2D( texSobel, 
			(float) (u+4*ib-Radius+3), (float) (v-Radius) );
	}
	if ( threadIdx.y < Radius*2 ) {
		//
		// copy trailing Radius*2 rows of pixels into shared
		//
		SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;
		for ( ib = threadIdx.x; ib < BlockWidth+2*Radius; ib += blockDim.x ) {
			LocalBlock[SharedIdx+4*ib+0] = tex2D( texSobel, 
				(float) (u+4*ib-Radius+0), (float) (v+blockDim.y-Radius) );
			LocalBlock[SharedIdx+4*ib+1] = tex2D( texSobel, 
				(float) (u+4*ib-Radius+1), (float) (v+blockDim.y-Radius) );
			LocalBlock[SharedIdx+4*ib+2] = tex2D( texSobel, 
				(float) (u+4*ib-Radius+2), (float) (v+blockDim.y-Radius) );
			LocalBlock[SharedIdx+4*ib+3] = tex2D( texSobel, 
				(float) (u+4*ib-Radius+3), (float) (v+blockDim.y-Radius) );
		}
	}

	__syncthreads();

	u >>= 2;    // index as uchar4 from here
	uchar4 *pSobel = (uchar4 *) (((char *) pSobelOriginal)+v*SobelPitch);
	SharedIdx = threadIdx.y * SharedPitch;

	for ( ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x ) {

		unsigned char pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+0];
		unsigned char pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+1];
		unsigned char pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+2];
		unsigned char pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+0];
		unsigned char pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+1];
		unsigned char pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+2];
		unsigned char pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+0];
		unsigned char pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+1];
		unsigned char pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+2];

		uchar4 out;

		out.x = ComputeSobel(pix00, pix01, pix02, 
			pix10, pix11, pix12, 
			pix20, pix21, pix22, fScale );

		pix00 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+3];
		pix10 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+3];
		pix20 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+3];
		out.y = ComputeSobel(pix01, pix02, pix00, 
			pix11, pix12, pix10, 
			pix21, pix22, pix20, fScale );

		pix01 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+4];
		pix11 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+4];
		pix21 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+4];
		out.z = ComputeSobel( pix02, pix00, pix01, 
			pix12, pix10, pix11, 
			pix22, pix20, pix21, fScale );

		pix02 = LocalBlock[SharedIdx+4*ib+0*SharedPitch+5];
		pix12 = LocalBlock[SharedIdx+4*ib+1*SharedPitch+5];
		pix22 = LocalBlock[SharedIdx+4*ib+2*SharedPitch+5];
		out.w = ComputeSobel( pix00, pix01, pix02, 
			pix10, pix11, pix12, 
			pix20, pix21, pix22, fScale );
		if ( u+ib < w/4 && v < h ) {
			pSobel[u+ib] = out;
		}
	}

	__syncthreads();
}

__GCU_FCT_GLOBAL void 
SobelTexKernel( unsigned char *pSobelOriginal, unsigned int Pitch, 
			   int w, int h, float fScale )
{ 
	unsigned char *pSobel = 
		(unsigned char *) (((char *) pSobelOriginal)+blockIdx.x*Pitch);
	for ( int i = threadIdx.x; i < w; i += blockDim.x ) {
		unsigned char pix00 = tex2D( texSobel, (float) i-1, (float) blockIdx.x-1 );
		unsigned char pix01 = tex2D( texSobel, (float) i+0, (float) blockIdx.x-1 );
		unsigned char pix02 = tex2D( texSobel, (float) i+1, (float) blockIdx.x-1 );
		unsigned char pix10 = tex2D( texSobel, (float) i-1, (float) blockIdx.x+0 );
		unsigned char pix11 = tex2D( texSobel, (float) i+0, (float) blockIdx.x+0 );
		unsigned char pix12 = tex2D( texSobel, (float) i+1, (float) blockIdx.x+0 );
		unsigned char pix20 = tex2D( texSobel, (float) i-1, (float) blockIdx.x+1 );
		unsigned char pix21 = tex2D( texSobel, (float) i+0, (float) blockIdx.x+1 );
		unsigned char pix22 = tex2D( texSobel, (float) i+1, (float) blockIdx.x+1 );
		pSobel[i] = ComputeSobel(pix00, pix01, pix02, 
			pix10, pix11, pix12,
			pix20, pix21, pix22, fScale );
	}
}

__GCU_FCT_DEVICE unsigned char
ComputeErode(unsigned char ul, // upper left
			 unsigned char um, // upper middle
			 unsigned char ur, // upper right
			 unsigned char ml, // middle left
			 unsigned char mm, // middle (unused)
			 unsigned char mr, // middle right
			 unsigned char ll, // lower left
			 unsigned char lm, // lower middle
			 unsigned char lr, // lower right
			 float fScale )
{
	unsigned char Result=	min(ul,
		min(um,
		min(ur,
		min(ml,
		min(mm,
		min(mr,
		min(ll,
		min(lm,lr))))))));
	return Result;
}

__GCU_FCT_GLOBAL void 
ErodeTexKernel( unsigned char *pSobelOriginal, unsigned int Pitch, 
			   int w, int h, float fScale )
{ 
	unsigned char *pSobel = 
		(unsigned char *) (((char *) pSobelOriginal)+blockIdx.x*Pitch);
	for ( int i = threadIdx.x; i < w; i += blockDim.x ) {
		unsigned char pix00 = tex2D( texSobel, (float) i-1, (float) blockIdx.x-1 );
		unsigned char pix01 = tex2D( texSobel, (float) i+0, (float) blockIdx.x-1 );
		unsigned char pix02 = tex2D( texSobel, (float) i+1, (float) blockIdx.x-1 );
		unsigned char pix10 = tex2D( texSobel, (float) i-1, (float) blockIdx.x+0 );
		unsigned char pix11 = tex2D( texSobel, (float) i+0, (float) blockIdx.x+0 );
		unsigned char pix12 = tex2D( texSobel, (float) i+1, (float) blockIdx.x+0 );
		unsigned char pix20 = tex2D( texSobel, (float) i-1, (float) blockIdx.x+1 );
		unsigned char pix21 = tex2D( texSobel, (float) i+0, (float) blockIdx.x+1 );
		unsigned char pix22 = tex2D( texSobel, (float) i+1, (float) blockIdx.x+1 );
		pSobel[i] = ComputeErode(pix00, pix01, pix02, 
			pix10, pix11, pix12,
			pix20, pix21, pix22, fScale );
	}
}


#endif
//===============================================================
//
//Dilate
//
//===============================================================
//extern __shared__ uchar4 LocalBlockDilate[];

//template <typename TPLData,bool USE_CUSTOM_ELEM> 
//#define FCT_PTR_COMP_DILATE(NAME, TYPE) TYPE(*NAME)(TYPE,TYPE,TYPE,TYPE,TYPE,TYPE,TYPE,TYPE,TYPE,unsigned char *)
//#define FCT_PTR_COMP_DILATE_TPL(NAME, TPL1, TPL2, TYPE) (TYPE)*NAME<TPL1,TPL2>(TYPE,TYPE,TYPE,TYPE,TYPE,TYPE,TYPE,TYPE,TYPE,unsigned char *)


//ConvolutionMacro
extern __shared__ unsigned char LocalBlockDilate[];
#define	__CONV_KERNEL_INIT_PARAMS(RADIUS)							\
	short u = TPLElemSize*blockIdx.x*BlockWidth;				\
	short v = blockIdx.y*blockDim.y + threadIdx.y;				\
	short ib;													\
	int SharedIdx = threadIdx.y * SharedPitch;					\
	unsigned char LocalParams[RADIUS*RADIUS];					\
	uchar4 out;

#define	__CONV_KERNEL_INIT_DEBUG(NAME)										\
{																	\
	GCUDA_KRNL_DBG_FIRST_THREAD(NAME,								\
	printf("- Width:\t %d\n", w);								\
	printf("- Height:\t %d\n", h);								\
	printf("- BlockWidth:\t %d\n", BlockWidth);					\
	printf("- SharedPitch:\t %d\n", SharedPitch);				\
	printf("- template<%s, %d>\n", "?", TPLElemSize);			\
	if(Params && ParamsNbr)										\
{															\
	printf("- element val: %d %d %d\n", Params[0], Params[1], Params[2]);\
	printf("- element val: %d %d %d\n", Params[3], Params[4], Params[5]);\
	printf("- element val: %d %d %d\n", Params[6], Params[7], Params[8]);\
}															\
	);																\
	GCUDA_KRNL_DBG_SHOW_THREAD_POS();								\
}

// copy trailing Radius*2 rows of pixels into shared
// index as uchar4 from here
#define	__CONV_KERNEL_LOAD_SHARED_DATA(RADIUS)																				\
	for ( ib = threadIdx.x; ib < BlockWidth+2*RADIUS; ib += blockDim.x ) {												\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+0] = d_src[(u+TPLElemSize*ib-RADIUS+0)+(v-RADIUS)*w];					\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+1] = d_src[(u+TPLElemSize*ib-RADIUS+1)+(v-RADIUS)*w];					\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+2] = d_src[(u+TPLElemSize*ib-RADIUS+2)+(v-RADIUS)*w];					\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+3] = d_src[(u+TPLElemSize*ib-RADIUS+3)+(v-RADIUS)*w];					\
	}	\
	if ( threadIdx.y < RADIUS*2 ) {																						\
	SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;																\
	for ( ib = threadIdx.x; ib < BlockWidth+2*(RADIUS); ib += blockDim.x ) {\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+0] = d_src[(u+TPLElemSize*ib-RADIUS+0)+(v+blockDim.y-RADIUS)*w];	\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+1] = d_src[(u+TPLElemSize*ib-RADIUS+1)+(v+blockDim.y-RADIUS)*w];	\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+2] = d_src[(u+TPLElemSize*ib-RADIUS+2)+(v+blockDim.y-RADIUS)*w];	\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+3] = d_src[(u+TPLElemSize*ib-RADIUS+3)+(v+blockDim.y-RADIUS)*w];	\
	}																												\
	}	\
	__syncthreads();																									\
	u >>= 2;																											\
	uchar4 *pSobel = (uchar4 *) (((char *) pSobelOriginal)+v*SobelPitch);												\
	SharedIdx = threadIdx.y * SharedPitch;

/*
if ( threadIdx.y < RADIUS*2 ) {																						\
SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;																\
for ( ib = threadIdx.x; ib < BlockWidth+2*(RADIUS); ib += blockDim.x ) {\
if(v+blockDim.y-RADIUS<h)\
{\
LocalBlockDilate[SharedIdx+TPLElemSize*ib+0] = d_src[(u+TPLElemSize*ib-RADIUS+0)+(v+blockDim.y-RADIUS)*w];	\
LocalBlockDilate[SharedIdx+TPLElemSize*ib+1] = d_src[(u+TPLElemSize*ib-RADIUS+1)+(v+blockDim.y-RADIUS)*w];	\
LocalBlockDilate[SharedIdx+TPLElemSize*ib+2] = d_src[(u+TPLElemSize*ib-RADIUS+2)+(v+blockDim.y-RADIUS)*w];	\
LocalBlockDilate[SharedIdx+TPLElemSize*ib+3] = d_src[(u+TPLElemSize*ib-RADIUS+3)+(v+blockDim.y-RADIUS)*w];	\
}\
}																												\
}																													\
__syncthreads();																									\
u >>= 2;																											\
uchar4 *pSobel = (uchar4 *) (((char *) pSobelOriginal)+v*SobelPitch);												\
SharedIdx = threadIdx.y * SharedPitch;
*/
#define	__CONV_KERNEL_GET_ELEMENTS(DST,SRC, NBR)\
	for(int i =0; i < NBR; i++)\
	DST[i] = SRC[i];

// Maco used into the compute loop to create/read variables


//! \brief Create a variable of type VAR_TYPE with name "BASENAME##X" and size is 3. Copy values from SRC and write them into the created variable, position in SRC is SharedIdx+TPLElemSize*ib+posY*SharedPitch+X, with posY:[0..2]
/*
#define __CONV_KERNEL_CREATE_ALL_TEMP_COL_3(VAR_TYPE, BASENAME)\
VAR_TYPE BASENAME##0[3];\
VAR_TYPE BASENAME##1[3];\
VAR_TYPE BASENAME##2[3];

#define __CONV_KERNEL_CREATE_ALL_TEMP_COL_5(VAR_TYPE, BASENAME)\
VAR_TYPE BASENAME##0[5];\
VAR_TYPE BASENAME##1[5];\
VAR_TYPE BASENAME##2[5];\
VAR_TYPE BASENAME##3[5];\
VAR_TYPE BASENAME##4[5];

#define __CONV_KERNEL_CREATE_ALL_TEMP_COL_7(VAR_TYPE, BASENAME)\
VAR_TYPE BASENAME##0[7];\
VAR_TYPE BASENAME##1[7];\
VAR_TYPE BASENAME##2[7];\
VAR_TYPE BASENAME##3[7];\
VAR_TYPE BASENAME##4[7];\
VAR_TYPE BASENAME##5[7];
VAR_TYPE BASENAME##6[7];
*/
//! \brief Create 3 variables of type VAR_TYPE with name "BASENAME##X" and size is 3. Copy values from SRC and write them into the created variable, position in SRC is SharedIdx+TPLElemSize*ib+posY*SharedPitch+posX, with posX:[0..2]
#define __CONV_KERNEL_CREATE_X_VAR_3(VAR_TYPE, BASENAME, X)		\
	VAR_TYPE BASENAME##X[3] = {							\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+0*SharedPitch+X],\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+1*SharedPitch+X],\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+2*SharedPitch+X]};\
	BASENAME[X] = BASENAME##X;

//! \brief Create a variable of type VAR_TYPE, name is "BASENAME##Col##X" and size is 5. Copy values from SRC and write them into the created variable, position in SRC is SharedIdx+TPLElemSize*ib+pos*SharedPitch+X, with pos:[0..4]
#define __CONV_KERNEL_CREATE_X_VAR_5(VAR_TYPE, BASENAME, X)		\
	VAR_TYPE BASENAME##X[5] = {							\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+0*SharedPitch+X],\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+1*SharedPitch+X],\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+2*SharedPitch+X],\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+3*SharedPitch+X],\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+4*SharedPitch+X]};\
	BASENAME[X] = BASENAME##X;

//! \brief Create a variable of type VAR_TYPE, name is "BASENAME##X" and size is 7. Copy values from SRC and write them into the created variable, position in SRC is SharedIdx+TPLElemSize*ib+pos*SharedPitch+X, with pos:[0..6]
#define __CONV_KERNEL_CREATE_X_VAR_7(VAR_TYPE, BASENAME, X)		\
	VAR_TYPE BASENAME##X[7] = {							\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+0*SharedPitch+X],\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+1*SharedPitch+X],\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+2*SharedPitch+X],\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+3*SharedPitch+X],\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+4*SharedPitch+X],\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+5*SharedPitch+X],\
	LocalBlockDilate[SharedIdx+TPLElemSize*ib+6*SharedPitch+X]};\
	BASENAME[X] = BASENAME##X;



#define __CONV_KERNEL_CREATE_XY_VAR_3(VAR_TYPE, BASENAME)		\
	__CONV_KERNEL_CREATE_X_VAR_3(VAR_TYPE,BASENAME, 0);	\
	__CONV_KERNEL_CREATE_X_VAR_3(VAR_TYPE,BASENAME, 1);	\
	__CONV_KERNEL_CREATE_X_VAR_3(VAR_TYPE,BASENAME, 2);



//! \brief Create 5 variables of type VAR_TYPE with name "BASENAME##X" and size is 5. Copy values from SRC and write them into the created variable, position in SRC is SharedIdx+TPLElemSize*ib+posY*SharedPitch+posX, with posX:[0..4]
#define __CONV_KERNEL_CREATE_XY_VAR_5(VAR_TYPE, BASENAME)		\
	__CONV_KERNEL_CREATE_X_VAR_5(VAR_TYPE,BASENAME, 0);	\
	__CONV_KERNEL_CREATE_X_VAR_5(VAR_TYPE,BASENAME, 1); \
	__CONV_KERNEL_CREATE_X_VAR_5(VAR_TYPE,BASENAME, 2); \
	__CONV_KERNEL_CREATE_X_VAR_5(VAR_TYPE,BASENAME, 3); \
	__CONV_KERNEL_CREATE_X_VAR_5(VAR_TYPE,BASENAME, 4);

//! \brief Create 7 variables of type VAR_TYPE with name "BASENAME##Col##X" and size is 7. Copy values from SRC and write them into the created variable, position in SRC is SharedIdx+TPLElemSize*ib+posY*SharedPitch+posX, with posX:[0..6]
#define __CONV_KERNEL_CREATE_XY_VAR_7(VAR_TYPE, BASENAME)		\
	__CONV_KERNEL_CREATE_X_VAR_7(VAR_TYPE,BASENAME, 0);	\
	__CONV_KERNEL_CREATE_X_VAR_7(VAR_TYPE,BASENAME, 1); \
	__CONV_KERNEL_CREATE_X_VAR_7(VAR_TYPE,BASENAME, 2); \
	__CONV_KERNEL_CREATE_X_VAR_7(VAR_TYPE,BASENAME, 3); \
	__CONV_KERNEL_CREATE_X_VAR_7(VAR_TYPE,BASENAME, 4); \
	__CONV_KERNEL_CREATE_X_VAR_7(VAR_TYPE,BASENAME, 5); \
	__CONV_KERNEL_CREATE_X_VAR_7(VAR_TYPE,BASENAME, 6);


//! \brief Get 3 values from SRC and write them into DST[3], position in SRC is OFFSET+SHAREDPITCH*x, with x:[0..2]
#define __CONV_GETVAL_3(SRC, DST, OFFSET, SHAREDPITCH)			\
	DST[0] = SRC[OFFSET+SHAREDPITCH*0];					\
	DST[1] = SRC[OFFSET+SHAREDPITCH*1];					\
	DST[2] = SRC[OFFSET+SHAREDPITCH*2];


//! \brief Get 5 values from SRC and write them into DST[5], position in SRC is OFFSET+SHAREDPITCH*x, with x:[0..4]
#define __CONV_GETVAL_5(SRC, DST, OFFSET, SHAREDPITCH)		\
	DST[0] = SRC[OFFSET+SHAREDPITCH*0];					\
	DST[1] = SRC[OFFSET+SHAREDPITCH*1];					\
	DST[2] = SRC[OFFSET+SHAREDPITCH*2];					\
	DST[3] = SRC[OFFSET+SHAREDPITCH*3];					\
	DST[4] = SRC[OFFSET+SHAREDPITCH*4];

//! \brief Get 7 values from SRC and write them into DST[7], position in SRC is OFFSET+SHAREDPITCH*x, with x:[0..6]
#define __CONV_GETVAL_7(SRC, DST, OFFSET, SHAREDPITCH)		\
	DST[0] = SRC[OFFSET+SHAREDPITCH*0];					\
	DST[1] = SRC[OFFSET+SHAREDPITCH*1];					\
	DST[2] = SRC[OFFSET+SHAREDPITCH*2];					\
	DST[3] = SRC[OFFSET+SHAREDPITCH*3];					\
	DST[4] = SRC[OFFSET+SHAREDPITCH*4];					\
	DST[5] = SRC[OFFSET+SHAREDPITCH*5];					\
	DST[6] = SRC[OFFSET+SHAREDPITCH*6];



#define __CONV_KERNEL_SWITCH_COL_LEFT_3(VAR_TYPE,BASENAME)\
	SwitchCol=BASENAME[0];\
	BASENAME[0] = BASENAME[1];\
	BASENAME[1] = BASENAME[2];\
	BASENAME[2] = SwitchCol;

#define __CONV_KERNEL_SWITCH_COL_LEFT_5(VAR_TYPE,BASENAME)\
	SwitchCol=BASENAME[0];\
	BASENAME[0] = BASENAME[1];\
	BASENAME[1] = BASENAME[2];\
	BASENAME[2] = BASENAME[3];\
	BASENAME[3] = BASENAME[4];\
	BASENAME[4] = SwitchCol;

#define __CONV_KERNEL_SWITCH_COL_LEFT_7(VAR_TYPE,BASENAME)\
	SwitchCol=BASENAME[0];\
	BASENAME[0] = BASENAME[1];\
	BASENAME[1] = BASENAME[2];\
	BASENAME[2] = BASENAME[3];\
	BASENAME[3] = BASENAME[4];\
	BASENAME[4] = BASENAME[5];\
	BASENAME[5] = BASENAME[6];\
	BASENAME[6] = SwitchCol;



//TPLData pix00 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+0*SharedPitch+0];						\
TPLData pix01 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+0*SharedPitch+1];						\
TPLData pix02 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+0*SharedPitch+2];						\
TPLData pix10 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+1*SharedPitch+0];						\
TPLData pix11 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+1*SharedPitch+1];						\
TPLData pix12 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+1*SharedPitch+2];						\
TPLData pix20 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+2*SharedPitch+0];						\
TPLData pix21 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+2*SharedPitch+1];						\
TPLData pix22 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+2*SharedPitch+2];		


//<TPLData,USE_CUSTOM_ELEM,0,0,Xorder,Yorder>
#define __CONV_KERNEL_COMPUTE(RADIUS)																		\
	int Coloffset = Xorder;																				\
	TPLData *SwitchCol=NULL;																			\
	TPLData *pixCol[Xorder];																			\
	for ( ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x )											\
{																									\
	__CONV_KERNEL_CREATE_XY_VAR_##RADIUS(TPLData, pixCol);											\
	KernelFunction::Do(&out.x, pixCol, LocalParams,fScale);											\
	Coloffset = Xorder;																				\
	__CONV_GETVAL_##RADIUS(LocalBlockDilate, pixCol[0], SharedIdx+TPLElemSize*ib+Coloffset, SharedPitch);\
	__CONV_KERNEL_SWITCH_COL_LEFT_##RADIUS(TPLData, pixCol);										\
	KernelFunction::Do(&out.y, pixCol, LocalParams,fScale);											\
	Coloffset++;																					\
	__CONV_GETVAL_##RADIUS(LocalBlockDilate, pixCol[0], SharedIdx+TPLElemSize*ib+Coloffset, SharedPitch);\
	__CONV_KERNEL_SWITCH_COL_LEFT_##RADIUS(TPLData, pixCol);										\
	KernelFunction::Do(&out.z, pixCol, LocalParams,fScale);											\
	Coloffset++;																					\
	__CONV_GETVAL_##RADIUS(LocalBlockDilate, pixCol[0], SharedIdx+TPLElemSize*ib+Coloffset, SharedPitch);\
	__CONV_KERNEL_SWITCH_COL_LEFT_##RADIUS(TPLData, pixCol);										\
	KernelFunction::Do(&out.w, pixCol, LocalParams,fScale);\
	Coloffset++;																					\
	if ( u+ib < w/TPLElemSize && v < h )															\
{																								\
	pSobel[u+ib]	= out;																		\
}																								\
}																										\
	__syncthreads();


//

//typedef template <typename TPLData, bool TPLCutomKernel, int BeginX, int BeginY, int EndX, int EndY> KernelStructType;

#define __CUDA_CONV_KERNEL_MACRO(RADIUS)\
	template	<typename TPLData,														\
	unsigned int TPLElemSize,												\
	bool USE_CUSTOM_ELEM,		\
	int Xorder,					\
	int Yorder,					\
	typename KernelFunction	\
	>\
	__GCU_FCT_GLOBAL void																		\
	CudaConvKernel##RADIUS(TPLData *d_src, uchar4 *pSobelOriginal, unsigned short SobelPitch,				\
	short BlockWidth, short SharedPitch,														\
	short w, short h, float fScale														\
	,unsigned int ParamsNbr, unsigned char *Params)											\
{																										\
	__CONV_KERNEL_INIT_PARAMS(Xorder);																	\
	__CONV_KERNEL_INIT_DEBUG("CudaConvKernel");															\
	__CONV_KERNEL_LOAD_SHARED_DATA(((RADIUS-1)/2));														\
	if (USE_CUSTOM_ELEM)																				\
{__CONV_KERNEL_GET_ELEMENTS(LocalParams,Params, ParamsNbr);}													\
	__CONV_KERNEL_COMPUTE(RADIUS);																			\
	GCUDA_KRNL_DBG_LAST_THREAD("CudaConvKernel",int a =0;);												\
}


template <typename TPLData, bool TPLCutomKernel>
__GCU_FCT_DEVICE 
TPLData 
ConvKernelCompute_LaplaceCustom(TPLData ul, // upper left
								TPLData um, // upper middle
								TPLData ur, // upper right
								TPLData ml, // middle left
								TPLData mm, // middle (unused)
								TPLData mr, // middle right
								TPLData ll, // lower left
								TPLData lm, // lower middle
								TPLData lr, // lower right
								float fScale,
								unsigned char *Params
								)
{
	/*Custom Laplace kernel elements:
	*/
	TPLData Val = ul*Params[0];
	Val += um*Params[1];	
	Val += ur*Params[2];
	Val += ml*Params[3];
	Val += mm*Params[4];
	Val += mr*Params[5];
	Val += ll*Params[6];
	Val += lm*Params[7];
	Val += lr*Params[8];
	/*	Val*=fScale;
	if(Val>255)
	Val = 255;
	*/  return Val;
}

__CUDA_CONV_KERNEL_MACRO(3);
__CUDA_CONV_KERNEL_MACRO(5);
//__CUDA_CONV_KERNEL_MACRO(7);


#if 0
template <typename TPLData, 
unsigned int TPLElemSize, 
bool USE_CUSTOM_ELEM>
__GCU_FCT_GLOBAL void 
ErodeSharedKernel(TPLData *d_src, uchar4 *pSobelOriginal, unsigned short SobelPitch, 
				  short BlockWidth, short SharedPitch,
				  short w, short h
				  ,unsigned int ParamsNbr, unsigned char *Params)
{ 
	short u = TPLElemSize*blockIdx.x*BlockWidth;
	short v = blockIdx.y*blockDim.y + threadIdx.y;
	short ib;

	int SharedIdx = threadIdx.y * SharedPitch;

	//unsigned char * __LocalBlock = (unsigned char *)LocalBlockDilate;

	GCUDA_KRNL_DBG_FIRST_THREAD("ErodeSharedKernel",
		printf("- Width:\t %d\n", w);
	printf("- Height:\t %d\n", h);
	printf("- BlockWidth:\t %d\n", BlockWidth);
	printf("- SharedPitch:\t %d\n", SharedPitch);
	printf("- template<%s, %d>\n", "?", TPLElemSize);

	if(Params && ParamsNbr)
	{
		printf("- element val: %d %d %d\n", Params[0], Params[1], Params[2]);
		printf("- element val: %d %d %d\n", Params[3], Params[4], Params[5]);
		printf("- element val: %d %d %d\n", Params[6], Params[7], Params[8]);
	}
	);

	GCUDA_KRNL_DBG_SHOW_THREAD_POS();
	for ( ib = threadIdx.x; ib < BlockWidth+2*Radius; ib += blockDim.x ) {
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+0] = d_src[(u+TPLElemSize*ib-Radius+0)+(v-Radius)*w];
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+1] = d_src[(u+TPLElemSize*ib-Radius+1)+(v-Radius)*w];
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+2] = d_src[(u+TPLElemSize*ib-Radius+2)+(v-Radius)*w];
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+3] = d_src[(u+TPLElemSize*ib-Radius+3)+(v-Radius)*w];

		/*
		GCUDA_KRNL_DBG(
		printf("%d %d %d %d\n", LocalBlockDilate[SharedIdx+TPLElemSize*ib+0].x,
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+0].y,
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+0].z,
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+0].w);
		printf("%d %d %d %d\n", LocalBlockDilate[SharedIdx+TPLElemSize*ib+1].x,
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+1].y,
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+1].z,
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+1].w);
		printf("%d %d %d %d\n", LocalBlockDilate[SharedIdx+TPLElemSize*ib+2].x,
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+2].y,
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+2].z,
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+2].w);
		printf("%d %d %d %d\n", LocalBlockDilate[SharedIdx+TPLElemSize*ib+3].x,
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+3].y,
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+3].z,
		LocalBlockDilate[SharedIdx+TPLElemSize*ib+3].w);
		);
		*/
	}
	if ( threadIdx.y < Radius*2 ) {
		//
		// copy trailing Radius*2 rows of pixels into shared
		//
		SharedIdx = (blockDim.y+threadIdx.y) * SharedPitch;
		for ( ib = threadIdx.x; ib < BlockWidth+2*Radius; ib += blockDim.x ) {
			LocalBlockDilate[SharedIdx+TPLElemSize*ib+0] = d_src[(u+TPLElemSize*ib-Radius+0)+(v+blockDim.y-Radius)*w];
			LocalBlockDilate[SharedIdx+TPLElemSize*ib+1] = d_src[(u+TPLElemSize*ib-Radius+1)+(v+blockDim.y-Radius)*w];
			LocalBlockDilate[SharedIdx+TPLElemSize*ib+2] = d_src[(u+TPLElemSize*ib-Radius+2)+(v+blockDim.y-Radius)*w];
			LocalBlockDilate[SharedIdx+TPLElemSize*ib+3] = d_src[(u+TPLElemSize*ib-Radius+3)+(v+blockDim.y-Radius)*w];
		}
	}

	__syncthreads();

	u >>= 2;    // index as uchar4 from here
	uchar4 *pSobel = (uchar4 *) (((char *) pSobelOriginal)+v*SobelPitch);
	SharedIdx = threadIdx.y * SharedPitch;

	unsigned char LocalParams[9];
#if USE_CUSTOM_ELEM
	LocalParams[0] = Params[0];
	LocalParams[1] = Params[1];
	LocalParams[2] = Params[2];
	LocalParams[3] = Params[3];
	LocalParams[4] = Params[4];
	LocalParams[5] = Params[5];
	LocalParams[6] = Params[6];
	LocalParams[7] = Params[7];
	LocalParams[8] = Params[8];
#endif

	for ( ib = threadIdx.x; ib < BlockWidth; ib += blockDim.x ) {

		TPLData pix00 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+0*SharedPitch+0];
		TPLData pix01 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+0*SharedPitch+1];
		TPLData pix02 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+0*SharedPitch+2];
		TPLData pix10 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+1*SharedPitch+0];
		TPLData pix11 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+1*SharedPitch+1];
		TPLData pix12 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+1*SharedPitch+2];
		TPLData pix20 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+2*SharedPitch+0];
		TPLData pix21 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+2*SharedPitch+1];
		TPLData pix22 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+2*SharedPitch+2];

		uchar4 out;

		out.x = ComputeErode<unsigned char,USE_CUSTOM_ELEM>(pix00, pix01, pix02, 
			pix10, pix11, pix12, 
			pix20, pix21, pix22,LocalParams);

		pix00 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+0*SharedPitch+3];
		pix10 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+1*SharedPitch+3];
		pix20 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+2*SharedPitch+3];
		out.y = ComputeErode<unsigned char,USE_CUSTOM_ELEM>(pix01, pix02, pix00, 
			pix11, pix12, pix10, 
			pix21, pix22, pix20,LocalParams);

		pix01 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+0*SharedPitch+4];
		pix11 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+1*SharedPitch+4];
		pix21 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+2*SharedPitch+4];
		out.z = ComputeErode<unsigned char,USE_CUSTOM_ELEM>( pix02, pix00, pix01, 
			pix12, pix10, pix11, 
			pix22, pix20, pix21,LocalParams);

		pix02 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+0*SharedPitch+5];
		pix12 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+1*SharedPitch+5];
		pix22 = LocalBlockDilate[SharedIdx+TPLElemSize*ib+2*SharedPitch+5];
		out.w = ComputeErode<unsigned char,USE_CUSTOM_ELEM>( pix00, pix01, pix02, 
			pix10, pix11, pix12, 
			pix20, pix21, pix22,LocalParams);
		if ( u+ib < w/TPLElemSize && v < h ) {
			pSobel[u+ib]	= out;
		}
	}
	__syncthreads();
	GCUDA_KRNL_DBG_LAST_THREAD("ErodeSharedKernel",int a =0;);
}
#endif

#endif//__GPUCV_CUDA_BASE_CONVOL_KERNEL_H
