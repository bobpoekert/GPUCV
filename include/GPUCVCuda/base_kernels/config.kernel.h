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
/**
*	\brief Configuration of the kernel settings, enum and defines...
*	\author Yannick Allusse
*/
#ifndef __GPUCV_CUDA_CONFIG_KERNEL_H
#define __GPUCV_CUDA_CONFIG_KERNEL_H


#ifdef _LINUX
	#define __GCU_STATIC_FLG static
	#define __GCU_INLINE_FLG inline
#else
	#define __GCU_STATIC_FLG static
	#define __GCU_INLINE_FLG inline
#endif
#define __GCU_FCT_DEVICE					__device__ __GCU_STATIC_FLG
#define __GCU_FCT_GLOBAL					__global__ __GCU_STATIC_FLG
/*static*/ //cause compilation errors with CUDA 2.0(crash compiler)
#define __GCU_FCT_ALLSIDE					__global__ __device__ __GCU_STATIC_FLG

#ifdef _MSC_VER
#define __GCU_FCT_DEVICE_INLINE		__GCU_INLINE_FLG __GCU_FCT_DEVICE
#define __GCU_FCT_GLOBAL_INLINE		__GCU_INLINE_FLG __GCU_FCT_GLOBAL
#define __GCU_FCT_ALLSIDE_INLINE	__GCU_INLINE_FLG __GCU_FCT_ALLSIDE
#else
#define __GCU_FCT_DEVICE_INLINE		__GCU_FCT_DEVICE
#define __GCU_FCT_GLOBAL_INLINE		__GCU_FCT_GLOBAL
#define __GCU_FCT_ALLSIDE_INLINE	__GCU_FCT_ALLSIDE
#endif


typedef unsigned char	uchar;
typedef unsigned int	uint;
typedef unsigned short	ushort;


#define IS_MULTIPLE_OF(VAL, DIV)((VAL)%(DIV)==0)
#define IS_INTEGER(VAL)((float)(VAL)-(int)(VAL)==0)

#define SWITCH_VAL(TYPE, VAL1, VAL2){\
	TYPE tmpVal = VAL1;\
	VAL1 = VAL2;\
	VAL2 = tmpVal;\
}

#define GCU_CPP_COMPLIANT 1

#ifndef _LINUX

#define CONVOLUTION_KERNEL_ERODE	0x01
#define CONVOLUTION_KERNEL_DILATE	0x02
#define CONVOLUTION_KERNEL_LAPLACE	0x04
#define CONVOLUTION_KERNEL_SOBEL	0x08
#define CONVOLUTION_KERNEL_ERODE_NOSHARE	0x10


struct GCU_KERNEL_SIZE_ST
{
	dim3 Threads;
	dim3 Blocks;
	dim3 BlockSize;
	int  SharedMem;
	uint SharedPitch;
	uint Pitch;
};
struct GCU_DATA_SIZE_ST
{
	dim3 Size;
	uchar Channels;

};

struct GCU_CONVOL_KERNEL_SETTINGS_ST
{
	GCU_KERNEL_SIZE_ST	KSize;
	GCU_DATA_SIZE_ST	DSize;
	uint				ParamsNbr;
	unsigned char*		Params;
};

struct GCU_CONV_Kernel//IplConvKernel
{
    int  nCols;
    int  nRows;
    int  anchorX;
    int  anchorY;
    int *values;
    int  nShiftR;
};
#endif

#ifndef CvArr
	//We must avoid at all cost to include OpenCV Stuff into CUDA parts, so we redefine some openCV structs here
	#define CvArr void

	typedef struct
	{
		double val[4];
	}gcuScalar; //to replace CvScalar
#endif

#endif
