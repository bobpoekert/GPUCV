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



/** \brief Includes all possible definitions of textures that can be used for any operators
*	\author Yannick Allusse
*/
#ifndef __GPUCV_CUDA_TEXTURES_KERNEL_H
#define __GPUCV_CUDA_TEXTURES_KERNEL_H
#include <cuda.h>
#include<GPUCVCuda/base_kernels/cuda_macro.kernel.cu>


#define GCU_USE_CHAR	1
#define GCU_USE_SHORT	1
#define GCU_USE_INT		1
#define GCU_USE_FLOAT	1
#define GCU_USE_DOUBLE	0



#define _CUDAG_GET_TEX_NM(PREFIXE, TYPE, DIM) tex##PREFIXE##_##TYPE##_##DIM

#define GPUCV_DECLARE_CU_TEXTURE(PREFIXE, TYPE, DIM, OPT)\
	static texture<TYPE, DIM,	OPT>	_CUDAG_GET_TEX_NM(PREFIXE, TYPE, DIM);

#define GPUCV_DECLARE_CU_TEXTURE_GRP(TYPE, DIM, OPT)\
	GPUCV_DECLARE_CU_TEXTURE(A, TYPE, DIM, OPT)\
	GPUCV_DECLARE_CU_TEXTURE(B, TYPE, DIM, OPT)\
	GPUCV_DECLARE_CU_TEXTURE(C, TYPE, DIM, OPT)\
	GPUCV_DECLARE_CU_TEXTURE(MASK, TYPE, DIM, OPT)\


#if 0
//declare all kind of textures:
#if GCU_USE_CHAR
//char
GPUCV_DECLARE_CU_TEXTURE_GRP (char1,	2,cudaReadModeElementType);
//GPUCV_DECLARE_CU_TEXTURE_GRP (char2,	2,cudaReadModeElementType);
GPUCV_DECLARE_CU_TEXTURE_GRP (char4,	2,cudaReadModeElementType);

//uchar
GPUCV_DECLARE_CU_TEXTURE_GRP (uchar1,	2,cudaReadModeElementType);
//GPUCV_DECLARE_CU_TEXTURE_GRP (uchar2,	2,cudaReadModeElementType);
GPUCV_DECLARE_CU_TEXTURE_GRP (uchar4,	2,cudaReadModeElementType);
#endif


#if GCU_USE_SHORT
//short
GPUCV_DECLARE_CU_TEXTURE_GRP (short1,	2,cudaReadModeElementType);
//GPUCV_DECLARE_CU_TEXTURE_GRP (short2,	2,cudaReadModeElementType);
GPUCV_DECLARE_CU_TEXTURE_GRP (short4,	2,cudaReadModeElementType);

//ushort
GPUCV_DECLARE_CU_TEXTURE_GRP (ushort1,	2,cudaReadModeElementType);
//GPUCV_DECLARE_CU_TEXTURE_GRP (ushort2,	2,cudaReadModeElementType);
GPUCV_DECLARE_CU_TEXTURE_GRP (ushort4,	2,cudaReadModeElementType);
#endif

#if GCU_USE_INT
//int
GPUCV_DECLARE_CU_TEXTURE_GRP (int1,		2,cudaReadModeElementType);
//GPUCV_DECLARE_CU_TEXTURE_GRP (int2,		2,cudaReadModeElementType);
GPUCV_DECLARE_CU_TEXTURE_GRP (int4,		2,cudaReadModeElementType);


//uint
GPUCV_DECLARE_CU_TEXTURE_GRP (uint1,	2,cudaReadModeElementType);
//GPUCV_DECLARE_CU_TEXTURE_GRP (uint2,	2,cudaReadModeElementType);
GPUCV_DECLARE_CU_TEXTURE_GRP (uint4,	2,cudaReadModeElementType);

#endif
#if GCU_USE_FLOAT
//float
GPUCV_DECLARE_CU_TEXTURE_GRP (float1,	2,cudaReadModeElementType);
//GPUCV_DECLARE_CU_TEXTURE_GRP (float2,	2,cudaReadModeElementType);
GPUCV_DECLARE_CU_TEXTURE_GRP (float4,	2,cudaReadModeElementType);
#endif
#endif//0
#endif //__GPUCV_CUDA_TEXTURES_KERNEL_H
