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
#ifndef __GPUCV_CUDA_CONFIG_H
#define __GPUCV_CUDA_CONFIG_H

#ifdef _CPLUSPLUS
#include	<GPUCV/include.h>
#endif

#define		_GPUCV_COMPILE_CUDA 1

#if _GPUCV_COMPILE_CUDA

#define		_GPUCV_CUDA_USE_RUNTIME_API 1
#define		_GPUCV_CUDA_USE_DRIVER_API 0
#define		_GPUCV_CUDA_USE_SINGLE_CHANNEL_IMG 1//??? still needed???
#define		_GPUCV_CUDA_SUPPORT_OPENGL 1

#ifdef _WINDOWS
	#if defined(_GPUCV_CUDA_DLL) || defined (_NVCC)
		#define		_GPUCV_CUDA_EXPORT		__declspec(dllexport)
		#define		_GPUCV_CUDA_EXPORT_C	_GPUCV_CUDA_EXPORT//extern "C" _GPUCV_CUDA_EXPORT//
		#define		_GPUCV_CUDA_EXPORT_CU	_GPUCV_CUDA_EXPORT_C  //extern "C"
	#else
		#define		_GPUCV_CUDA_EXPORT		__declspec(dllimport)
		#define		_GPUCV_CUDA_EXPORT_C	_GPUCV_CUDA_EXPORT//extern "C" _GPUCV_CUDA_EXPORT
		#define		_GPUCV_CUDA_EXPORT_CU	_GPUCV_CUDA_EXPORT_C  //extern "C"
	#endif
#else
	#define		_GPUCV_CUDA_EXPORT
	#define		_GPUCV_CUDA_EXPORT_C
	#define		_GPUCV_CUDA_EXPORT_CU  extern "C"
#endif


#ifdef _MSC_VER
#define _GPUCV_CUDA_INLINE inline
#else
#define _GPUCV_CUDA_INLINE
#endif

#define GPUCV_DEBUG_CUDA GPUCV_DEBUG
//#define GPUCV_CUDA_SUPPORT_3CH_IMG 3

#ifdef _GPUCV_CUDA_DLL
#define _GPUCV_DISABLE_EXCEPTION 0
#define GCU_Assert(COND, MSG) SG_Assert(COND, MSG)
#else
#define _GPUCV_DISABLE_EXCEPTION 0
#define GCU_Assert(COND, MSG) SG_Assert(COND, MSG)
#endif

/*
// \brief Enable CUDA memory check, print out % of available memory before and after allocation/delete.
#if _DEBUG
#define _GCU_DEBUG_MEMORY_ALLOC 1
#else
#define _GCU_DEBUG_MEMORY_ALLOC 0
#endif
*/
#define GCU_CLASS_ASSERT(COND,MSG)\
	{\
	gcudaCheckError(MSG);\
	CLASS_ASSERT(COND,MSG);\
	}

//! Check for CUDA error
#ifdef _DEBUG
#  define GCU_OPER_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
	GCV_OPER_ASSERT(cudaSuccess == err, "Cuda error: '"<< errorMessage << "', msg: '" << cudaGetErrorString( err) <<"'");\
    err = cudaThreadSynchronize();                                           \
	GCV_OPER_ASSERT(cudaSuccess == err, "Cuda error: '"<< errorMessage << "', msg: '" << cudaGetErrorString( err) <<"'");\
    }
#else
#  define GCU_OPER_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
	GCV_OPER_ASSERT(cudaSuccess == err, "Cuda error: '"<< errorMessage << "', msg: '" << cudaGetErrorString( err) <<"'");\
    }
#endif

#endif //support CUDA
#endif //__GPUCV_CUDA_CONFIG_H
