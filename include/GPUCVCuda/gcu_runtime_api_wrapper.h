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
/** \brief Contains CPP wrapping functions for CUDA and CU C functions.
*	All CUDA function must be called within a '.cu' file to be compiled by cuda compiler.
*	'.cu' files do not support CPP yet(Cuda 1.1).
*	\author Yannick Allusse
*/
#ifndef __GPUCV_gcu_runtime_api_wrapper_H
#define __GPUCV_gcu_runtime_api_wrapper_H
#include <GPUCVCuda/config.h>


#if _GPUCV_COMPILE_CUDA

#if _GPUCV_CUDA_USE_RUNTIME_API
	#include <cuda_runtime_api.h>
#endif

#include <cuda.h>
#include <cutil.h>

//Round a / b to nearest higher integer value
//#define iDivUp(a, b)(a % b != 0) ? (a / b + 1) : (a / b)
//Round a / b to nearest lower integer value
//#define iDivDown(a, b)(a / b)
//Align a to nearest higher multiple of b
//#define iAlignUp(a, b)((a % b != 0) ?  (a - a % b + b) : a)
//Align a to nearest lower multiple of b
//#define iAlignDown(a, b) (a - a % b)


#include <GPUCVHardware/gcvGL.h>
#ifdef _GPUCV_CUDA_SUPPORT_OPENGL
#ifdef __cplusplus
#if CUDA_VERSION > 2300
#else
	//deprecated since version 3.0
	_GPUCV_CUDA_EXPORT_CU void gcudaGLRegisterBufferObject	(GLuint _bufferObj);
	_GPUCV_CUDA_EXPORT_CU void gcudaGLUnregisterBufferObject(GLuint _bufferObj);
	_GPUCV_CUDA_EXPORT_CU void gcudaGLMapBufferObject	(void** _devPtr, GLuint _bufferObj);
	_GPUCV_CUDA_EXPORT_CU void gcudaGLUnmapBufferObject	(GLuint _bufferObj);
#endif
#endif
#endif

#ifdef _GPUCV_CUDA_SUPPORT_DIRECTX
	#error ("GPUCV CUDA support for DirectX is not done...")
#endif

//we redefine CUT_SAFE_CALL which is exiting application to a softer version
#if 0
#  define CUDA_SAFE_CALL( call) do {                                         \
	CUDA_SAFE_CALL_NO_SYNC(call);                                            \
	cudaError err = cudaThreadSynchronize();                                 \
	if( cudaSuccess != err) {                                                \
	fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
	__FILE__, __LINE__, cudaGetErrorString( err) );              \
	exit(EXIT_FAILURE);                                                  \
	} } while (0)
#endif


#if 0 //this macro generate memory leaks.????
#define GCU_CUDA_SAFE_CALL( call) CUDA_SAFE_CALL_NO_SYNC(call); 
#else
//CUDA_SAFE_CALL_NO_SYNC(
#define GCU_CUDA_SAFE_CALL( call) do {               \
	cudaError err = call;         \
	if( cudaSuccess != err) {                        \
	fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
	__FILE__, __LINE__, cudaGetErrorString( err) );              \
	getchar();																\
	} } while (0)
	
	//exit(EXIT_FAILURE);                                                 
	
#endif
#if 0
#  define CUT_SAFE_CALL( call)                                               \
	if( CUTTrue != call) {                                                   \
	fprintf(stderr, "Cut error in file '%s' in line %i.\n",              \
	__FILE__, __LINE__);                                         \
	exit(EXIT_FAILURE);                                                  \
	}
#endif
#define GCU_SAFE_CALL(call)														\
	if( CUTTrue != call) {														\
	fprintf(stderr, "Cut error in file '%s' in line %i.\n",					\
	__FILE__, __LINE__);											\
	fprintf(stderr, "Press a key to exit application.\n");					\
	getchar();																\
	exit(EXIT_FAILURE);														\
	}

#include <stdio.h>


	//________________________________________
	//
	/** \brief Round a / b to nearest higher integer value
	*/
_GPUCV_CUDA_EXPORT_CU int iDivUp(int a, int b);
_GPUCV_CUDA_EXPORT_CU void gcudaMalloc(void ** _buffer, unsigned int _size);
_GPUCV_CUDA_EXPORT_CU void gcudaMallocPitch(void ** _buffer, size_t* pitch,size_t widthInBytes, size_t height);
_GPUCV_CUDA_EXPORT_CU void gcudaMemset(void* devPtr, int value, size_t count);
	//
_GPUCV_CUDA_EXPORT_CU void gcudaFree(void * _buffer);
	//
_GPUCV_CUDA_EXPORT_CU void gcudaMallocArray(struct cudaArray** array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height);
_GPUCV_CUDA_EXPORT_CU void gcudaMemcpyToArray(struct cudaArray* dstArray,size_t dstX, size_t dstY,const void* src, size_t count,enum cudaMemcpyKind kind);
_GPUCV_CUDA_EXPORT_CU void gcudaMemcpyFromArray(void* dst,const struct cudaArray* srcArray,size_t srcX, size_t srcY,size_t count,enum cudaMemcpyKind kind);
_GPUCV_CUDA_EXPORT_CU void gcudaFreeArray(struct cudaArray* array);
_GPUCV_CUDA_EXPORT_CU cudaChannelFormatDesc gcudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);
_GPUCV_CUDA_EXPORT_CU cudaChannelFormatDesc * gcudaCopyChannelDesc(const cudaChannelFormatDesc & _channelDesc, cudaChannelFormatDesc *_channelD_dst);
_GPUCV_CUDA_EXPORT_CU const char * gcuGetStrPixelType(CUarray_format_enum _type);

	//
_GPUCV_CUDA_EXPORT_CU void gcudaMemCopy(void * _bufferDst, void * _bufferSrc, unsigned int _size, cudaMemcpyKind _destination);
#define gcudaMemCopyHostToDevice(DST, SRC, SIZE) gcudaMemCopy(DST, SRC, SIZE, cudaMemcpyHostToDevice)
#define gcudaMemCopyDeviceToHost(DST, SRC, SIZE) gcudaMemCopy(DST, SRC, SIZE, cudaMemcpyDeviceToHost)
#define gcudaMemCopyDeviceToDevice(DST, SRC, SIZE) gcudaMemCopy(DST, SRC, SIZE, cudaMemcpyDeviceToDevice)
	//
_GPUCV_CUDA_EXPORT_CU void gcudaThreadSynchronize(void);
	//
_GPUCV_CUDA_EXPORT_CU void gcudaGetDeviceCount(int * _count);
	//
_GPUCV_CUDA_EXPORT_CU void gcudaGetDeviceProperties(cudaDeviceProp * _devideProp, int _devId);


	_GPUCV_CUDA_EXPORT_CU void gcudaPrintProperties();
	/** \todo Add CUDA RT error detection
	*/
	_GPUCV_CUDA_EXPORT_CU void gcudaCheckError(const char *_msg);
	//#endif
	//=======================================
	//
	//		CU CPP wrapping functions
	//
	//=======================================
	//void cuwrInit(unsigned int Flags);
	//void gcudaGetDeviceCount(int* count);
	//void cuwrDeviceGetName(char* name, int len, CUdevice dev);
	//void gcudaDeviceGet(CUdevice* dev, int ordinal);
	//void cuwrDeviceTotalMem(unsigned int* bytes, CUdevice dev);
	//void cuwrDeviceComputeCapability(int* major, int* minor,CUdevice dev);
	//void gcudaGetDeviceProperties(CUdevprop* prop,CUdevice dev);
	//=======================================
	//
	//		CUT CPP wrapping functions
	//
	//=======================================
	/*
	_GPUCV_CUDA_EXPORT_CU void cutwrDeviceInit(void);

	_GPUCV_CUDA_EXPORT_CU void cutwrCheckError(const char *_msg);
	_GPUCV_CUDA_EXPORT_CU void cuwrMemGetInfo(unsigned int *Free, unsigned int *Total);
	*/
	//________________________________________
#endif// _GPUCV_COMPILE_CUDA
#endif//__GPUCV_gcu_runtime_api_wrapper_H
