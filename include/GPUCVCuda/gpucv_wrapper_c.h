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

/** \brief Includes Some C wrapper functions to access some GpuCV CPP objects from CUDA .cu files.
*	\author Yannick Allusse
*/
#ifndef __GPUCV_CUDA_WRAPPER_C_H
#define __GPUCV_CUDA_WRAPPER_C_H

#include <GPUCVCuda/config.h>
#include <GPUCVCuda/gcu_runtime_api_wrapper.h>

#if _GPUCV_COMPILE_CUDA
	/**
	\sa CUmemorytype_enum
	*/
	enum GCU_IO_TYPE
	{
		GCU_INPUT,
		GCU_OUTPUT
	};

	/** \brief Pre-process a data object and load it into the requested CUDA memory.
	Pre-process a data object before a CUDA operator. It affects a I/O type to it and make sure the data are copied into the
	correct CUDA memory zone (_cudaMemoryType). If the zone is a CUDA array, a channel descriptor can be specified to use texture memory.
	\param _img -> Pointer to the input object(ex: IplImage or CvMat).
	\param _iotype -> [GCU_INPUT|GCU_OUTPUT].
	\param _cudaMemoryType -> Memory type, see CUmemorytype_enum.
	\param _channelDesc -> Optional channel descriptor when _cudaMemoryType==CU_MEMORYTYPE_ARRAY.
	*	\sa gcuPostProcess(), gcuGetPitch().
	*	\author Yannick Allusse
	*/
		_GPUCV_CUDA_EXPORT_C
			void* gcuPreProcess(void * _img, GCU_IO_TYPE _iotype, int _cudaMemoryType=0, cudaChannelFormatDesc* _channelDesc=NULL);

	/** \brief Post-process a data object and copy result to CPU if required.
	Post-process a data object after a CUDA operator. Data might be copied to CPU if object options is specified.
	\param _img -> Pointer to the input object(ex: IplImage or CvMat).
	*	\sa gcuPreProcess(), gcuGetPitch().
	*	\author Yannick Allusse
	*/
		_GPUCV_CUDA_EXPORT_C
			bool gcuPostProcess(void * _img);

	/** \brief Return the pitch size of a data object.
	\param _img -> Pointer to the input object(ex: IplImage or Cvmat).
	*	\sa gcuPreProcess(), gcuPostProcess().
	*	\author Yannick Allusse
	*/
		_GPUCV_CUDA_EXPORT_C
			size_t gcuGetPitch(void * _img);

		_GPUCV_CUDA_EXPORT_C
			void* gcuSyncToCPU(void * _img, bool _dataTransfer);
		
#if _GPUCV_DEPRECATED
		_GPUCV_CUDA_EXPORT_C
			void gcuSetReshapeObj(void * _img, GCU_IO_TYPE _iotype, int _cudaMemoryType, int _newChannels);

		_GPUCV_CUDA_EXPORT_C
			void gcuUnsetReshapeObj(void * _img,int _cudaMemoryType);
#endif

		_GPUCV_CUDA_EXPORT_C
			bool gcuGetDataDscSize(void * _img,int _cudaMemoryType, unsigned int & width, unsigned int &height);

		/** \deprecated
		*/
		_GPUCV_CUDA_EXPORT_C
			bool FindBestLoad(unsigned int _size, unsigned int & _blockNbr, unsigned int & _ThreadNbr, unsigned int & _ThreadWidth);

#if defined (_LINUX) || defined (_MACOS)
#define gcuGLuint GLuint	
#else
#define gcuGLuint unsigned int
#endif
		_GPUCV_CUDA_EXPORT_C
			gcuGLuint gcuGetWidth	(const void * arr);
		_GPUCV_CUDA_EXPORT_C
			gcuGLuint gcuGetHeight	(const void * arr);
		_GPUCV_CUDA_EXPORT_C
			gcuGLuint gcuGetGLDepth	(const void * arr);
		_GPUCV_CUDA_EXPORT_C
			gcuGLuint gcuGetCVDepth	(const void * arr);
		_GPUCV_CUDA_EXPORT_C
			gcuGLuint gcuGetnChannels	(const void * arr);
		_GPUCV_CUDA_EXPORT_C
			gcuGLuint gcuGetGLTypeSize	(unsigned int _depth);

		_GPUCV_CUDA_EXPORT_C
			bool gcuGetDoubleSupport();
		
		_GPUCV_CUDA_EXPORT_C
			cudaDeviceProp * gcuGetDeviceProperties();
	//=================================================
		_GPUCV_CUDA_EXPORT_C
			void gcuShowImage(char* Name, unsigned int width, unsigned int height, unsigned int depth, unsigned int channels, void * _device_data, int _pixelSize, float extra_scale=1.);
#endif //_GPUCV_COMPILE_CUDA
#endif //__GPUCV_CUDA_WRAPPER_C_H
