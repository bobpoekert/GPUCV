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
#include <GPUCVCuda/gcu_runtime_api_wrapper.h>

#if 1
#ifdef _GPUCV_CUDA_SUPPORT_OPENGL
	#include <cuda_gl_interop.h>
	void gcudaGLRegisterBufferObject(GLuint _bufferObj)
	{ GCU_CUDA_SAFE_CALL(cudaGLRegisterBufferObject(_bufferObj));}

	void gcudaGLUnregisterBufferObject(GLuint _bufferObj)
	{ GCU_CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(_bufferObj));}

	void gcudaGLMapBufferObject(void** _devPtr, GLuint _bufferObj)
	{ GCU_CUDA_SAFE_CALL(cudaGLMapBufferObject(_devPtr, _bufferObj));}

	void gcudaGLUnmapBufferObject(GLuint _bufferObj)
	{ GCU_CUDA_SAFE_CALL(cudaGLUnmapBufferObject(_bufferObj));}
#endif


	//=======================================
	//
	//		CUDA CPP wrapping functions
	//
	//=======================================
	int iDivUp(int a, int b)
	{
		return (a % b != 0) ? (a / b + 1) : (a / b);
	}
	//
	void gcudaMalloc(void ** _buffer, unsigned int _size)
	{GCU_CUDA_SAFE_CALL( cudaMalloc((void **)_buffer, _size));}
	//
	void gcudaMallocPitch(void ** _buffer, size_t* pitch,size_t widthInBytes, size_t height)
	{GCU_CUDA_SAFE_CALL( cudaMallocPitch((void **)_buffer, pitch, widthInBytes, height));}
	//
	void gcudaMemset(void* devPtr, int value, size_t count)
	{GCU_CUDA_SAFE_CALL( cudaMemset(devPtr, value,count));}
	//
	void gcudaFree(void * _buffer)
	{GCU_CUDA_SAFE_CALL( cudaFree(_buffer) );}
	//Array
	void gcudaMallocArray(struct cudaArray** array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height)
	{GCU_CUDA_SAFE_CALL(cudaMallocArray(array, desc, width, height));}

	void gcudaMemcpyToArray(struct cudaArray* dstArray,size_t dstX, size_t dstY,const void* src, size_t count,enum cudaMemcpyKind kind)
	{GCU_CUDA_SAFE_CALL(cudaMemcpyToArray(dstArray, dstX, dstY, src,count,kind));}

	void gcudaMemcpyFromArray(void* dst,const struct cudaArray* srcArray,size_t srcX, size_t srcY,size_t count,enum cudaMemcpyKind kind)
	{GCU_CUDA_SAFE_CALL(cudaMemcpyFromArray(dst,srcArray, srcX, srcY,count,kind));}

	void gcudaFreeArray(struct cudaArray* array)
	{GCU_CUDA_SAFE_CALL(cudaFreeArray(array));}

	cudaChannelFormatDesc gcudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
	{return cudaCreateChannelDesc(x, y, z, w, f);}

	cudaChannelFormatDesc * gcudaCopyChannelDesc(const cudaChannelFormatDesc & _channelD_src, cudaChannelFormatDesc *_channelD_dst)
	{
		_channelD_dst->x = _channelD_src.x;
		_channelD_dst->y = _channelD_src.y;
		_channelD_dst->z = _channelD_src.z;
		_channelD_dst->w = _channelD_src.w;
		_channelD_dst->f = _channelD_src.f;
		return _channelD_dst;
	}

	//
	void gcudaMemCopy(void * _bufferDst, void * _bufferSrc, unsigned int _size, cudaMemcpyKind _destination)
	{GCU_CUDA_SAFE_CALL(cudaMemcpy(_bufferDst, _bufferSrc, _size, _destination));}

	//#define gcudaMemCopyHostToDevice(DST, SRC, SIZE) gcudaMemCopy(DST, SRC, SIZE, cudaMemcpyHostToDevice)
	//#define gcudaMemCopyDeviceToHost(DST, SRC, SIZE) gcudaMemCopy(DST, SRC, SIZE, cudaMemcpyDeviceToHost)
	//#define gcudaMemCopyDeviceToDevice(DST, SRC, SIZE) gcudaMemCopy(DST, SRC, SIZE, cudaMemcpyDeviceToDevice)
	//
	void gcudaThreadSynchronize(void)
	{
		//GCU_CUDA_SAFE_CALL( cudaThreadSynchronize() );
	}

	//
	void gcudaGetDeviceCount(int * _count)
	{GCU_CUDA_SAFE_CALL(cudaGetDeviceCount(_count));}

	//
	void gcudaGetDeviceProperties(cudaDeviceProp * _devideProp, int _devId)
	{GCU_CUDA_SAFE_CALL(cudaGetDeviceProperties(_devideProp, _devId));}
	//GL


	void gcudaPrintProperties()
	{
		int deviceCount;
		gcudaGetDeviceCount(&deviceCount);
		if (deviceCount == 0)
			printf("There is no device supporting CUDA\n");
		int dev;
		for (dev = 0; dev < deviceCount; ++dev) {
			cudaDeviceProp deviceProp;
			gcudaGetDeviceProperties(&deviceProp, dev);
			if (dev == 0) {
				if (deviceProp.major < 1)
					printf("There is no device supporting CUDA.\n");
				else if (deviceCount == 1)
					printf("There is 1 device supporting CUDA\n");
				else
					printf("There are %d devices supporting CUDA\n", deviceCount);
			}
			printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
			printf("  Major revision number:                         %d\n",
				deviceProp.major);
			printf("  Minor revision number:                         %d\n",
				deviceProp.minor);
			printf("  Total amount of global memory:                 %d bytes\n",
				deviceProp.totalGlobalMem);
			printf("  Total amount of constant memory:               %d bytes\n",
				deviceProp.totalConstMem); 
			printf("  Total amount of shared memory per block:       %d bytes\n",
				deviceProp.sharedMemPerBlock);
			printf("  Total number of registers available per block: %d\n",
				deviceProp.regsPerBlock);
			printf("  Warp size:                                     %d\n",
				deviceProp.warpSize);
			printf("  Maximum number of threads per block:           %d\n",
				deviceProp.maxThreadsPerBlock);
			printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
				deviceProp.maxThreadsDim[0],
				deviceProp.maxThreadsDim[1],
				deviceProp.maxThreadsDim[2]);
			printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
				deviceProp.maxGridSize[0],
				deviceProp.maxGridSize[1],
				deviceProp.maxGridSize[2]);
			printf("  Maximum memory pitch:                          %d bytes\n",
				deviceProp.memPitch);
			printf("  Texture alignment:                             %d bytes\n",
				deviceProp.textureAlignment);
			printf("  Clock rate:                                    %d kilohertz\n",
				deviceProp.clockRate);
		}
	}
	void gcudaCheckError(const char *_msg)
	{
		cudaError_t ErrorId = cudaGetLastError();
		if(ErrorId != cudaSuccess)
		{
			const char* ErrMsg =  cudaGetErrorString(ErrorId);
			printf("\nCUDA error msg: %s", ErrMsg);
			printf("\nCUDA error comment: %s", _msg);
		}
	}
	const char * gcuGetStrPixelType(CUarray_format_enum _type)
	{
		char * Format=NULL;
		switch(_type)
		{
		case  CU_AD_FORMAT_UNSIGNED_INT8:    Format = "CU_AD_FORMAT_UNSIGNED_INT8";break;
		case  CU_AD_FORMAT_UNSIGNED_INT16:   Format = "CU_AD_FORMAT_UNSIGNED_INT16";break;
		case  CU_AD_FORMAT_UNSIGNED_INT32:   Format = "CU_AD_FORMAT_UNSIGNED_INT32";break;
		case  CU_AD_FORMAT_SIGNED_INT8:		 Format = "CU_AD_FORMAT_SIGNED_INT8";break;
		case  CU_AD_FORMAT_SIGNED_INT16:	 Format = "CU_AD_FORMAT_SIGNED_INT16";break;
		case  CU_AD_FORMAT_SIGNED_INT32:	 Format = "CU_AD_FORMAT_SIGNED_INT32";break;
		case  CU_AD_FORMAT_HALF:/*Float32*/		Format = "CU_AD_FORMAT_HALF(float32)";break;
		case  CU_AD_FORMAT_FLOAT:/*double*/		Format = "CU_AD_FORMAT_FLOAT(float64)";break;
		default: Format = "unknown CUDA format";
		}
		return Format;
	}
	//=======================================
	//
	//		CU CPP wrapping functions
	//
	//=======================================
	/*
	void cuwrInit(unsigned int Flags)
	{CU_SAFE_CALL(cuInit(Flags));}
	*/
	//void gcudaGetDeviceCount(int* count)
	//{GCU_CUDA_SAFE_CALL(cudaGetDeviceCount( count));}
	/*
	void cuwrDeviceGetName(char* name, int len, CUdevice dev)
	{CU_SAFE_CALL(cuDeviceGetName( name, len, dev));}
	*/
	//void gcudaDeviceGet(CUdevice* dev, int ordinal)
	//{GCU_CUDA_SAFE_CALL(cudaDeviceGet( dev,ordinal));}
	/*
	void cuwrDeviceTotalMem(unsigned int* bytes, CUdevice dev)
	{CU_SAFE_CALL(cuDeviceTotalMem( bytes,dev));}

	void cuwrDeviceComputeCapability(int* major, int* minor,CUdevice dev)
	{CU_SAFE_CALL(cuDeviceComputeCapability( major, minor, dev));}
	*/
	//void gcudaGetDeviceProperties(CUdevprop* prop,CUdevice dev)
	//{GCU_CUDA_SAFE_CALL(cudaGetDeviceProperties(prop, dev));}
	//=======================================
	//
	//		CUT CPP wrapping functions
	//
	//=======================================
	/*
	void cutwrDeviceInit(void)
	{CUT_DEVICE_INIT();}


	void cutwrCheckError(const char *_msg)
	{CUT_CHECK_ERROR(_msg);}

	void cuwrMemGetInfo(unsigned int *Free, unsigned int *Total)
	{cuMemGetInfo(Free, Total);}
	*/

#endif//