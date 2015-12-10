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



/** \brief Contains a GPU class for NVIDIA CUDA capable cards.
*	\author Yannick Allusse
*/
#ifndef __GPU_NVIDIA_CUDA_H
#define __GPU_NVIDIA_CUDA_H

#include <GPUCVHardware/GPU_NVIDIA.h>
#include <GPUCVCuda/gcu_runtime_api_wrapper.h>
#if _GPUCV_COMPILE_CUDA

namespace GCV{
/*!
\brief NVIDIA GPU class definition + CUDA support.
\sa GenericGPU, GPU_NVIDIA.
\todo Check DOUBLE format support and CUDA compatibility version.
*/
class _GPUCV_CUDA_EXPORT GPU_NVIDIA_CUDA
	: public GPU_NVIDIA
{
public:
	GPU_NVIDIA_CUDA();
	~GPU_NVIDIA_CUDA();

	virtual bool ReadBrandSettings();
	virtual bool ReadGPUSettings();
	//	virtual bool SetMultiGPUMode(MultiGPUMode _mode);

	//void GetMemUsage(unsigned int * _Free, unsigned int * _Total);
	virtual std::string GetMemUsage()const;

protected:
	//! Id of the CUDA device, default is 0 for single GPU system.
	_DECLARE_MEMBER(CUdevice, CudaDeviceID);
	//CUdevice			m_CudaDevice;		//!< Pointer to the corresponding CUDA device. See cuDeviceGet().
	//! Pointer to the corresponding CUDA device properties. See cudaGetDeviceProperties().
	_DECLARE_MEMBER(cudaDeviceProp,CudaProperties);
	//! \sa cuDeviceTotalMem().
	_DECLARE_MEMBER(unsigned int, CudaTotalMem);		
	//! \sa cuDeviceGetName().
	_DECLARE_MEMBER(std::string, CudaName);			
	_DECLARE_MEMBER(bool,  DoubleSupport);
	//cuDeviceComputeCapability()
};
_GPUCV_HARDWARE_EXPORT bool createGPU(GenericGPU * _GpuTable, int * _gpuNbr);
}//namespace GCV
#endif
#endif//support CUDA..
