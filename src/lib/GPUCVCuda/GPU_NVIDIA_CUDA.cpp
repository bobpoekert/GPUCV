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
#include "StdAfx.h"
#include <GPUCVCuda/GPU_NVIDIA_CUDA.h>

#if _GPUCV_COMPILE_CUDA
#include <cublas.h>

using namespace GCV;
//=================================================
GPU_NVIDIA_CUDA::GPU_NVIDIA_CUDA()
:GPU_NVIDIA()
,m_CudaDeviceID(-1)
//,m_CudaDevice(-1)
,m_CudaProperties()
,m_CudaTotalMem(0)
,m_DoubleSupport(false)
{
	//	cublasInit(); it is done in cvgcuInit()
}
//=================================================
GPU_NVIDIA_CUDA::~GPU_NVIDIA_CUDA()
{
	cublasShutdown();
}
//=================================================
std::string GPU_NVIDIA_CUDA::GetMemUsage()const
{
	std::string Msg;
	/*
	unsigned int FreeMem, TotalMem;
	cuwrMemGetInfo(&FreeMem, &TotalMem);
	Msg+= "\nCuda Total memory: ";
	Msg+= SGE::ToCharStr(TotalMem);
	Msg+= "\nCuda Free memory: ";
	Msg+= SGE::ToCharStr(FreeMem);
	Msg+= " (";
	Msg+= SGE::ToCharStr((float)FreeMem/TotalMem*100.);
	Msg+= "%)";
	*/
#if 1
	unsigned int TotalMem=0;
	unsigned int FreeMem=0;
	cuMemGetInfo(&FreeMem,&TotalMem);
	if(FreeMem==0 || TotalMem==0)
	{
		Msg = "Can not retrieve CUDA memory information";
	}
	else
	{
		Msg = "Cuda free memory:";
		Msg += SGE::ToCharStr(FreeMem);
		Msg += "(";
		Msg += SGE::ToCharStr((double)FreeMem/TotalMem*100);
		Msg += "%)";
	}
#else
	Msg = "Unknown memory usage";
#endif

	return Msg;
}
//=================================================
/*virtual*/
bool GPU_NVIDIA_CUDA::ReadGPUSettings()
{
	//read classic GPU settings
	GPU_NVIDIA::ReadGPUSettings();

	//read cuda GPU settings
	GCU_Assert(m_CudaDeviceID>-1, "GPU_NVIDIA_CUDA::ReadGPUSettings()=>No cuda device detected");

	GPUCV_DEBUG(GetMemUsage());
	SetGLSLProfile(GenericGPU::HRD_PRF_CUDA);
	//(m_CudaDevice,m_cudaDeviceID);
	//GCU_Assert(m_CudaDevice, "Could not get CUDA device object.");
	//char * name=NULL;
	//cuwrDeviceGetName(name,256,*m_cudaDevice);
	//cuwrDeviceTotalMem(&m_cudaTotalMem,*m_cudaDevice);
	gcudaGetDeviceProperties(&m_CudaProperties,m_CudaDeviceID);

	return true;
}
//=================================================
bool GPU_NVIDIA_CUDA::ReadBrandSettings()
{
	return GPU_NVIDIA::ReadBrandSettings();
}
//=================================================

#endif//CUDA_SUPORT
