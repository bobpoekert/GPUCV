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
#include <GPUCVCuda/config.h>
#if _GPUCV_COMPILE_CUDA
#include <GPUCV/misc.h>
#include <GPUCVCuda/cuda_misc.h>
#include <GPUCVCuda/GPU_NVIDIA_CUDA.h>
#include <cublas.h>
#include <cutil_inline.h>
#include <cuda_gl_interop.h>
#include <GPUCVHardware/moduleInfo.h>
/*
*   \author Yannick Allusse
*/

using namespace GCV;



_GPUCV_CUDA_EXPORT_C LibraryDescriptor *modGetLibraryDescriptor(void)
{
	static LibraryDescriptor * pLibraryDescriptor=NULL;
	if(pLibraryDescriptor ==NULL)//init
	{
		pLibraryDescriptor = new LibraryDescriptor();
		pLibraryDescriptor->SetVersionMajor("1");
		pLibraryDescriptor->SetVersionMinor("0");
		pLibraryDescriptor->SetSvnRev("570");
		pLibraryDescriptor->SetSvnDate("");
		pLibraryDescriptor->SetWebUrl(DLLINFO_DFT_URL);
		pLibraryDescriptor->SetAuthor(DLLINFO_DFT_AUTHOR);
		pLibraryDescriptor->SetDllName("GPUCVCuda");
		pLibraryDescriptor->SetImplementationName("CUDA");
		pLibraryDescriptor->SetBaseImplementationID(GPUCV_IMPL_CUDA);
		pLibraryDescriptor->SetUseGpu(true);
		pLibraryDescriptor->SetStartColor(GPUCV_IMPL_CUDA_COLOR_START);
		pLibraryDescriptor->SetStopColor(GPUCV_IMPL_CUDA_COLOR_STOP);
	}
	return pLibraryDescriptor;
}
//========================================================================
char ucCudaDeviceID = -1;
void cvgCudaSetProcessingDevice(char _devID/*=-1*/)
{
	ucCudaDeviceID = (_devID==-1)? cutGetMaxGflopsDeviceId():_devID;
}
//========================================================================
int  cvgcuInit(bool InitGLContext/*=true*/, bool isMultiThread/*=false*/)
{
	static bool CUDA_INIT_DONE = false;

	if(CUDA_INIT_DONE==true)
		return 0;//already done


	if(cvgInit(InitGLContext, isMultiThread)==-1)
		return -1;

	//check that we have some NVIDIA GPUs:
	unsigned int uiNvidiaGPU =0;
	std::vector<GenericGPU*>::iterator iterGPU;
	for(iterGPU = GetHardProfile()->GetFirstGPUIter();
		iterGPU != GetHardProfile()->GetLastGPUIter();
		iterGPU++)
	{
		if( (*iterGPU)->GetBrand() == "NVIDIA")
			uiNvidiaGPU ++;
	}

	if(uiNvidiaGPU  == 0)
		return true;// not need to init CUDA


	//init cuda
//	cuInit(cutGetMaxGflopsDeviceId());

//do not work with NSIGHT on.??
	if(ucCudaDeviceID==-1)
		cvgCudaSetProcessingDevice(-1);
	GCU_CUDA_SAFE_CALL(cudaGLSetGLDevice(ucCudaDeviceID));
	

	int DeviceCount = 0;
	gcudaGetDeviceCount(&DeviceCount);
	int iMainGPUId = 0;
	if(DeviceCount ==0)
	{
		GPUCV_NOTICE("cvgcuInit()=>No compatible device, must run in emulation mode");
		return false;
	}
	else
	{
		GPUCV_NOTICE(DeviceCount << " CUDA capable GPU found, selecting first one");
#if CUDA_VERSION > 2300
		iMainGPUId =cutGetMaxGflopsDeviceId(); 
		//cudaSetDevice(iMainGPUId);
		//GCU_CUDA_SAFE_CALL(cudaGLSetGLDevice( iMainGPUId ));
#endif
	}



/*
	if(DeviceCount > 1)
		cudaSetDevice( cutGetMaxGflopsDeviceId() );
	else
		cudaSetDevice(0);
*/

	

	GPU_NVIDIA_CUDA * CudaGpu = NULL;
	for (int i =0; i < DeviceCount; i++)
	{
		//add CUDA gpu to list of GPUs
		CudaGpu = new GPU_NVIDIA_CUDA();
		CudaGpu->SetCudaDeviceID(i);
		CudaGpu->ReadGPUSettings();
		GetHardProfile()->AddGPU(CudaGpu);
		if(i==iMainGPUId)
		{//?? set first one as main...??
			GetHardProfile()->SetMainGPU(CudaGpu);
			GetHardProfile()->SetRenderingGPU(CudaGpu);
			GetHardProfile()->SetProcessingGPU(CudaGpu);
		}
		// Create context
		//CUcontext cuContext;
		//cuGLCtxCreate(&cuContext, 0, i);
		//what about the already detected GPU from GPU_NVIDIA??
	}	

	//cublasShutdown() is called in GPU_NVIDIA::~GPU_NVIDIA()
	//cublasInit();
	
#if NDEBUG
	if(GetGpuCVSettings()->GetOption(CL_Options::LCL_OPT_DEBUG))
#endif
//		gcudaPrintProperties();//print always in debug mode.

	CUDA_INIT_DONE=true;
	return 1;//success
}

//==================================================
void cvgCudaThreadSynchronize()
{
	//gcudaThreadSynchronize();
	GCU_CUDA_SAFE_CALL( cudaThreadSynchronize() );
}
#endif//CUDA_SUPPORT
