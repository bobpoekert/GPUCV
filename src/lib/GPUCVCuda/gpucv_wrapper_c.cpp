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
#include <GPUCV/include.h>
#include <GPUCVCuda/config.h>

#if _GPUCV_COMPILE_CUDA
#include <GPUCVCuda/DataDsc_CUDA_Buffer.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <GPUCVCuda/GPU_NVIDIA_CUDA.h>

#include <cxcoreg/cxcoreg.h>
#include <highguig/highguig.h>

using namespace GCV;
//=================================================
GLuint gcuGetWidth(const CvArr * arr)
{
	return GetWidth(arr);
}
//=================================================
GLuint gcuGetHeight(const CvArr * arr)
{
	return GetHeight(arr);
}
//=================================================
GLuint gcuGetGLDepth(const CvArr * arr)
{
	return GetGLDepth(arr);
}
//=================================================
GLuint gcuGetCVDepth	(const CvArr * arr)
{
	return GetCVDepth(arr);
}
//=================================================
GLuint gcuGetnChannels(const CvArr * arr)
{
	return GetnChannels(arr);
}
//=================================================
GLuint gcuGetGLTypeSize	(unsigned int _depth)
{
	return GetGLTypeSize(_depth);
}
//=================================================
bool gcuGetDoubleSupport()
{
	GPU_NVIDIA_CUDA *cuda_gpu = dynamic_cast<GPU_NVIDIA_CUDA *>(GCV::ProcessingGPU());
	if(cuda_gpu)
		return cuda_gpu->IsDoubleSupported();
	return false;
}
//=================================================
cudaDeviceProp * gcuGetDeviceProperties()
{
	GPU_NVIDIA_CUDA *cuda_gpu = dynamic_cast<GPU_NVIDIA_CUDA *>(GCV::ProcessingGPU());
	if(cuda_gpu)
		return &cuda_gpu->GetCudaProperties();
	return NULL;//default is 16
}
//=================================================
/**
\bug FIXED: the local cache is not working properly cause an IplImage address can be reaffected by the OS to another IplImage, then it was looking for an non allocated CvgArr pointer.
*/
inline
CvgArr * GetCvgArr(void * _img)
{
	GCU_Assert(_img,"GetCvgArr()=>Empty obj");
#if 0
	static void * lastImg=NULL;
	static CvgArr * lastCvgImg=NULL;

	if(_img == lastImg)
		return lastCvgImg;
	else

	{
		lastCvgImg = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(_img));
		if(lastCvgImg)
			lastImg = _img;
		return lastCvgImg;
	}
#else
	return dynamic_cast<CvgArr*>(GPUCV_GET_TEX(_img));
#endif
}
//=================================================
#if _GPUCV_DEPRECATED
inline
DataDsc_CUDA_Base * GetCudaDesc(void * _img)
{
	static void * lastImg=NULL;
	static DataDsc_CUDA_Base * lastCudaDsc=NULL;
	GCU_Assert(_img,"GetCvgArr()=>Empty obj");

	if(_img == lastImg)
		return lastCudaDsc;
	else
	{
		CvgArr * cvgImg = GetCvgArr(_img);
		if(cvgImg)
		{
			lastCudaDsc= cvgImg->GetDataDsc<DataDsc_CUDA_Base>();
			lastImg = _img;
			return lastCudaDsc;
		}
		else
			return NULL;
	}
}
#endif
//=================================================
void* gcuPreProcess(void * _img, GCU_IO_TYPE _iotype, int _cudaMemoryType/*=0*/, cudaChannelFormatDesc* _channelDesc/*=NULL*/)
{
	GPUCV_FUNCNAME("gcu_PreProcess");

	//CvgArr * CurCvArr = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(_img));
	CvgArr * CurCvArr = GetCvgArr(_img);

	if (!CurCvArr)
	{
		GPUCV_ERROR(GPUCV_GET_FCT_NAME() <<"=> Input obj not an OpenCV object");
		return NULL;
	}

	DataDsc_CUDA_Base * CudaImg=NULL;

	//we control data transfer
	bool Datatransfer = false;
	if(_iotype == GCU_OUTPUT)
	{//manage destination Textures ========================
		CurCvArr->PushSetOptions(DataContainer::DEST_IMG, true);
	}
	else
	{//manage sources Textures ========================
		CurCvArr->PushSetOptions(DataContainer::UBIQUITY, true);	//is used to preserve CPU image if we have cpu_return set
		CurCvArr->SetOption(DataContainer::DEST_IMG, false);
		Datatransfer = true;
	}

	//make sure image is on GPU
	if(_cudaMemoryType == CU_MEMORYTYPE_DEVICE)
	{
		CudaImg = CurCvArr->SetLocation<DataDsc_CUDA_Buffer>(Datatransfer);
		if(_channelDesc)
			GPUCV_WARNING("_channelDsc is defined and memory type is CU_MEMORYTYPE_DEVICE, it should be CU_MEMORYTYPE_ARRAY");
	}
	else if(_cudaMemoryType = CU_MEMORYTYPE_ARRAY)
	{
		if(_channelDesc)
			CurCvArr->GetDataDsc<DataDsc_CUDA_Array>()->_SetCudaChannelFormatDesc(_channelDesc);
		CudaImg = CurCvArr->SetLocation<DataDsc_CUDA_Array>(Datatransfer);
	}
	else
		GPUCV_ERROR("Unknown CUDA MEMORY TYPE:" << _cudaMemoryType);

	return CudaImg->_GetDataPtr();

}
//=================================================
bool gcuPostProcess(void * _img)
{
	GPUCV_FUNCNAME("gcu_PostProcess");

	//CvgArr * CurCvArr = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(_img));
	CvgArr * CurCvArr = GetCvgArr(_img);

	if (!CurCvArr)
	{
		GPUCV_ERROR(GPUCV_GET_FCT_NAME() <<"=> Input obj not an OpenCV object");
		return false;
	}

	if(CurCvArr->GetOption(DataContainer::DEST_IMG))
	{//manage destination Textures ========================
		//	CurCvArr->PushSetOptions(DataContainer::DEST_IMG, true);

		if (CurCvArr->GetOption(DataContainer::CPU_RETURN))
		{
			CurCvArr->SetLocation<DataDsc_CPU>(true);
		}
		CurCvArr->PopOptions();
		//destination is now considered as input texture...
		CurCvArr->SetOption(DataContainer::DEST_IMG, false);
		return true;
	}
	else
	{//manage sources Textures ========================
		//???if (CurCvArr->GetOption(DataContainer::CPU_RETURN))
		//???	CurCvArr->SetLocation<DataDsc_CPU>(true);
		CurCvArr->PopOptions();
		return true;
	}
}

//=================================================
size_t gcuGetPitch(void * _img)
{
	GPUCV_FUNCNAME("gcuGetPitch");

	//CvgArr * CurCvArr = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(_img));
	CvgArr * CurCvArr = GetCvgArr(_img);

	if (!CurCvArr)
	{
		GPUCV_ERROR(GPUCV_GET_FCT_NAME() <<"=> Input obj not an OpenCV object");
		return NULL;
	}

	if(CurCvArr->FindDataDscID<DataDsc_CUDA_Buffer>()>=0)
	{
		DataDsc_CUDA_Buffer * CudaBuff= CurCvArr->GetDataDsc<DataDsc_CUDA_Buffer>();
		if(CudaBuff)
			return CudaBuff->_GetPitch();
	}
	return 0;
}
//=================================================
void* gcuSyncToCPU(void * _img, bool _dataTransfer)
{
	GPUCV_FUNCNAME("gcuSyncToCPU");

	if(_img==NULL)
		return NULL;
	//CvgArr * CurCvArr = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(_img));
	CvgArr * CurCvArr = GetCvgArr(_img);

	if (!CurCvArr)
	{
		GPUCV_ERROR(GPUCV_GET_FCT_NAME() <<"=> Input obj not an OpenCV object");
		return NULL;
	}

	DataDsc_CPU * CPUImg = CurCvArr->SetLocation<DataDsc_CPU>(_dataTransfer);
	return *CPUImg->_GetPixelsData();
}
//======================================
#if _GPUCV_DEPRECATED
void gcuSetReshapeObj(void * _img, GCU_IO_TYPE _iotype, int _cudaMemoryType, int _newChannels)
{
	GPUCV_FUNCNAME("gcuSetReshapeObj");

	if(_img==NULL)
		return;
	//CvgArr * CurCvArr = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(_img));
	CvgArr * CurCvArr = GetCvgArr(_img);

	if (!CurCvArr)
	{
		GPUCV_ERROR(GPUCV_GET_FCT_NAME() <<"=> Input obj not an OpenCV object");
		return;
	}

	DataDsc_CUDA_Base * CudaImg=NULL;
	bool DataTransfer = (_iotype==GCU_OUTPUT)?false:true;

	//make sure image is on GPU
	if(_cudaMemoryType == CU_MEMORYTYPE_DEVICE)
	{
		CudaImg = CurCvArr->SetLocation<DataDsc_CUDA_Buffer>(DataTransfer);
	}
	else if(_cudaMemoryType = CU_MEMORYTYPE_ARRAY)
	{
		CudaImg = CurCvArr->SetLocation<DataDsc_CUDA_Array>(DataTransfer);
	}
	else
		GPUCV_ERROR("Unknown CUDA MEMORY TYPE:" << _cudaMemoryType);

	CudaImg->SetReshape(_newChannels);
}
//=================================================
void gcuUnsetReshapeObj(void * _img,int _cudaMemoryType)
{
	GPUCV_FUNCNAME("gcuUnsetReshapeObj");
	if(_img==NULL)
		return;

	//CvgArr * CurCvArr = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(_img));
	CvgArr * CurCvArr = GetCvgArr(_img);

	if (!CurCvArr)
	{
		GPUCV_ERROR(GPUCV_GET_FCT_NAME() <<"=> Input obj not an OpenCV object");
		return;
	}

	DataDsc_CUDA_Base * CudaImg=NULL;

	//get DataDesc
	if(_cudaMemoryType == CU_MEMORYTYPE_DEVICE)
	{
		CudaImg = CurCvArr->GetDataDsc<DataDsc_CUDA_Buffer>();
	}
	else if(_cudaMemoryType = CU_MEMORYTYPE_ARRAY)
	{
		CudaImg = CurCvArr->GetDataDsc<DataDsc_CUDA_Array>();
	}
	else
		GPUCV_ERROR("Unknown CUDA MEMORY TYPE:" << _cudaMemoryType);

	CudaImg->UnsetReshape();
}
#endif//DEPRECATED
//=================================================
bool gcuGetDataDscSize(void * _img,int _cudaMemoryType, unsigned int & width, unsigned int &height)
{
	GPUCV_FUNCNAME("gcuDataDscSize");
	//CvSize Size;
	//Size.width=Size.height=0;
	if(_img==NULL)
	{
		return false;
	}

	//CvgArr * CurCvArr = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(_img));
	CvgArr * CurCvArr = GetCvgArr(_img);

	if (!CurCvArr)
	{
		GPUCV_ERROR(GPUCV_GET_FCT_NAME() <<"=> Input obj not an OpenCV object");
		return false;
	}

	DataDsc_CUDA_Base * CudaImg=NULL;

	//make sure image is on GPU
	if(_cudaMemoryType == CU_MEMORYTYPE_DEVICE)
	{
		CudaImg = CurCvArr->GetDataDsc<DataDsc_CUDA_Buffer>();
	}
	else if(_cudaMemoryType = CU_MEMORYTYPE_ARRAY)
	{
		CudaImg = CurCvArr->GetDataDsc<DataDsc_CUDA_Array>();
	}
	else
	{
		GPUCV_ERROR("Unknown CUDA MEMORY TYPE:" << _cudaMemoryType);
		return false;
	}

	width = CudaImg->_GetWidth();
	height = CudaImg->_GetHeight();
	return true;
}
//=================================================
#if _GPUCV_DEPRECATED
bool FindBestLoad(unsigned int _size, unsigned int & _blockNbr, unsigned int & _ThreadNbr, unsigned int & _ThreadWidth)
{
	unsigned int CurProcSize = _blockNbr*_ThreadNbr*_ThreadWidth;
	if(_size == CurProcSize)
	{//well, it fits...see optimisations later
		return true;
	}

	//make sure current value are not too high
	while(_size < _ThreadWidth)
	{
		_ThreadWidth -= 16;
		GPUCV_ERROR("Reduce _ThreadWidth to "<< _ThreadWidth);
		if(_ThreadWidth < 1)
		{
			_ThreadWidth = 1;
			break;
		}
	}

	//========================================
	float fCoef = 0;
	int iCoef = 0;

	if(IS_MULTIPLE_OF(_size,2))
	{
		fCoef = _size/_ThreadWidth;
		if(IS_INTEGER(fCoef))
		{
			iCoef = fCoef;

			if(IS_MULTIPLE_OF(iCoef,2))
			{//we can use a simple *2 |/2 mechanism
				//dispatch the load on ThreadNbr and BlockNbr
				for(int i = _ThreadNbr; i > 0; i>>=1)
				{
					if (IS_MULTIPLE_OF(iCoef,i))
					{
						_ThreadNbr = i;
						if(_blockNbr>=iCoef/_ThreadNbr)
						{
							_blockNbr = iCoef/_ThreadNbr;
							return true;
						}
						else
						{
							GPUCV_ERROR("Block Nbr exceed maximum block number defined");
							return false;
						}
					}
				}
			}
		}
		else
		{//find something else...
		}
	}
	GPUCV_ERROR("Could not find best values");
	return false;


	//using the given parameter we try to adjust using the following rules.
	// 1- if _ThreadWidth == 1, we don't change it.
	// 2- _ThreadNbr should be multiple of 2, and inferior to 512(later we may consider the 2 dimensions).
	// 3- _blockNbr is not really limited

	/*
	if(Coef < 1)
	{//we increase values
	Coef = CurProcSize_size/
	}
	*/
}
#endif// _GPUCV_DEPRECATED
//=======================================
void gcuShowImage(char* Name, unsigned int width, unsigned int height, unsigned int depth, unsigned int channels, void * _device_data, int _pixelSize, float extra_scale)
{
//#if defined (_LINUX) || defined (_MACOS)
	IplImage * TempImge = cvgCreateImage(cvSize(width, height), depth, channels);
	//	cvCreateData(TempImge);
	gcudaThreadSynchronize();
	gcudaMemCopyDeviceToHost(TempImge->imageData, _device_data, width*height*channels*_pixelSize);
	gcudaThreadSynchronize();
	if(extra_scale!=1.)
		cvScale(TempImge, TempImge, extra_scale);
	cvNamedWindow(Name, 1);
	cvgShowImage(Name, TempImge);
	cvWaitKey(0);
	cvDestroyWindow(Name);
	cvgReleaseImage(&TempImge);
//#else
//	GPUCV_NOTICE("gcuShowImage(): Fonction no supported under windows");
//#endif
}
//=======================================
#endif//#if _GPUCV_COMPILE_CUDA
