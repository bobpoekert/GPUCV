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
#include <GPUCVTexture/TextureRecycler.h>
#include <GPUCVCuda/DataDsc_CUDA_Base.h>
#if _GPUCV_COMPILE_CUDA

#include <GPUCVCuda/GPU_NVIDIA_CUDA.h>
#include <GPUCVTexture/DataDsc_GLTex.h>
#include <GPUCVTexture/DataContainer.h>
#include <GPUCVTexture/DataDsc_GLBuff.h>
#include <GPUCVTexture/DataDsc_CPU.h>

using namespace GCV;
//==================================================
DataDsc_CUDA_Base::DataDsc_CUDA_Base()
: DataDsc_Base("DataDsc_CUDA_Base")
//,m_deviceDataPtr(NULL)
//,m_pitch(0)
//,m_deviceDataType(CUDA_NO_TYPE)
//,m_glBufferMapped(false)
,m_cudaPixelType(0)
//,m_textureArrayPtr(NULL)
//,m_textureChannelDesc(NULL)
,m_AutoMapGLBuff(true)
{
	m_data.m_textureArrayPtr	= NULL;
	m_data.m_deviceDataPtr		= NULL;
}
//==================================================
DataDsc_CUDA_Base::~DataDsc_CUDA_Base(void)
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("~DataDsc_CUDA_Base");
	//no Bench and log in Destructor//	CLASS_DEBUG("");
	Free(); //is done in ~DataDsc_Base()
}
//=======================================================
/*virtual*/
std::string DataDsc_CUDA_Base::LogException(void)const
{
	std::string Msg;
	Msg = CL_Profiler::LogException();
	GPU_NVIDIA_CUDA * CudaGPU = dynamic_cast<GPU_NVIDIA_CUDA*>(ProcessingGPU());
	if(CudaGPU)
	{
		Msg+=CudaGPU->GetMemUsage();
	}
	return Msg;
}
//==================================================
/*virtual*/
void DataDsc_CUDA_Base::SetFormat(const GLuint _pixelFormat,const GLuint _pixelType)
{
	DataDsc_Base::SetFormat(_pixelFormat,_pixelType);
	//??...?? m_nChannels = GetGLNbrComponent(_pixelFormat);
	//... find internal format...??
}
//==================================================
/*virtual*/
bool DataDsc_CUDA_Base::CopyTo(DataDsc_Base* _destination, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_destination->GetDscType(),"CopyTo");
	TEXTUREDESC_COPY_START(_destination,_datatransfer);

	//don't know how to copy...must be done in another object
	return false;
}
//==================================================
/*virtual*/
#if 0
bool DataDsc_CUDA_Base::CopyFrom(DataDsc_Base* _source, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_source->GetDscType(),"CopyFrom");
	TEXTUREDESC_COPY_START(_source,_datatransfer);


	//don't know how to copy...must be done in another object
	return false;
}
#endif
//==================================================
/*virtual*/
#if 0
DataDsc_Base * DataDsc_CUDA_Base::Clone(DataDsc_CUDA_Base * _src, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME("Clone");
	CLASS_FCT_PROF_CREATE_START();

	DataDsc_Base::Clone(_src, _datatransfer);
	Free();
	return this;
}
//==================================================
/*virtual*/

DataDsc_Base * DataDsc_CUDA_Base::CloneToNew(bool _datatransfer/*=true*/)
{
	DataDsc_CUDA_Base * TempTex = new DataDsc_CUDA_Base();
	return TempTex->Clone(this,_datatransfer);
}
#endif
//==================================================
/*virtual*/
void DataDsc_CUDA_Base::Allocate()
{
	CLASS_FCT_SET_NAME("Allocate");
	CLASS_FCT_PROF_CREATE_START();
	if(!IsAllocated())
	{
		DataDsc_Base::Allocate();//for compatibility
		//bool result = false;
		CLASS_DEBUG("Allocate device buffer");

#if _GPUCV_DEBUG_MODE
		GPU_NVIDIA_CUDA * CudaGPU = dynamic_cast<GPU_NVIDIA_CUDA*>(ProcessingGPU());
		if(CudaGPU)
		{
			GPUCV_DEBUG(CudaGPU->GetMemUsage());
		}
#endif
		GCU_Assert(m_cudaPixelType, "Empty m_cudaPixelType");
		_AllocateDevice(m_cudaPixelType);

		GCU_Assert(m_memSize, "empty memory size after device allocation!");

#if  _GPUCV_DEBUG_MODE
		if(CudaGPU)
		{
			GPUCV_DEBUG(CudaGPU->GetMemUsage());
		}
#endif
	}
}
//==================================================
/*virtual */
#if 0
void DataDsc_CUDA_Base::_AllocateDevice(unsigned int _datatype)
{
}
#endif
//==================================================
/*virtual*/
void DataDsc_CUDA_Base::Free()
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("Free");
	//no Bench and log in Destructor//	CLASS_FCT_PROF_CREATE_START();
	//if(IsAllocated())
	{
		DataDsc_Base::Free();
#if 0
		if(GetParent() && GetParent()->GetOption(DataContainer::DBG_IMG_MEMORY))
		{
			unsigned int TotalMem=0;
			unsigned int FreeMem=0;
			cuMemGetInfo(&FreeMem,&TotalMem);
			//no Bench and log in Destructor//			CLASS_NOTICE("");
			//no Bench and log in Destructor//			GPUCV_NOTICE("AFTER=> Cuda free memory:" << FreeMem << "(" << (double)FreeMem/TotalMem*100. <<"%)");
		}
#endif
	}
}
//==================================================
/*virtual*/
bool DataDsc_CUDA_Base::IsAllocated()const
{
	if(m_data.m_textureArrayPtr || m_data.m_deviceDataPtr)
		return true;
	return false;
}
//==================================================
void * DataDsc_CUDA_Base::_GetDataPtr(void)
{
	if(!IsAllocated())
		Allocate();

	return m_data.m_deviceDataPtr;//this is a union...so it should work in every cases
}
//==================================================
/*static virtual*/
void	DataDsc_CUDA_Base::ConvertPixelFormat_GLToLocal(const GLuint _pixelFormat)
{
	//GCU_CLASS_ASSERT(0, "DataDsc_CUDA_Base::ConvertPixelFormat_GLToLocal()=>No conversion available yet");
	if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
	{
		CLASS_FCT_SET_NAME("ConvertPixelFormat_GLToLocal");
		CLASS_WARNING("No conversion available yet");
	}
	//m_glPixelFormat = _pixelFormat;
}
//==================================================
/*static virtual*/
GLuint	DataDsc_CUDA_Base::ConvertPixelFormat_LocalToGL(void)
{
	//GCU_CLASS_ASSERT(0, "DataDsc_CUDA_Base::ConvertPixelFormat_GLToLocal()=>No conversion available yet");
	if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
	{
		CLASS_FCT_SET_NAME("ConvertPixelFormat_LocalToGL");
		CLASS_WARNING("No conversion available yet");
	}
	return m_glPixelFormat;
}
//==================================================
/*static virtual*/
void	DataDsc_CUDA_Base::ConvertPixelType_GLToLocal(const GLuint _pixelType)
{
	//	m_glPixelType = _pixelType;
	switch (_pixelType)
	{
	case GL_UNSIGNED_BYTE 	:  	m_cudaPixelType = CU_AD_FORMAT_UNSIGNED_INT8; break;
	case GL_UNSIGNED_SHORT 	:  	m_cudaPixelType = CU_AD_FORMAT_UNSIGNED_INT16; break;
	case GL_UNSIGNED_INT	: 	m_cudaPixelType = CU_AD_FORMAT_UNSIGNED_INT32;break;

	case GL_BYTE			:  	m_cudaPixelType = CU_AD_FORMAT_SIGNED_INT8; break;
	case GL_SHORT			:  	m_cudaPixelType = CU_AD_FORMAT_SIGNED_INT16; break;
	case GL_INT				: 	m_cudaPixelType = CU_AD_FORMAT_SIGNED_INT32;break;
	case GL_RGBA_FLOAT16_ATI:   GPUCV_WARNING("Conversion from GL_RGBA_FLOAT16_ATI to CUDA format is not supported. Choosing CU_AD_FORMAT_HALF.");
	case GL_FLOAT			:
	case GL_RGBA_FLOAT32_ATI:	m_cudaPixelType = CU_AD_FORMAT_HALF;break;//????
	case GL_DOUBLE:
		if(ProcessingGPU()->IsDoubleSupported())
		{
			m_cudaPixelType = CU_AD_FORMAT_FLOAT;
		}
		else
		{
			GPUCV_WARNING("Double format not supported. Choosing Half float.")
				m_cudaPixelType = CU_AD_FORMAT_HALF;
		}
		break;
	default :  GPUCV_ERROR("Critical : DataDsc_CUDA_Base::ConvertPixelType_GLToLocal()=> Unknown pixel type...Using GL_UNSIGNED_INT...");
		m_cudaPixelType = CU_AD_FORMAT_SIGNED_INT16;
		break;
	}

	if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
	{
		CLASS_FCT_SET_NAME("ConvertPixelType_GLToLocal");
		CLASS_DEBUG("Convert from " << GetStrGLTexturePixelType(_pixelType) << " to " << gcuGetStrPixelType((CUarray_format_enum)m_cudaPixelType));
	}
}
//==================================================
/*static virtual*/
GLuint	DataDsc_CUDA_Base::ConvertPixelType_LocalToGL(void)
{
	GLuint PixelType;
	switch (m_cudaPixelType)
	{
	case CU_AD_FORMAT_UNSIGNED_INT8:	PixelType = GL_UNSIGNED_BYTE;break;
	case CU_AD_FORMAT_SIGNED_INT8:		PixelType = GL_BYTE;break;
	case CU_AD_FORMAT_UNSIGNED_INT16:	PixelType = GL_UNSIGNED_SHORT;break;
	case CU_AD_FORMAT_SIGNED_INT16:		PixelType = GL_SHORT;break;
	case CU_AD_FORMAT_UNSIGNED_INT32:	PixelType = GL_UNSIGNED_INT;break;
	case CU_AD_FORMAT_SIGNED_INT32:		PixelType = GL_INT;break;
	case CU_AD_FORMAT_HALF:				PixelType = GL_FLOAT;break;
	case CU_AD_FORMAT_FLOAT:
		if(ProcessingGPU()->IsDoubleSupported())
		{
			PixelType = GL_DOUBLE;
		}
		else
		{
			GPUCV_WARNING("Double format not supported. Choosing float.")
				PixelType = GL_FLOAT;
		}
		break;
	default :  GPUCV_ERROR("Critical : DataDsc_CUDA_Base::ConvertPixelType_LocalToGL()=> Unknown pixel type...Using GL_UNSIGNED_INT...");
		PixelType = GL_INT;
		break;
	}
	if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
	{
		CLASS_FCT_SET_NAME("ConvertPixelType_LocalToGL");
		CLASS_DEBUG("Convert from " << gcuGetStrPixelType((CUarray_format_enum)m_cudaPixelType) << " to " << GetStrGLTexturePixelType(PixelType) );
	}
	return PixelType;
}
//==================================================
/*virtual*/
std::ostringstream & DataDsc_CUDA_Base::operator << (std::ostringstream & _stream)const
{
	DataDsc_Base::operator << (_stream);
	_stream << LogIndent() << "DataDsc_CUDA_Base==============" << std::endl;
	LogIndentIncrease();
	_stream << LogIndent() << "Pixel type:\t\t\t\t" << gcuGetStrPixelType((CUarray_format_enum)m_cudaPixelType) << std::endl;
	LogIndentDecrease();
	_stream << LogIndent() << "DataDsc_CUDA_Base==============" << std::endl;
	return _stream;
}
//==================================================
std::string DataDsc_CUDA_Base::PrintMemoryInformation(std::string text)const
{
	if(GetParent() && GetParent()->GetOption(DataContainer::DBG_IMG_MEMORY))
	{
		std::string Msg;
		Msg = "\n____________________________________";
		Msg += "\n" + GetParent()->GetValStr();
		Msg += "\n" + text;
		Msg += "\n____________________________________";
		Msg += "\nLocal data size: \t"	+	SGE::ToCharStr(m_dataSize);
		Msg += "\nLocal memory size: \t"+	SGE::ToCharStr(m_memSize);
		Msg += "\nTotal memory allocated in all DD*: \t"+	SGE::ToCharStr(ms_totalMemoryAllocated);
		Msg += "\nLocal memory allocated in this DD: \t"+	SGE::ToCharStr(m_localMemoryAllocated);
#if 0
		unsigned int TotalMem=0;
		unsigned int FreeMem=0;
		cuMemGetInfo(&FreeMem,&TotalMem);
		//CLASS_NOTICE("");
		Msg += "Cuda free memory:" + SGE::ToCharStr(FreeMem) + "(" + SGE::ToCharStr((double)FreeMem/TotalMem*100.) + "%)";
		if(FreeMem==0 || TotalMem==0)
		{
			GPUCV_NOTICE("We don't have free memory..??");
		}
#endif
		Msg += "\n____________________________________\n";
		Msg += "____________________________________\n";
		return Msg;
	}
	return "";
}
//==================================================
void  DataDsc_CUDA_Base::Flush(void)
{
	 DataDsc_Base::Flush();
	 //gcudaThreadSynchronize();	 
	GCU_CUDA_SAFE_CALL( cudaThreadSynchronize() );
}
//==================================================
//==================================================
//Local functions
//==================================================
//==================================================
//==================================================
//==================================================
#endif//_GPUCV_COMPILE_CUDA
