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
#include <GPUCVCuda/DataDsc_CUDA_Array.h>

#if _GPUCV_COMPILE_CUDA

#include <GPUCVCuda/GPU_NVIDIA_CUDA.h>
#include <GPUCVTexture/DataDsc_GLTex.h>
#include <GPUCVTexture/DataContainer.h>
#include <GPUCVTexture/DataDsc_GLBuff.h>
#include <GPUCVTexture/DataDsc_CPU.h>

using namespace GCV;
//==================================================
DataDsc_CUDA_Array::DataDsc_CUDA_Array()
: DataDsc_CUDA_Base()
,DataDsc_Base("DataDsc_CUDA_Array")
//,m_deviceDataPtr(NULL)
,m_pitch(0)
//,m_deviceDataType(CUDA_NO_TYPE)
//,m_textureArrayPtr(NULL)
,m_textureChannelDesc(NULL)
{
}
//==================================================
DataDsc_CUDA_Array::~DataDsc_CUDA_Array(void)
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("~DataDsc_CUDA_Array");
	//no Bench and log in Destructor//	CLASS_DEBUG("");
	Free();
}
//=======================================================
/*virtual*/
std::string DataDsc_CUDA_Array::LogException(void)const
{
	return DataDsc_CUDA_Base::LogException();
}
//==================================================
/*virtual*/
bool DataDsc_CUDA_Array::CopyTo(DataDsc_Base* _destination, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_destination->GetDscType(),"CopyTo");
	TEXTUREDESC_COPY_START(_destination,_datatransfer);

	//copy to CUDA buffer
	DataDsc_CUDA_Array * TempCUDA = dynamic_cast<DataDsc_CUDA_Array *>(_destination);
	if(TempCUDA)
	{//clone CUDA
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CUDA_ARRAY", "DD_CUDA_ARRAY", this, TempCUDA,_datatransfer);
		return (TempCUDA->Clone(this, _datatransfer))?true:false;
	}
	//====================

	//copy to CPU
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_destination);
	if(TempCPU)
	{//read data back to CPU
		CLASS_DEBUG("");
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CUDA_ARRAY", "DD_CPU", this, TempCPU,_datatransfer);
		UnsetReshape();
		TempCPU->TransferFormatFrom(this);
		TempCPU->Allocate();
		if(_datatransfer)
		{
			gcudaMemcpyFromArray(*TempCPU->_GetPixelsData(),m_data.m_textureArrayPtr, 0,0,m_memSize,cudaMemcpyDeviceToHost);
			//gcudaMemCopyDeviceToHost(*TempCPU->_GetPixelsData(), m_data.m_textureArrayPtr, m_memSize);
			CLASS_DEBUG("cudaMemcpyFromArray(*TempCPU->_GetPixelsData(), m_textureArrayPtr, 0,0," << ","<< m_memSize << "cudaMemcpyDeviceToHost)");
			TempCPU->SetDataFlag(true);
		}
		return true;
	}
	//====================

	//copy to GL...=======
	DataDsc_GLTex * TempGL = dynamic_cast<DataDsc_GLTex*>(_destination);
	if(TempGL)
	{
		CLASS_DEBUG("");
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CUDA_ARRAY", "DD_GLTEX", this, TempGL,_datatransfer);
		//CUDA can't manage openGL texture directly...using a GL buffer to make the copy
		if(GetParent())
		{
			DataDsc_GLBuff * TempGLBuffer = GetParent()->GetDataDsc<DataDsc_GLBuff>();
			if(this->CopyTo(TempGLBuffer, _datatransfer))
			{
				if(TempGL->CopyFrom(TempGLBuffer,_datatransfer))
				{
					TempGL->SetDataFlag(true);
					return true;
				}
			}
			else
				return false;
		}
		else
		{
			CLASS_ERROR("Object has no parent Container and a DataDsc_GLBuff is requested to transfer to DataDsc_GLTex");
			return false;
		}
	}
	//====================

	//copy to GL buffer===
	//======================


	//don't know how to copy...must be done in another object
	return false;
}
//==================================================
/*virtual*/
bool DataDsc_CUDA_Array::CopyFrom(DataDsc_Base* _source, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_source->GetDscType(),"CopyFrom");
	TEXTUREDESC_COPY_START(_source,_datatransfer);


	//GCU_CLASS_ASSERT(_source->HaveData() && , "DataDsc_CUDA_Array::CopyFrom()=> no input data");

	//copy from CUDA buffer
	DataDsc_CUDA_Array * TempCUDA = dynamic_cast<DataDsc_CUDA_Array *>(_source);
	if(TempCUDA)
	{//clone CUDA buffer
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CUDA_ARRAY", "DD_CUDA_ARRAY", TempCUDA, this,_datatransfer);
		return (Clone(TempCUDA, _datatransfer))?true:false;
	}
	//====================

	//copy from CPU
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_source);
	if(TempCPU)
	{
		CLASS_DEBUG("");
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CPU", "DD_CUDA_ARRAY", TempCPU, this,_datatransfer);
		UnsetReshape();
		TransferFormatFrom(_source);

		Allocate();
		if(_datatransfer && TempCPU->HaveData())
		{
			gcudaMemcpyToArray(m_data.m_textureArrayPtr, 0,0, *TempCPU->_GetPixelsData(),m_memSize,cudaMemcpyHostToDevice);
			//gcudaMemCopyHostToDevice(m_data.m_textureArrayPtr, *TempCPU->_GetPixelsData() , m_memSize);
			CLASS_DEBUG( "cudaMemcpyToArray(m_textureArrayPtr, 0,0,desc, *TempCPU->_GetPixelsData(),"<< m_memSize << ",cudaMemcpyHostToDevice)");
			SetDataFlag(true);
		}
		return true;
	}
	//====================

	//copy from GL...=======
	DataDsc_GLTex * TempGL = dynamic_cast<DataDsc_GLTex*>(_source);
	if(TempGL)
	{
		CLASS_DEBUG("");
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_GLTEX", "DD_CUDA_ARRAY", TempGL, this,_datatransfer);
		if(GetParent())
		{
			//CUDA can't manage openGL texture directly...using a GL buffer to make the copy
			DataDsc_GLBuff * TempGLBuffer = GetParent()->GetDataDsc<DataDsc_GLBuff>();
			if(TempGL->CopyTo(TempGLBuffer,_datatransfer))
				return this->CopyFrom(TempGLBuffer, _datatransfer);
			else
				return false;
		}
		else
		{
			CLASS_ERROR("Object has no parent Container and a DataDsc_GLBuff is requested to transfer from DataDsc_GLTex");
			return false;
		}
	}
	//====================

	//copy from GL buffer===
	//======================

	//don't know how to copy...must be done in another object
	return false;
}

//==================================================
/*virtual*/
DataDsc_Base * DataDsc_CUDA_Array::Clone(DataDsc_CUDA_Array * _src, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME("Clone");
	CLASS_FCT_PROF_CREATE_START();

	DataDsc_CUDA_Base::Clone(_src, _datatransfer);

	if(_datatransfer && _src->HaveData())
	{
		Allocate();
		cudaMemcpyArrayToArray((cudaArray *)_GetDataPtr(),0,0,(cudaArray *)_src->_GetDataPtr(), 0,0,m_memSize,cudaMemcpyDeviceToDevice);
		CLASS_DEBUG("cudaMemCopy(m_deviceDataPtr, _src->m_deviceDataPtr, "<<  m_memSize << ", cudaMemcpyDeviceToDevice)");
		SetDataFlag(true);
	}
	return this;
}
//==================================================
/*virtual*/
DataDsc_Base * DataDsc_CUDA_Array::CloneToNew(bool _datatransfer/*=true*/)
{
	DataDsc_CUDA_Array * TempTex = new DataDsc_CUDA_Array();
	return TempTex->Clone(this,_datatransfer);
}
//==================================================
/*virtual */
void DataDsc_CUDA_Array::_AllocateDevice(unsigned int _datatype)
{
	bool result=false;
	switch(_datatype)
	{
	case CU_AD_FORMAT_UNSIGNED_INT8:	result =_AllocateDataPtr<unsigned char>();break;
	case CU_AD_FORMAT_UNSIGNED_INT16:	result =_AllocateDataPtr<unsigned short>();break;
	case CU_AD_FORMAT_UNSIGNED_INT32:	result =_AllocateDataPtr<unsigned int>();break;
	case CU_AD_FORMAT_SIGNED_INT8:		result =_AllocateDataPtr<char>();break;
	case CU_AD_FORMAT_SIGNED_INT16:		result =_AllocateDataPtr<short>();break;
	case CU_AD_FORMAT_SIGNED_INT32:		result =_AllocateDataPtr<int>();break;
	case CU_AD_FORMAT_HALF:				result =_AllocateDataPtr<float>();break;
	case CU_AD_FORMAT_FLOAT:
		if(ProcessingGPU()->IsDoubleSupported())
		{
			result =_AllocateDataPtr<double>();
		}
		else
		{
			GPUCV_WARNING("Double format not supported. Choosing float.")
				result =_AllocateDataPtr<float>();
		}
		break;
	default :  GCU_Assert(0,"Critical : DataDsc_CUDA_Array::_AllocateDevice()=> Unknown pixel type......");
		break;
	}
	Log_DataAlloc(m_memSize);
	GCU_CLASS_ASSERT(result, "DataDsc_CUDA_Array::_AllocateDevice()=> Allocation failed");
}
//==================================================
/*virtual*/
void DataDsc_CUDA_Array::Free()
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("Free");
	//no Bench and log in Destructor//	CLASS_FCT_PROF_CREATE_START();

	if(IsAllocated())
	{
		//no Bench and log in Destructor//		CLASS_DEBUG("");
		gcudaFreeArray(m_data.m_textureArrayPtr);
		Log_DataFree(m_memSize);
		//no Bench and log in Destructor//		CLASS_DEBUG("->cudaFreeArray(m_textureArrayPtr)");
		m_data.m_textureArrayPtr=NULL;
		//YCK??
		//if(m_textureChannelDesc)
		//S	delete m_textureChannelDesc;
		DataDsc_CUDA_Base::Free();
	}
	
}
//==================================================
/*virtual*/
/*bool DataDsc_CUDA_Array::IsAllocated()const
{
if(m_textureArrayPtr)
return true;
}*/
//==================================================
/*virtual*/
std::ostringstream & DataDsc_CUDA_Array::operator << (std::ostringstream & _stream)const
{
	DataDsc_CUDA_Base::operator << (_stream);
	_stream << LogIndent() <<"DataDsc_CUDA_Array==============" << std::endl;
	LogIndentIncrease();
	_stream << LogIndent() <<"Pitch size: \t\t\t\t"<< m_pitch << std::endl;
	_stream << LogIndent() <<"Device memory type:\t\t\t\tCUDA_TEXTURE_ARRAY" << std::endl;
	_stream << LogIndent() <<"Device pointer(CUDA_ARRAY)\t\t\t" << (const long)m_data.m_textureArrayPtr << std::endl;
	if(m_textureChannelDesc)
	{
		_stream << LogIndent() <<"Texture channel descriptor:\t\t\t" << std::endl;
		_stream << LogIndent() <<"\tx:" << m_textureChannelDesc->x << std::endl;
		_stream << LogIndent() <<"\ty:" << m_textureChannelDesc->y << std::endl;
		_stream << LogIndent() <<"\tz:" << m_textureChannelDesc->z << std::endl;
		_stream << LogIndent() <<"\tw:" << m_textureChannelDesc->w << std::endl;
		_stream << LogIndent() <<"\tf:" << m_textureChannelDesc->f << std::endl;
	}
	else
		_stream << LogIndent() <<"Texture channel descriptor:\t\t\t" << "empty" << std::endl;
	LogIndentDecrease();
	_stream << LogIndent() << "DataDsc_CUDA_Array==============" << std::endl;
	return _stream;
}
//==================================================
//==================================================
//Local functions
//==================================================
size_t  DataDsc_CUDA_Array::_GetPitch(void)const
{
	return m_pitch;
}
void DataDsc_CUDA_Array::_SetCudaChannelFormatDesc(cudaChannelFormatDesc * _channelDesc)
{
	CLASS_FCT_SET_NAME("_SetCudaChannelFormatDesc");

	if(m_textureChannelDesc != _channelDesc)
	{
		CLASS_DEBUG("Updating channelDesc");
		CLASS_DEBUG("new channels descriptor("<<_channelDesc->x<<","<<_channelDesc->y<<","<<_channelDesc->z<<","<<_channelDesc->w<<",)");
		m_textureChannelDesc = _channelDesc;
	}

}
//==================================================
cudaChannelFormatDesc* DataDsc_CUDA_Array::_GetCudaChannelFormatDesc(void)
{
	return m_textureChannelDesc;
}
//==================================================
#endif//_GPUCV_COMPILE_CUDA
