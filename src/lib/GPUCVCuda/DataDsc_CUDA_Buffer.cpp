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


#include <GPUCVCuda/DataDsc_CUDA_Buffer.h>
#if _GPUCV_COMPILE_CUDA

#include <GPUCVCuda/GPU_NVIDIA_CUDA.h>
#include <GPUCVTexture/DataDsc_GLTex.h>
#include <GPUCVTexture/DataContainer.h>
#include <GPUCVTexture/DataDsc_GLBuff.h>
#include <GPUCVTexture/DataDsc_CPU.h>

#include <cuda_gl_interop.h>

using namespace GCV;
//==================================================
DataDsc_CUDA_Buffer::DataDsc_CUDA_Buffer()
: DataDsc_CUDA_Base()
,DataDsc_Base("DataDsc_CUDA_Buffer")
//	,m_deviceDataPtr(NULL)
,m_pitch(0)
#if CUDA_VERSION > 2300
,	m_pGLResource(NULL)
#endif

{
	CLASS_FCT_SET_NAME("DataDsc_CUDA_Buffer");
	CLASS_DEBUG("");
}
//==================================================
DataDsc_CUDA_Buffer::~DataDsc_CUDA_Buffer(void)
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("~DataDsc_CUDA_Buffer");
	//no Bench and log in Destructor//	CLASS_DEBUG("");
	Free();
}
//==================================================
/*virtual*/
std::string DataDsc_CUDA_Buffer::LogException(void)const
{
	return DataDsc_CUDA_Base::LogException();
}
//==================================================
/*virtual*/
bool DataDsc_CUDA_Buffer::CopyTo(DataDsc_Base* _destination, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_destination->GetDscType(),"CopyTo");
	TEXTUREDESC_COPY_START(_destination,_datatransfer);

	if(_destination == GetLockedObj())
	{
		_destination->SetDataFlag(_datatransfer && HaveData() );
		return true;//no need to copy
	}

	//copy to CUDA buffer
	DataDsc_CUDA_Buffer * TempCUDA = dynamic_cast<DataDsc_CUDA_Buffer *>(_destination);
	if(TempCUDA)
	{//clone CUDA
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CUDA_BUFF", "DD_CUDA_BUFF", this, TempCUDA,_datatransfer);
		return (TempCUDA->Clone(this, _datatransfer))?true:false;
	}
	//====================

	//copy to CPU
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_destination);
	if(TempCPU)
	{//read data back to CPU
#if 1
		CLASS_DEBUG("Start");
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CUDA_BUFF", "DD_CPU", this, TempCPU,_datatransfer);
		UnsetReshape();
		TempCPU->TransferFormatFrom(this);
		TempCPU->Allocate();
		if(_datatransfer)
		{
			gcudaMemCopyDeviceToHost(*TempCPU->_GetPixelsData(), m_data.m_deviceDataPtr, m_memSize);
			CLASS_DEBUG("cudaMemCopyDeviceToHost(*TempCPU->_GetPixelsData(), m_deviceDataPtr ," <<m_memSize <<")");
			TempCPU->SetDataFlag(true);
		}
		CLASS_DEBUG("Stop");
#endif
		return true;
	}
	//====================

	//copy to GL...=======
	DataDsc_GLTex * TempGL = dynamic_cast<DataDsc_GLTex*>(_destination);
	if(TempGL)
	{
		CLASS_DEBUG("Start");
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CUDA_BUFF", "DD_GLTEX", this, TempGL,_datatransfer);
		//CUDA can't manage openGL texture directly...using a GL buffer to make the copy
		if( GetParent())
		{
#if CUDA_VERSION > 2300
			CLASS_DEBUG("");
			if(GetLockedObj())
			{
				GCU_CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &m_pGLResource, 0));
				bool result = TempGL->CopyFrom(GetLockedObj(), _datatransfer);
				GCU_CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &m_pGLResource, 0));
				return result;
			}
			else
				return _MapGLObject(TempGL, cudaGraphicsMapFlagsWriteDiscard);
#else
			DataDsc_GLBuff * TempGLBuffer = GetParent()->GetDataDsc<DataDsc_GLBuff>();
			if(this->CopyTo(TempGLBuffer, _datatransfer))
			{
				if(TempGL->CopyFrom(TempGLBuffer,_datatransfer))
				{
					TempGL->SetDataFlag(true);
					CLASS_DEBUG("Stop");
					_MapGLObject(TempGLBuffer);
					return true;
				}
				_MapGLObject(TempGLBuffer);
			}
			else
				return false;
#endif
		}
		else
		{
			CLASS_ERROR("Object has no parent Container and a DataDsc_GLBuff is requested to transfer to DataDsc_GLTex");
			return false;
		}
	}
	//====================

	//copy to GL buffer===
	DataDsc_GLBuff * TempGLBuffer = dynamic_cast<DataDsc_GLBuff*>(_destination);
	if(TempGLBuffer)
	{
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CUDA_BUFF", "DD_GLBUFF", this, TempGLBuffer,_datatransfer);
		return _MapGLObject(TempGLBuffer, cudaGraphicsMapFlagsWriteDiscard);
	}
	//======================

	//copy to CUDA Array===
	DataDsc_CUDA_Array * TempCUDAArray = dynamic_cast<DataDsc_CUDA_Array*>(_destination);
	if(TempCUDAArray)
	{
		CLASS_DEBUG("Start");
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CUDA_BUFF", "DD_CUDA_ARRAY", this, TempCUDAArray,_datatransfer);
		TempCUDAArray->TransferFormatFrom(this);
		TempCUDAArray->Allocate();

		if(_datatransfer)
		{
			cudaMemcpyToArray((cudaArray *)TempCUDAArray->_GetDataPtr(),0, 0, _GetDataPtr(), m_memSize,cudaMemcpyDeviceToDevice);
			//gcudaMemCopyDeviceToDevice(TempCUDAArray->_GetDataPtr(), _GetDataPtr(), m_memSize);
			GPUCV_DEBUG_CUDA(GetParent()->GetValStr() << "=>" <<FctName << "gcudaMemCopyDeviceToDevice(TempCUDAArray->_GetDataPtr(), _GetDataPtr() ," <<m_memSize <<")");
			TempCUDAArray->SetDataFlag(true);
		}
		CLASS_DEBUG("Stop");
		return true;
	}
	//======================


	//don't know how to copy...must be done in another object
	return false;
}
//==================================================
/*virtual*/
bool DataDsc_CUDA_Buffer::CopyFrom(DataDsc_Base* _source, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_source->GetDscType(),"CopyFrom");
	TEXTUREDESC_COPY_START(_source,_datatransfer);


	
	if(_source==GetLockedObj())
	{
		SetDataFlag(_datatransfer && _source->HaveData());
		return true;//no need
	}
	//GCU_CLASS_ASSERT(_source->HaveData() && , "DataDsc_CUDA_Buffer::CopyFrom()=> no input data");

	//copy from CUDA buffer
	DataDsc_CUDA_Buffer * TempCUDA = dynamic_cast<DataDsc_CUDA_Buffer *>(_source);
	if(TempCUDA)
	{//clone CUDA buffer
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CUDA_BUFF", "DD_CUDA_BUFF", TempCUDA, this,_datatransfer);
		return (Clone(TempCUDA, _datatransfer))?true:false;
	}
	//====================

	//copy from CPU
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_source);
	if(TempCPU)
	{
#if 1
		CLASS_DEBUG("");
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CPU", "DD_CUDA_BUFF", TempCPU, this,_datatransfer);
		UnsetReshape();
		TransferFormatFrom(_source);

		Allocate();
		if(_datatransfer && TempCPU->HaveData())
		{
			gcudaMemCopyHostToDevice(m_data.m_deviceDataPtr, *TempCPU->_GetPixelsData() , m_memSize);
			CLASS_DEBUG( "cudaMemCopyHostToDevice(m_deviceDataPtr, 0,0,desc, *TempCPU->_GetPixelsData(),"<< m_memSize << ",cudaMemcpyHostToDevice)");
			SetDataFlag(true);
		}
#endif
		return true;
	}
	//====================

	//copy from GL...=======
	DataDsc_GLTex * TempGL = dynamic_cast<DataDsc_GLTex*>(_source);
	if(TempGL)
	{
		CLASS_DEBUG("");
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_GLTEX", "DD_CUDA_BUFF", TempGL, this,_datatransfer);
#if CUDA_VERSION > 2300
		TransferFormatFrom(TempGL);
		Allocate();
		if(GetLockedObj())
		{
			GCU_CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &m_pGLResource, 0));
			bool result = TempGL->CopyTo(GetLockedObj(), _datatransfer);
			GCU_CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &m_pGLResource, 0));
			return result;
		}
		else
			return _MapGLObject(TempGL, cudaGraphicsMapFlagsReadOnly);
#else
		if( GetParent())
		{//CUDA can't manage openGL texture directly...using a GL buffer to make the copy
			if(TempGL->CopyTo(TempGLBuffer,_datatransfer))
			{
				if(this->CopyFrom(TempGLBuffer, _datatransfer))
				{
					_UnMapGLObject();//TempGLBuffer);
					return true;
				}
				_UnMapGLObject();//TempGLBuffer);
			}
			else
				return false;
		}
		else
		{
			CLASS_ERROR("Object has no parent Container and a DataDsc_GLBuff is requested to transfer from DataDsc_GLTex");
			return false;
		}
#endif
	}
	//====================

	//copy from GL buffer===
	DataDsc_GLBuff * TempGLBuffer = dynamic_cast<DataDsc_GLBuff*>(_source);
	if(TempGLBuffer)
	{
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_GLBUFF", "DD_CUDA_BUFF", TempGLBuffer, this,_datatransfer);
		CLASS_DEBUG("");
		return _MapGLObject(TempGLBuffer, cudaGraphicsMapFlagsReadOnly);
	}
	//======================

	//copy from CUDA Array===
	DataDsc_CUDA_Array * TempCUDAArray = dynamic_cast<DataDsc_CUDA_Array*>(_source);
	if(TempCUDAArray)
	{
		CLASS_DEBUG("");
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CUDA_ARRAY", "DD_CUDA_BUFF", TempCUDAArray, this,_datatransfer);
		TransferFormatFrom(TempCUDAArray);
		Allocate();

		if(_datatransfer)
		{
			cudaMemcpyFromArray(_GetDataPtr(),(cudaArray *)TempCUDAArray->_GetDataPtr(), 0,0,m_memSize,cudaMemcpyDeviceToDevice);
			//gcudaMemCopyDeviceToDevice(_GetDataPtr(), TempCUDAArray->_GetDataPtr(), m_memSize);
			CLASS_DEBUG("gcudaMemCopyDeviceToDevice(_GetDataPtr(), TempCUDAArray->_GetDataPtr() ," <<m_memSize <<")");
			SetDataFlag(true);
		}
		return true;
	}
	//======================

	//don't know how to copy...must be done in another object
	return false;
}

//==================================================
/*virtual*/
DataDsc_Base * DataDsc_CUDA_Buffer::Clone(DataDsc_CUDA_Buffer * _src, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME("Clone");
	CLASS_FCT_PROF_CREATE_START();

	DataDsc_CUDA_Base::Clone(_src, _datatransfer);

	if(_datatransfer && _src->HaveData())
	{
		Allocate();
		gcudaMemCopy(m_data.m_deviceDataPtr, _src->m_data.m_deviceDataPtr, m_memSize, cudaMemcpyDeviceToDevice);
		CLASS_DEBUG("cudaMemCopy(m_deviceDataPtr, _src->m_deviceDataPtr, "<<  m_memSize << ", cudaMemcpyDeviceToDevice)");
		SetDataFlag(true);
	}
	return this;
}
//==================================================
/*virtual*/
DataDsc_Base * DataDsc_CUDA_Buffer::CloneToNew(bool _datatransfer/*=true*/)
{
	DataDsc_CUDA_Buffer * TempTex = new DataDsc_CUDA_Buffer();
	return TempTex->Clone(this,_datatransfer);
}
//==================================================
/*virtual*/
void DataDsc_CUDA_Buffer::Free()
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("Free");
	//no Bench and log in Destructor//	CLASS_FCT_PROF_CREATE_START();

#ifdef _GPUCV_CUDA_SUPPORT_OPENGL
	if(GetLockedObj())//&& _GetDeviceDataType() == CUDA_GL_BUFFER)
	{//unmap existing GL buffer
		_UnMapGLObject();//m_mappedGLObj);
		//..??DataDsc_CUDA_Base::Free();
	}
	else if(IsAllocated())//if buffer was mapped, no need to free memory...
#else
	if(IsAllocated())
#endif
	{
		//no Bench and log in Destructor//		CLASS_DEBUG("");
		gcudaFree(m_data.m_deviceDataPtr);
		Log_DataFree(m_memSize);
		//no Bench and log in Destructor//		CLASS_DEBUG("->cudaFree(m_deviceDataPtr)");
		m_data.m_deviceDataPtr = NULL;
		DataDsc_CUDA_Base::Free();
	}

}
//==================================================
//==================================================
/*virtual*/
std::ostringstream & DataDsc_CUDA_Buffer::operator << (std::ostringstream & _stream)const
{
	DataDsc_CUDA_Buffer* pNonConst = (DataDsc_CUDA_Buffer*)this;
	DataDsc_CUDA_Base::operator << (_stream);
	_stream << LogIndent() << "DataDsc_CUDA_Buffer==============" << std::endl;
	LogIndentIncrease();
	_stream << LogIndent() << "GL buffer mapped: \t\t\t\t"<< ((pNonConst->GetLockedObj())?"true":"false") << std::endl;
	//_stream << LogIndent() << "Device pointer\t" << (unsigned char *)m_data.m_deviceDataPtr << std::endl;
	LogIndentDecrease();
	_stream << LogIndent() << "DataDsc_CUDA_Buffer==============" << std::endl;
	return _stream;
}

void DataDsc_CUDA_Buffer::PostProcessUpdate(void)
{
#ifdef _GPUCV_CUDA_SUPPORT_OPENGL
	if(GetLockedObj())
	{//unmap existing GL buffer
		if(GetParent())
		{
			DataDsc_GLBuff * TempGLBuffer = GetParent()->GetDataDsc<DataDsc_GLBuff>();
			_UnMapGLObject();//TempGLBuffer);
		}
	}
#endif
}

void DataDsc_CUDA_Buffer::PreProcessUpdate(void)
{

}

//==================================================
//==================================================
//Local functions
//==================================================
//==================================================
/*virtual*/
void DataDsc_CUDA_Buffer::_AllocateDevice(unsigned int _datatype)
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
	GCU_CLASS_ASSERT(result, "DataDsc_CUDA_Buffer::_AllocateDevice()=> Allocation failed");
}
//==============================
bool  DataDsc_CUDA_Buffer::_MapGLObject(DataDsc_GLBase *_pGLObj, enum cudaGraphicsMapFlags _flag)
{
	CLASS_FCT_SET_NAME("_MapGLObject");
#ifdef _GPUCV_CUDA_SUPPORT_OPENGL
	CLASS_ASSERT(_pGLObj, "No input object")

	
	//lock GL object so no nobody else can map it
	if(_pGLObj->Lock(this)==NULL)//this buffer is already mapped or locked by another object
	{
		CLASS_WARNING("_pGLObj GL object was locked by: ??");// << TempGLBuffer->m_lockedBy->GetValStr());
		CLASS_WARNING("Could not perform transfer");
		return false;
	}
	//free local data if any
	Free();


	DataDsc_GLBuff*  pGLbuff = dynamic_cast<DataDsc_GLBuff*>(_pGLObj);
	DataDsc_GLTex*  pGLTex = dynamic_cast<DataDsc_GLTex*>(_pGLObj);
	

	if((_flag == cudaGraphicsMapFlagsWriteDiscard)|| 
		(_flag == cudaGraphicsMapFlagsNone) )
	{//send format informations to GL OBJ
		_GPUCV_CLASS_GL_ERROR_TEST();
		_pGLObj->TransferFormatFrom(/*TempGLBuffer*/this);//BUg fixe??
		_pGLObj->Allocate();
		
		if(pGLbuff)
		{
			pGLbuff->_Bind();
			pGLbuff->_Writedata(NULL, GetMemSize(), false);//alloc buffer
		}
		else
		{
			if(pGLTex)
			{
				pGLTex->_Writedata(NULL, false);
			}
		}
	}
	else if(_flag == cudaGraphicsMapFlagsReadOnly)
	{//retrieve format informations from GL OBJ
		TransferFormatFrom(_pGLObj);
	}
	else//well, we do not know who is writting to who
	{
		CLASS_ASSERT(0, "Unknown transfer direction...");
	}		
	
	if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
	{
		GPUCV_NOTICE("Trying to map current object:");
		GPUCV_NOTICE(*this);
		
		GPUCV_NOTICE("Wih OpenGL object:");
		GPUCV_NOTICE(*_pGLObj);
	}

	//Map existing buffer with cuda
#if CUDA_VERSION > 2300
		_pGLObj->_UnBind();

		//CUDA 3.0 does not support 8Bit mapping, only float are supported.
		if((_pGLObj->GetPixelType() != GL_FLOAT)
			&&(_pGLObj->GetPixelType() != GL_UNSIGNED_BYTE))
		{
			CLASS_WARNING("CUDA 3.0 does support buffer mapping with float/byte only");
			_pGLObj->UnLock(this);//free GL obj
			return false;
		}

		size_t num_bytes=0;

		CLASS_DEBUG("cudaGraphicsGLRegisterBuffer("<< _pGLObj->GetGLId()<<")");
		if(pGLTex)
			GCU_CUDA_SAFE_CALL(cudaGraphicsGLRegisterImage(&m_pGLResource, pGLTex->GetGLId(), pGLTex->_GetTexType(), _flag));
		else if (pGLbuff)
			GCU_CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&m_pGLResource, pGLbuff->GetGLId(), _flag));
		else
		{
			CLASS_ASSERT(0, "_pGLObj has unknown type.")
		}
		CLASS_DEBUG("cudaGraphicsMapResources()");
		GCU_CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &m_pGLResource, 0));
		
		CLASS_DEBUG("cudaGraphicsResourceGetMappedPointer()");
		GCU_CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&m_data, &num_bytes, m_pGLResource));
		
		CLASS_ASSERT(num_bytes==_pGLObj->GetMemSize(), "cudaGraphicsResourceGetMappedPointer() could not map enought memory");
#else
		gcudaGLRegisterBufferObject(_pGLObj->GetGLId());
		gcudaGLMapBufferObject((void**)&m_data.m_deviceDataPtr, _pGLObj->GetGLId());
		CLASS_DEBUG("cudaGLRegisterBufferObject("<< _pGLObj->GetGLId()<<")");
		CLASS_DEBUG("cudaGLMapBufferObject(m_deviceDataPtr, "<< _pGLObj->GetGLId()<<")");
#endif
		SetLockedObj(_pGLObj);
		_pGLObj->SetDataFlag(true);
		_GPUCV_CLASS_GL_ERROR_TEST();
		return true;
#else
		CLASS_ERROR("_GPUCV_CUDA_SUPPORT_OPENGL is not defined, DataDsc_CUDA* could not use OpenGL");
#endif
	return false;
}
//=======================
void  DataDsc_CUDA_Buffer::_UnMapGLObject()//DataDsc_GLBase *_pGLObj)
{
#ifdef _GPUCV_CUDA_SUPPORT_OPENGL
		//CLASS_FCT_SET_NAME("_MapGLObject");
		if(!GetLockedObj())
		{
		//	CLASS_WARNING("DataDsc_GLBuff was not mapped with this object");
			return;
		}
		//CLASS_ASSERT(_pGLObj,"No input Obj");
		
		if(GetLockedObj()->Lock(this)==NULL)//this buffer is already mapped or locked by another object
		{
	//		CLASS_WARNING("DataDsc_GLBuff buffer object was locked by: ??");// << TempGLBuffer->m_lockedBy->GetValStr());
	//		CLASS_WARNING("Could not perform transfer");
			return;
		}
		
	#if CUDA_VERSION > 2300
		GCU_CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &m_pGLResource, 0));
		GCU_CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(m_pGLResource));
	#else
		gcudaGLUnmapBufferObject(TempGLBuffer->GetGLId());
		gcudaGLUnregisterBufferObject(TempGLBuffer->GetGLId());
	#endif
	//CLASS_DEBUG( "cudaGLUnmapBufferObject("<< TempGLBuffer->GetGLId()<<")");
	//	CLASS_DEBUG("cudaGLUnregisterBufferObject("<< TempGLBuffer->GetGLId()<<")");
		GetLockedObj()->UnLock(this);
		SetLockedObj(NULL);
		_GPUCV_CLASS_GL_ERROR_TEST();
		return;
#else
		CLASS_ERROR("_GPUCV_CUDA_SUPPORT_OPENGL is not defined, DataDsc_CUDA* could not use OpenGL");
#endif
}
#endif//_GPUCV_COMPILE_CUDA
