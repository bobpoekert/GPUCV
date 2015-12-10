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
#include <GPUCVTexture/DataDsc_GLBuff.h>
#include <GPUCVTexture/DataDsc_CPU.h>
#include <GPUCVTexture/DataDsc_GLTex.h>

namespace GCV{
//=========================================================
DataDsc_GLBuff::DataDsc_GLBuff()
:  DataDsc_GLBase()
,DataDsc_Base("DataDsc_GLBuff")
//,m_bufferType(ARRAY_BUFFER)
,m_bufferType(PIXEL_UNPACK_BUFFER)
,m_transferMode(STREAM_DRAW)
,m_pointerType(COLOR_POINTER)
{

}
//=========================================================
DataDsc_GLBuff::~DataDsc_GLBuff(void)
{
	Free();
}
//=========================================================
/*virtual*/
DataDsc_Base * DataDsc_GLBuff::CloneToNew(bool _datatransfer/*=true*/)
{
	DataDsc_GLBuff * TempBuff = new DataDsc_GLBuff();
	return TempBuff->Clone(this,_datatransfer);
}
//=========================================================
/*virtual*/
DataDsc_GLBuff*	DataDsc_GLBuff::Clone(DataDsc_GLBuff* _source, bool _datatransfer/*=true*/)
{
	GPUCV_DEBUG_IMG_TRANSFER("DataDsc_GLBuff::Cloning DataDsc_GLBuff");
	GPUCV_WARNING("DataDsc_GLBuff::Cloning DataDsc_GLBuff=> NOT DONE!!!");

	//....copy buffers here...//
	SG_Assert(0,"Clone buffer not done yet");
	DataDsc_Base::Clone(_source, _datatransfer);
	return this;
}
//=========================================================
/*virtual*/
//! \todo Copy to GL Texture
bool DataDsc_GLBuff::CopyTo(DataDsc_Base* _destination, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_destination->GetDscType(),"CopyTo");
	TEXTUREDESC_COPY_START(_destination,_datatransfer);

	//copy to buffer
	DataDsc_GLBuff * TempGLbuff = dynamic_cast<DataDsc_GLBuff *>(_destination);
	if(TempGLbuff)
	{//clone buffer
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_GLBuff", "DD_GLBuff", this, TempGLbuff,_datatransfer);
		UnsetReshape();
		return TempGLbuff->Clone(this, _datatransfer)?true:false;
	}
	//====================

	//copy to CPU
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_destination);
	if(TempCPU)
	{//read data back to CPU
		UnsetReshape();
		TempCPU->TransferFormatFrom(this);
		TempCPU->Allocate();
		if(HaveData() &&_datatransfer)
		{
			_GPUCV_CLASS_GL_ERROR_TEST();
			_Bind();
			//glBufferDataARB()
			void * _data = glMapBuffer(_GetType(), GL_READ_ONLY);
			CLASS_ASSERT(_data, "DataDsc_GLBuff::CopyTo(): empty source data from buffer");
			CLASS_ASSERT(*TempCPU->_GetPixelsData(), "DataDsc_GLBuff::CopyTo(): empty destination data");
			memcpy(*TempCPU->_GetPixelsData(), _data, GetMemSize());
			glUnmapBuffer(_GetType());
			//glReadPixels(0, 0, _GetWidth(), _GetHeight(), GetPixelFormat(), GetPixelType(), *TempCPU->_GetPixelsData());
			CLASS_DEBUG("");
			_UnBind();
			_GPUCV_CLASS_GL_ERROR_TEST();
			return true;
		}
	}
	//====================


	//don't know how to copy...must be done in another object
	return false;
}
//=========================================================
/*virtual*/
bool DataDsc_GLBuff::CopyFrom(DataDsc_Base* _source, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_source->GetDscType(),"CopyFrom");
	TEXTUREDESC_COPY_START(_source,_datatransfer);

	//copy from texture
	DataDsc_GLBuff * TempGL = dynamic_cast<DataDsc_GLBuff *>(_source);
	if(TempGL)
	{//clone texture
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_GLBuff", "DD_GLBuff", TempGL, this,_datatransfer);
		//Allocate();//???????
		return Clone(TempGL, _datatransfer)?true:false;
	}
	//====================

	//copy from CPU => load data into Buffer
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_source);
	if(TempCPU)
	{
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_GLBuff", "DD_CPU", TempCPU, this,_datatransfer);
		CLASS_DEBUG("");
		UnsetReshape();
		TransferFormatFrom(_source);
		_SetTransferMode(DataDsc_GLBuff::DYNAMIC_COPY/*DYNAMIC_DRAW*/);
		Allocate();
		_Bind();
		if(_datatransfer && TempCPU->HaveData())
		{
			_Writedata(/*NULL*/(const GCV::DataDsc_Base::PIXEL_STORAGE_TYPE **)TempCPU->_GetPixelsData(), GetMemSize(), _datatransfer);
			//void * _data = glMapBufferARB(_GetType(), GL_WRITE_ONLY);
			//CLASS_ASSERT(_data, "DataDsc_GLBuff::CopyFrom(): empty destination data from buffer");
			//CLASS_ASSERT(*TempCPU->_GetPixelsData(), "DataDsc_GLBuff::CopyFrom(): empty source data");
			//memcpy(_data,*TempCPU->_GetPixelsData(), GetMemorySize());
			//glUnmapBufferARB(_GetType());
		}
		else//only allocation
		{
			_Writedata(NULL, GetMemSize(), false);
		}
		_UnBind();
		SetDataFlag(true);
		return true;
	}
	//====================


	//don't know how to copy...must be done in another object
	return false;
}
//=========================================================
/*virtual*/
void DataDsc_GLBuff::Allocate(void)
{
	CLASS_FCT_SET_NAME("Allocate");
	CLASS_FCT_PROF_CREATE_START();
	DataDsc_Base::Allocate();
	if(!IsAllocated())
	{
		CLASS_DEBUG("");
		_GPUCV_CLASS_GL_ERROR_TEST();
		glGenBuffers(1, &m_GLId);
		Log_DataAlloc(m_memSize);
		_GPUCV_CLASS_GL_ERROR_TEST();
		//	glBufferDataARB(m_bufferType, GetMemSize(), 0, m_transferMode);
		CLASS_ASSERT(GetGLId(), "DataDsc_GLTex::Allocate()=> Allocation failed");
	}
}

//=========================================================
/*virtual*/
void DataDsc_GLBuff::Free()
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("Allocate");
	//no Bench and log in Destructor//	CLASS_FCT_PROF_CREATE_START();
	if(IsAllocated())
	{
		//no Bench and log in Destructor//		CLASS_DEBUG("");
		glDeleteBuffers(1,&m_GLId);
		Log_DataFree(m_memSize);
		DataDsc_Base::Free();
	}
}
//=========================================================
/*virtual*/
bool DataDsc_GLBuff::IsAllocated()const
{
	return (GetGLId()>0);
}
//=========================================================

//=========================================================
//=========================================================
//Local functions
//=========================================================
//=========================================================
/*virtual*/
void DataDsc_GLBuff::_Bind(void)const
{
	_GPUCV_CLASS_GL_ERROR_TEST();
	glBindBuffer( _GetType(), GetGLId());
	_GPUCV_CLASS_GL_ERROR_TEST();
}
//=========================================================
/*virtual*/
void DataDsc_GLBuff::_UnBind(void)const
{
	glBindBuffer( _GetType(), 0);
}
//========================================================
GLuint DataDsc_GLBuff::_GetType()const
{
	return m_bufferType;
}
//========================================================
void DataDsc_GLBuff::_SetType(GLuint _TexType)
{
	m_bufferType = (BufferType)_TexType;
}
//========================================================
void	DataDsc_GLBuff::_SetTransferMode(TransferMode _mode)
{
	m_transferMode = _mode;
}
//========================================================
DataDsc_GLBuff::TransferMode	DataDsc_GLBuff::_GetTransferMode()
{
	return m_transferMode;
}
//========================================================
void DataDsc_GLBuff::_Writedata(const PIXEL_STORAGE_TYPE ** _data, GLsizei _size, bool _datatransfer)
{
	CLASS_FCT_SET_NAME("_Writedata");
	_GPUCV_CLASS_GL_ERROR_TEST();
	CLASS_ASSERT(GetGLId(), "No buffer allocated");
	const PIXEL_STORAGE_TYPE * data = (_data)? *_data:NULL;
	CLASS_DEBUG("Decription:\n"<< * this);
	glBufferDataARB(_GetType(), _size, data, _GetTransferMode());
	_GPUCV_CLASS_GL_ERROR_TEST();
}
//========================================================
void DataDsc_GLBuff::_DrawArrays(GLenum _mode, GLint _first, GLsizei _count)
{
	_GPUCV_CLASS_GL_ERROR_TEST();
	glDrawArrays(_mode, _first, _count);
	_GPUCV_CLASS_GL_ERROR_TEST();
}
//========================================================
void DataDsc_GLBuff::_MultiDrawArrays(GLenum _mode, GLint *_first, GLsizei *_count, GLsizei _primcount)
{
	_GPUCV_CLASS_GL_ERROR_TEST();
	glMultiDrawArrays(_mode, _first, _count, _primcount);
	_GPUCV_CLASS_GL_ERROR_TEST();
}
//========================================================
void
DataDsc_GLBuff::_ReadData(GLuint _x/*=0*/, GLuint _y/*=0*/, GLuint _width/*=0*/, GLuint _height/*=0*/)
{
	_GPUCV_CLASS_GL_ERROR_TEST();
	if(_width*_height !=0)
		glReadPixels(_x, _y, _width, _height, GetPixelFormat(), GetPixelType(), NULL);
	else
		glReadPixels(_x, _y, _GetWidth(), _GetHeight(), GetPixelFormat(), GetPixelType(), NULL);

	_GPUCV_CLASS_GL_ERROR_TEST();
}
//========================================================
std::ostringstream & DataDsc_GLBuff::operator << (std::ostringstream & _stream)const
{
	_stream << std::endl;
	_stream << std::endl;
	DataDsc_GLBase::operator <<(_stream);
	_stream << LogIndent() <<"DataDsc_GLBuff==============" << std::endl;
	LogIndentIncrease();
	_stream << LogIndent() <<"m_bufferType: \t\t\t\t"			;
	switch(m_bufferType)
	{
		case GL_ARRAY_BUFFER:			_stream << "GL_ARRAY_BUFFER";break;
		case GL_ELEMENT_ARRAY_BUFFER:	_stream << "GL_ELEMENT_ARRAY_BUFFER";break;
		case GL_PIXEL_PACK_BUFFER:		_stream << "GL_PIXEL_PACK_BUFFER";break;
		case GL_PIXEL_UNPACK_BUFFER:	_stream << "GL_PIXEL_UNPACK_BUFFER";break;
		default:						_stream << m_bufferType <<"(Unknown type)";break;
	}
	_stream << std::endl;
	_stream << LogIndent() <<"m_transferMode: \t\t\t\t";
	switch(m_transferMode )
	{
		case GL_STREAM_DRAW:			_stream << "GL_STREAM_DRAW";break;
		case GL_STREAM_READ:			_stream << "GL_STREAM_READ";break;
		case GL_STREAM_COPY:			_stream << "GL_STREAM_COPY";break;
		case GL_STATIC_DRAW:			_stream << "GL_STATIC_READ";break;
		case GL_STATIC_COPY:			_stream << "GL_STATIC_COPY";break;
		case GL_DYNAMIC_DRAW:			_stream << "GL_DYNAMIC_DRAW";break;
		case GL_DYNAMIC_READ:			_stream << "GL_DYNAMIC_DREAD";break;
		case GL_DYNAMIC_COPY:			_stream << "GL_DYNAMIC_COPY";break;
		default:
			_stream << m_bufferType <<"(Unknown type)";break;
	}
	_stream << std::endl;
	_stream << LogIndent() <<"m_pointerType: \t\t\t\t"		<< m_pointerType << std::endl;
	LogIndentDecrease();
	_stream << LogIndent() <<"DataDsc_GLBuff==============" << std::endl;
	return _stream;
}

}//namespace GCV
