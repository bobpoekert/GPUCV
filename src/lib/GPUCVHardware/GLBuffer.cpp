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
#include <GPUCVHardware/GLBuffer.h>
#include <SugoiTools/exceptions.h>
#include <GPUCVHardware/GLContext.h>
#include <GPUCVHardware/GlobalSettings.h>

namespace GCV{
GLBuffer* GLBuffer::mst_bindedBuffer = NULL;

//=================================================
GLBuffer::GLBuffer(BufferType _type, GLBuffer::TransferMode _mode,  GLsizei _size, const GLvoid * _data)
: m_type(ARRAY_BUFFER),
m_size(0),
m_data(NULL),
m_bufferId(0),
m_transferMode(STREAM_DRAW),
m_pointerType(VERTEX_POINTER)
{
	_GPUCV_GL_ERROR_TEST();
	SetType(_type);
	Generate();
	Bind();
	SetTransferMode(_mode);
	SetData(_size, _data);
	_GPUCV_GL_ERROR_TEST();
}
//=================================================
GLBuffer::GLBuffer()
: m_type(ARRAY_BUFFER),
m_size(0),
m_data(NULL),
m_bufferId(0)
,m_pointerType(COLOR_POINTER)
{

}
//=================================================
GLBuffer::~GLBuffer()
{
	if(m_bufferId)
		glDeleteBuffers(1, &m_bufferId);
}
//=================================================
GLvoid GLBuffer::Generate()
{
	_GPUCV_GL_ERROR_TEST();
	glGenBuffers(1, &m_bufferId);
	SG_Assert(m_bufferId, "Could not create new OpenGL buffer");
	_GPUCV_GL_ERROR_TEST();
}
//=================================================
void GLBuffer::Bind()
{
	_GPUCV_GL_ERROR_TEST();
	if(m_bufferId)
	{
		glBindBuffer(m_type, m_bufferId);
		mst_bindedBuffer = this;
	}
	_GPUCV_GL_ERROR_TEST();
}
//=================================================
void GLBuffer::UnBind()
{
	glBindBuffer(m_type, 0);
	mst_bindedBuffer = NULL;
}
//=================================================
bool GLBuffer::IsBinded()
{
	return (mst_bindedBuffer == this)?
		true : false;
}
//=================================================

//====================
//accessors
//====================
void GLBuffer::SetType(GLBuffer::BufferType _type)
{
	m_type = _type;
}
//=================================================
GLBuffer::BufferType	GLBuffer::GetType()
{
	return m_type;
}
//=================================================
void	GLBuffer::SetTransferMode(TransferMode _mode)
{
	m_transferMode = _mode;
}
//=================================================
GLBuffer::TransferMode	GLBuffer::GetTransferMode()
{
	return m_transferMode;
}
//=================================================
void GLBuffer::SetData(GLsizei _size, const GLvoid * _data)
{
	_GPUCV_GL_ERROR_TEST();
	//if(!IsBinded())
	//	Bind();
	//SetTransferMode(_mode);
	m_data = _data;
	m_size =_size;
	glBufferData(GetType(), _size, _data, GetTransferMode());
	_GPUCV_GL_ERROR_TEST();
}
//=================================================
const GLvoid * GLBuffer::GetData()
{
	return m_data;
}
//=================================================
void GLBuffer::SetPointer(PointerType _pointerType, GLuint _pixelSize, GLuint _pointerPixelType, GLuint _offset, const GLvoid *pointer)
{
	_GPUCV_GL_ERROR_TEST();
	m_pointerType =_pointerType;
	if(m_pointerType == COLOR_POINTER)
	{
		glColorPointer(_pixelSize, _pointerPixelType, _offset, pointer);
	}
	else if (m_pointerType == VERTEX_POINTER)
	{
		glVertexPointer(_pixelSize, _pointerPixelType, _offset, pointer);
	}
	else if (m_pointerType == TEXTURE_POINTER)
	{
		glTexCoordPointer(_pixelSize, _pointerPixelType, _offset, pointer);
	}
	else
	{
		SG_Assert(0, "Unknown type");
	}
	_GPUCV_GL_ERROR_TEST();
}
//=================================================
void GLBuffer::Enable()
{
	_GPUCV_GL_ERROR_TEST();
	if(m_pointerType == COLOR_POINTER)
	{
		glEnableClientState(GL_COLOR_ARRAY);
	}
	else if (m_pointerType == VERTEX_POINTER)
	{
		glEnableClientState(GL_VERTEX_ARRAY);
	}
	else if (m_pointerType == TEXTURE_POINTER)
	{
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	}
	else
	{
		SG_Assert(0, "Unknown type");
	}
	_GPUCV_GL_ERROR_TEST();
}
//=================================================
void GLBuffer::Disable()
{
	_GPUCV_GL_ERROR_TEST();
	if(m_pointerType == COLOR_POINTER)
	{
		glDisableClientState(GL_COLOR_ARRAY);
	}
	else if (m_pointerType == VERTEX_POINTER)
	{
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	else if (m_pointerType == TEXTURE_POINTER)
	{
		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	}
	else
	{
		SG_Assert(0, "Unknown type");
	}
	_GPUCV_GL_ERROR_TEST();
}
//=================================================
void GLBuffer::DrawArrays(GLenum _mode, GLint _first, GLsizei _count)
{
	_GPUCV_GL_ERROR_TEST();
	glDrawArrays(_mode, _first, _count);
	_GPUCV_GL_ERROR_TEST();
}
//=================================================
void GLBuffer::MultiDrawArrays(GLenum _mode, GLint *_first, GLsizei *_count, GLsizei _primcount)
{
	_GPUCV_GL_ERROR_TEST();
	glMultiDrawArrays(_mode, _first, _count, _primcount);
	_GPUCV_GL_ERROR_TEST();
}
//=================================================
}//namespace GCV
