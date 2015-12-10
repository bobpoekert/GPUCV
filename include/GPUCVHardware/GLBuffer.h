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
/**
\file GLBuffer.h
\author Yannick Allusse
*/

#ifndef __GPUCV_HARDWARE_GLBUFFER_H
#define __GPUCV_HARDWARE_GLBUFFER_H


#include <GPUCVHardware/ToolsGL.h>
namespace GCV{

/**
\brief Base class to support access to OpenGL buffers.
*/
class _GPUCV_HARDWARE_EXPORT_CLASS GLBuffer
{
public:
	enum TransferMode
	{//used for vertex buffers
		STREAM_DRAW		=	GL_STREAM_DRAW,
		STREAM_READ		=	GL_STREAM_READ,
		STREAM_COPY		=	GL_STREAM_COPY,
		DYNAMIC_DRAW	=	GL_DYNAMIC_DRAW,
		DYNAMIC_READ	=	GL_DYNAMIC_READ,
		DYNAMIC_COPY	=	GL_DYNAMIC_COPY,
		STATIC_DRAW		=	GL_STATIC_DRAW,
		STATIC_READ		=	GL_STATIC_READ,
		STATIC_COPY		=	GL_STATIC_COPY
	};

	enum PointerType
	{
		COLOR_POINTER,
		VERTEX_POINTER,
		TEXTURE_POINTER
	};

	enum BufferType{
		ARRAY_BUFFER = GL_ARRAY_BUFFER
	};

	GLBuffer(BufferType _type, TransferMode _mode, GLsizei _size,const GLvoid * _buffers);
	GLBuffer();
	~GLBuffer();

	GLvoid Generate();

	void Bind();
	void UnBind();
	bool IsBinded();

	void	SetType(BufferType _type);
	BufferType	GetType();

	void			SetTransferMode(TransferMode _mode);
	TransferMode	GetTransferMode();

	void		SetData(GLsizei _size, const GLvoid * _data);
	const GLvoid *	GetData();

	void SetPointer(PointerType _pointerType, GLuint _pixelSize, GLuint _pointerPixelType, GLuint _offset, const GLvoid *pointer);
	//void SetPointer(PointerType _pointerType, GLuint _pointerPixelType, GLuint _start=0, GLuint _offset =0);

	bool CheckHardwareCompatibility();

	void Enable();
	void Disable();

	/*!
	GL_POINTS,
	GL_LINE_STRIP,
	GL_LINE_LOOP,
	GL_LINES,
	GL_TRIANGLE_STRIP,
	GL_TRIANGLE_FAN,
	GL_TRIANGLES,
	GL_QUAD_STRIP,
	GL_QUADS
	GL_POLYGON
	*/
	void DrawArrays(GLenum _mode, GLint _first, GLsizei _count);
	void MultiDrawArrays(GLenum _mode, GLint *_first, GLsizei *_count, GLsizei _primcount);
	//DrawElements

protected:
	BufferType		m_type;
	GLuint			m_size;
	const GLvoid *	m_data;
	GLuint			m_bufferId;
	TransferMode	m_transferMode;
	PointerType		m_pointerType;
	static 	GLBuffer*	mst_bindedBuffer;
};

}//namespace GCV
#endif
