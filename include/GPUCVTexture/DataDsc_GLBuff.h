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



#ifndef __GPUCV_TEXTURE_DATADSC_GLBUFF_H
#define __GPUCV_TEXTURE_DATADSC_GLBUFF_H

#include <GPUCVTexture/DataDsc_GLBase.h>
namespace GCV{

/**	\brief DataDsc_GLBuff class describes the data storage into OpenGL BUFFERS.
*	\sa CL_Profiler, DataDsc_Base, DataDsc_GLBase
*	\author Yannick Allusse
*/
class _GPUCV_TEXTURE_EXPORT DataDsc_GLBuff
	:public virtual DataDsc_GLBase
	//,virtual public DataDsc_Base
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
		ARRAY_BUFFER = GL_ARRAY_BUFFER,
		PIXEL_PACK_BUFFER = GL_PIXEL_PACK_BUFFER,
		PIXEL_UNPACK_BUFFER = GL_PIXEL_UNPACK_BUFFER
	};

protected:
	BufferType			m_bufferType;   //!< OpenGL buffer type, GL_PIXEL_PACK_BUFFER...
	//GLuint				m_size;
	TransferMode		m_transferMode;
	PointerType			m_pointerType;
public:
	/** \brief Default constructor. */	
	__GPUCV_INLINE
		DataDsc_GLBuff(void);
	/** \brief Default destructor. */
	__GPUCV_INLINE virtual
		~DataDsc_GLBuff(void);


	//Redefinition of global parameters manipulation functions

	//Redefinition of data parameters manipulation

	//Redefinition of data manipulation
	__GPUCV_INLINE virtual void Allocate(void);		
	__GPUCV_INLINE virtual bool IsAllocated(void)const; 
	__GPUCV_INLINE virtual void Free();

	//Redefinition of DataDsc_Base interaction with other objects
	virtual bool			CopyTo(DataDsc_Base* _destination, bool _datatransfer=true);
	virtual bool			CopyFrom(DataDsc_Base* _source, bool _datatransfer=true);
	virtual	DataDsc_GLBuff *Clone(DataDsc_GLBuff * _src, bool _datatransfer=true);
	virtual	DataDsc_Base *	CloneToNew(bool _datatransfer=true);

	//===========================================
	//local functions:
	//===========================================
	__GPUCV_INLINE virtual void _Bind(void)const;
	__GPUCV_INLINE virtual void _UnBind(void)const;
	__GPUCV_INLINE void			_SetType(GLuint _TexType);
	__GPUCV_INLINE GLuint 		_GetType()const;
	__GPUCV_INLINE void			_SetTransferMode(TransferMode _mode);
	__GPUCV_INLINE TransferMode	_GetTransferMode();


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
	void _DrawArrays(GLenum _mode, GLint _first, GLsizei _count);
	void _MultiDrawArrays(GLenum _mode, GLint *_first, GLsizei *_count, GLsizei _primcount);

	//	__GPUCV_INLINE GLenum	_GetInternalPixelFormat()const;
	//	__GPUCV_INLINE void	_SetInternalPixelFormat(const GLuint _internalPixelFormat);
	void	_Writedata(const PIXEL_STORAGE_TYPE ** _data, GLsizei _size, bool _datatransfer);
	void	_ReadData(GLuint _xmin=0, GLuint _xmax=0, GLuint _ymin=0, GLuint _ymax=0);

	virtual std::ostringstream & operator << (std::ostringstream & _stream)const;
};
}//namespace GCV
#endif//__GPUCV_TEXTURE_DATADSC_GLBUFF_H
