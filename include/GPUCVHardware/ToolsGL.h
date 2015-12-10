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
\brief Header file containing some tools for openGL functions.
\author Yannick Allusse
*/
#ifndef __GPUCV_HARDWARE_TOOLSGL_H
#define __GPUCV_HARDWARE_TOOLSGL_H

#include <GPUCVHardware/config.h>
#include <GPUCVHardware/gcvGL.h>


#ifdef _WINDOWS
#include <windef.h>
#endif


namespace GCV{

#if defined MACOS
//! Default QUAD vertices.
_GPUCV_HARDWARE_EXPORT extern const float GPUCVDftQuadVertexes[4*3];
//! Default QUAD texture coordinates.
_GPUCV_HARDWARE_EXPORT extern const double GPUCVDftTextCoord[4*2];
#else
//! Default QUAD vertices.
_GPUCV_HARDWARE_EXPORT extern const float GPUCVDftQuadVertexes[4*3];
//! Default QUAD texture coordinates.
_GPUCV_HARDWARE_EXPORT extern const double GPUCVDftTextCoord[4*2];

#endif
/*
#ifndef GLsizei //MACOS..??
	#define GLsizei unsigned int
	#define GLuint unsigned int
#endif
*/

class TextureGrp;
/** \brief Class to store and manipulate texture/buffer size.
*/
template <typename _Type=GLsizei>
class TextSize
{
	friend class TextureGrp;
protected:
	_Type m_width;
	_Type m_height;
public:
	TextSize()
		:m_width(0), m_height(0)
	{}

	TextSize(_Type _w, _Type _h)
		:m_width(_w),m_height(_h)
	{}

	virtual ~TextSize()
	{};

	__GPUCV_INLINE
		virtual void _SetSize(_Type _w, _Type _h)
	{
		m_width  = _w;
		m_height = _h;
	}

	__GPUCV_INLINE
		virtual _Type _GetWidth()const
	{ return m_width;}

	__GPUCV_INLINE
		virtual _Type _GetHeight()const
	{ return m_height;}

	__GPUCV_INLINE
		virtual void _SetWidth(_Type _val)
	{ m_width = _val;};

	__GPUCV_INLINE
		virtual void _SetHeight(_Type _val)
	{ m_height = _val;};

	__GPUCV_INLINE
		virtual bool operator ==(const TextSize<_Type>& src2)
	{
		if(m_height != src2.m_height)
			return false;
		if(m_width != src2.m_width)
			return false;
		return true;
	}
	std::ostringstream & operator << (std::ostringstream & _stream)const
	{
		_stream << "w:" << m_width << "/h:" << m_height;
		return _stream;
	}
};

template <typename _Type>
std::ostringstream & operator << (std::ostringstream & _stream, const TextSize<_Type> & TexDsc)
{
	return TexDsc.operator << (_stream);
}

}//namespace GCV
#endif//#define CVGTOOLS_H
