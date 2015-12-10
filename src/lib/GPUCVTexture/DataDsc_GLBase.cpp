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
#include <GPUCVTexture/DataContainer.h>
#include "GPUCVTexture/TextureRenderManager.h"

namespace GCV{
//==================================================
DataDsc_GLBase::DataDsc_GLBase()
: DataDsc_Base("DataDsc_GLBase")
,m_GLId(0)
{
}
//==================================================
DataDsc_GLBase::~DataDsc_GLBase(void)
{
	Free();
}
//==================================================
/* virtual*/
void DataDsc_GLBase::ConvertPixelFormat_GLToLocal(const GLuint _pixelFormat)
{
}
//==================================================
/*virtual*/
GLuint	DataDsc_GLBase::ConvertPixelFormat_LocalToGL(void)
{
	return m_glPixelFormat;
}
//==================================================
/*virtual*/
void	DataDsc_GLBase::ConvertPixelType_GLToLocal(const GLuint _pixelType)
{
	m_glPixelType = _pixelType;
}
//==================================================
/*static virtual*/
GLuint	DataDsc_GLBase::ConvertPixelType_LocalToGL(void)
{
	return m_glPixelType;
}
//==================================================
void  DataDsc_GLBase::Flush(void)
{
	 DataDsc_Base::Flush();
	 glFlush();
	 glFinish();
}
//==================================================

//==================================================
//==================================================
//Local functions
//==================================================
//==================================================
//==================================================
std::ostringstream & DataDsc_GLBase::operator << (std::ostringstream & _stream)const
{
	_stream << std::endl;
	_stream << std::endl;
	DataDsc_Base::operator <<(_stream);
	_stream << LogIndent() <<"DataDsc_GLBase==============" << std::endl;
	LogIndentIncrease();
	_stream << LogIndent() <<"m_GLId: \t\t\t\t"			<< m_GLId << std::endl;
	LogIndentDecrease();
	_stream << LogIndent() <<"DataDsc_GLBase==============" << std::endl;
	return _stream;
}
}//namespace GCV
