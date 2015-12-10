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
#include <GPUCVTexture/TextureRenderBuffer.h>
#include <GPUCVTexture/DataDsc_GLTex.h>


namespace GCV{
//#define BUFFER_OFFSET(i) ((char *)NULL + (i)) //for PBO
//#define _RENDER_BUFFER_DEBUG_VERBOSE

#ifdef _RENDER_BUFFER_DEBUG_VERBOSE
#define __Fct_RB_START(FctName){\
	printf("->%s Start\n", FctName);\
}

#define __Fct_RB_STOP(FctName){\
	printf("->%s Stop\n", FctName);\
}
#else
#define __Fct_RB_START(FctName)
#define __Fct_RB_STOP(FctName)
#endif


TextureRenderBuffer * TextureRenderBuffer::MainRenderer = NULL;
//=================================================
TextureRenderBuffer::TextureRenderBuffer(bool verbose/*=false*/)
:CL_Profiler("TextureRenderBuffer")
,m_textID(NULL)
,m_textureGrp(NULL)
{
}
//=================================================
TextureRenderBuffer::~TextureRenderBuffer()
{
}
//=================================================
/*virtual std::string TextureRenderBuffer::LogException(void)const
{

}*/
//=================================================
/*virtual*/
std::ostringstream & TextureRenderBuffer::operator << (std::ostringstream & _stream)const
{
	_stream << LogIndent() <<"======================================" << std::endl;
	_stream << LogIndent() <<"TextureRenderBuffer==============" << std::endl;
	LogIndentIncrease();
	std::string Type;
	switch(m_type)
	{
	case RENDER_OBJ_AUTO: Type = "RENDER_OBJ_AUTO";break;
	case RENDER_OBJ_OPENGL_BASIC: Type = "RENDER_OBJ_OPENGL_BASIC";break;
	case RENDER_OBJ_FBO: Type = "RENDER_OBJ_FBO";break;
	case RENDER_OBJ_PBUFF: Type = "RENDER_OBJ_PBUFF";break;
	}
	_stream << LogIndent() << "Type:" << Type << std::endl;
	//_stream << "DataDsc_Base::m_dscType => "		<< m_dscType << std::endl;
	if(m_textID)
	{
		_stream << LogIndent() << "Texture object:"  << std::endl;
		LogIndentIncrease();
		m_textID->operator <<(_stream);
		LogIndentIncrease();
	}
	if(m_textureGrp)
	{
		_stream << LogIndent() << "Texture group object" << std::endl;
		LogIndentIncrease();
		m_textureGrp->operator <<(_stream);
		LogIndentDecrease();
	}
	_stream << LogIndent() <<"m_width: \t"	<< m_width << std::endl;
	_stream << LogIndent() <<"m_height: \t"	<< m_height << std::endl;
	LogIndentDecrease();
	_stream << LogIndent() <<"TextureRenderBuffer==============" << std::endl;
	return _stream;
}
//=================================================
/*virtual*/
TextureRenderBuffer::RENDER_OBJ_TYPE
TextureRenderBuffer::GetType()
{return m_type;}
//=================================================
void TextureRenderBuffer::SetValues(DataDsc_GLTex * _tex)
{
	if(_tex)
		SetValues(_tex, _tex->_GetWidth(), _tex->_GetHeight());
	else
		SetValues(NULL, 0, 0);

}
//=================================================
void TextureRenderBuffer::SetValues(DataDsc_GLTex * _tex,int _width, int _height)
{
	m_textID =_tex;
	if(_tex)
	{
		m_width		= _width;
		m_height	= _height;

		//DataDsc_GLTex * GLtex = _tex->GetDataDsc<DataDsc_GLTex>();
		SG_Assert(MainGPU()->m_glExtension.CheckAttachmentFormat(_tex->_GetInternalPixelFormat(),  _tex->_GetColorAttachment()),
			"Color attachment mismatch with texture internal format");
	}
	else
	{
		m_width		= 0;
		m_height	= 0;
	}
}
//=================================================
/*virtual*/
void TextureRenderBuffer::GetResult(DataDsc_GLTex * _tex)
{
	SetValues(_tex);
	GetResult();
}
//=================================================
/*virtual*/
bool TextureRenderBuffer::InitRenderBuff()
{//nothing to to...
	return true;
}
//=================================================
/*virtual */
void TextureRenderBuffer::SetContext		(DataDsc_GLTex * _tex)
{
	SG_Assert(_tex, "No DataContainer");
	SetContext(_tex, _tex->_GetWidth(), _tex->_GetHeight());
}
//=================================================
void TextureRenderBuffer::SetContext		(DataDsc_GLTex * _tex, int _width, int _height)
{
	SG_Assert(_tex, "No DataContainer");
	SetValues(_tex, _width, _height);
}
//=================================================
/*virtual*/
int  TextureRenderBuffer::SetContext(TextureGrp * _grp, int _width, int _height)
{
	SG_Assert(0,"TextureRenderBuffer::SetContext() not supported yet by default TextureRenderBuffer.");
	return 0;
}
//=================================================
/*virtual*/
void TextureRenderBuffer::UnSetContext	()
{
	m_textID=NULL;
	m_textureGrp=NULL;
	SetValues(NULL);
}
//=================================================
/*virtual*/
void TextureRenderBuffer::Force(DataDsc_GLTex * _tex)
{
	SG_Assert(_tex, "No DataContainer");
	Force(_tex, _tex->_GetWidth(), _tex->_GetHeight());
}
//=================================================
void TextureRenderBuffer::Force(DataDsc_GLTex * _tex, int _width, int _height)
{
	SG_Assert(_tex, "No DataContainer");
	SetValues(_tex, _width, _height);
}
//=================================================
int  TextureRenderBuffer::Force(TextureGrp * _grp)
{
	SG_Assert(_grp, "No texture group");
	DataDsc_GLTex * DDGLTex= _grp->operator [](0)->GetDataDsc<DataDsc_GLTex>();
	Force(_grp, DDGLTex->_GetWidth(),DDGLTex->_GetHeight());
	SG_Assert(0,"TextureRenderBuffer::Force() not supported yet by default TextureRenderBuffer.");
	return 0;
}
//=================================================
int  TextureRenderBuffer::Force(TextureGrp * _grp, int _width, int _height)
{
	SG_Assert(0,"TextureRenderBuffer::Force() not supported yet by default TextureRenderBuffer.");
	return 0;
}
//=================================================
/*virtual*/
void TextureRenderBuffer::UnForce(void)
{
	m_textID=NULL;
	m_textureGrp=NULL;
	SetValues(0);
}
//=================================================
/*virtual*/
int  TextureRenderBuffer::IsForced(void)
{
	if (m_textID)
		return true;
	else
		return false;
}
//=================================================
/*virtual*/
void TextureRenderBuffer::GetResult(void)
{
	_GPUCV_CLASS_GL_ERROR_TEST();
	//DataDsc_GLTex * GLtex = m_textID->GetDataDsc<DataDsc_GLTex>();
	glCopyTexImage2D(m_textID->_GetTexType(),
		0,
		m_textID->GetPixelFormat(),
		0, 0,
		m_textID->_GetWidth(),
		m_textID->_GetHeight(),
		0);
	_GPUCV_CLASS_GL_ERROR_TEST();

}
//=================================================
/*virtual*/
#if _GPUCV_USE_TEMP_FBO_TEX
GPUCV_TEXT_TYPE TextureRenderBuffer::SelectInternalTexture(RENDER_INTERNAL_OBJ _InternTextID)
{
	GPUCV_WARNING("TextureRenderBuffer::SelectInternalTexture()=> Fct not supported");
	return NULL;
}
#endif
//=================================================
/*static*/
TextureRenderBuffer *
TextureRenderBuffer::GetMainRenderer()
{return MainRenderer;}
//=================================================
/*static*/
TextureRenderBuffer *
TextureRenderBuffer::SetMainRenderer(TextureRenderBuffer * _Renderer)
{return MainRenderer = _Renderer;}
//=================================================
void TextureRenderBuffer::PushAttribs()
{
	getGLContext()->PushAttribs();
}
//=================================================
void TextureRenderBuffer::PopAttribs()
{
	getGLContext()->PopAttribs();

}
//=================================================
bool TextureRenderBuffer::IsTextureBinded(DataDsc_GLTex* _tex)
{
	if(m_textID)
		if(_tex == m_textID)
			return true;
	if(m_textureGrp)
		if(m_textureGrp->IsTextureInGroup(_tex->GetParent()))
			return true;
	return false;
}
//=================================================

}//namespace GCV


