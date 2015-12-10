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
#include <GPUCVTexture/TextureRenderBuffer_FBO.h>
#include <GPUCVTexture/DataDsc_GLTex.h>

namespace GCV{

//=================================================
TextureRenderBufferFBO::TextureRenderBufferFBO(bool verbose/*=false*/)
:TextureRenderBuffer(verbose)
{
	CL_Profiler::SetClassName("TextureRenderBufferFBO");
	CLASS_FCT_SET_NAME("TextureRenderBufferFBO");
	CLASS_FCT_PROF_CREATE_START();
	m_type = TextureRenderBuffer::RENDER_OBJ_FBO;
}
//=================================================
TextureRenderBufferFBO::~TextureRenderBufferFBO()
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("~TextureRenderBufferFBO");
	//no Bench and log in Destructor//	CLASS_FCT_PROF_CREATE_START();
#if _GPUCV_USE_TEMP_FBO_TEX
	delete TexFP16;
	delete TexFP32_1;
	delete TexFP32_2;
	delete TexFP8_RGB;
#endif
}
//=================================================
bool TextureRenderBufferFBO::InitRenderBuff()
{
	CLASS_FCT_SET_NAME("InitRenderBuff");
	CLASS_FCT_PROF_CREATE_START();
	_GPUCV_CLASS_GL_ERROR_TEST();
	if (!GetHardProfile()->IsFBOCompatible())
		return false;

	//save env before init
	PushAttribs();

	FBOManager()->init();
#if _GPUCV_USE_TEMP_FBO_TEX
	TexFP8_RGB = new DataContainer();
	DataDsc_GLTex * DD_GLTex = TexFP8_RGB->GetDataDsc<DataDsc_GLTex>();
	DD_GLTex->SetFormat(GL_RGB, GL_UNSIGNED_BYTE);
	//DD_GLTex->_SetInternalPixelFormat(GL_RGB8);
	DD_GLTex->_SetSize(_GPUCV_TEXTURE_MAX_SIZE_X, _GPUCV_TEXTURE_MAX_SIZE_Y);
	TexFP8_RGB->SetLabel("FBO temp text RGB8");
	DD_GLTex->Allocate();


	TexFP16 = new DataContainer();
	DD_GLTex = TexFP16->GetDataDsc<DataDsc_GLTex>();
	DD_GLTex->SetFormat(GL_RGBA, GL_FLOAT);
	//DD_GLTex->_SetInternalPixelFormat(GL_RGBA16F_ARB);
	DD_GLTex->_SetSize(512, 512);
	TexFP16->SetLabel("FBO temp text RGBA16");
	DD_GLTex->Allocate();


	//float texture
	GLuint floatType;
	if(MainGPU()->m_glExtension.IsFloat32Compatible())
		floatType = GL_RGBA32F_ARB;
	else
		floatType = GL_RGBA16F_ARB;

	TexFP32_1 = new DataContainer();
	DD_GLTex = TexFP32_1->GetDataDsc<DataDsc_GLTex>();
	DD_GLTex->SetFormat(GL_RGBA, GL_FLOAT);
	DD_GLTex->_SetInternalPixelFormat(floatType);
	DD_GLTex->_SetSize(_GPUCV_TEXTURE_MAX_SIZE_Y, _GPUCV_TEXTURE_MAX_SIZE_Y);
	TexFP32_1->SetLabel(std::string("FBO temp text RGBA32_1(") + GetStrGLTextureFormat(floatType) +")");
	DD_GLTex->Allocate();


	TexFP32_2 = new DataContainer();
	DD_GLTex = TexFP32_2->GetDataDsc<DataDsc_GLTex>();
	DD_GLTex->SetFormat(GL_RGBA, GL_FLOAT);
	DD_GLTex->_SetInternalPixelFormat(floatType);
	DD_GLTex->_SetSize(_GPUCV_TEXTURE_MAX_SIZE_Y, _GPUCV_TEXTURE_MAX_SIZE_Y);
	TexFP32_2->SetLabel(std::string("FBO temp text RGBA32_2(") + GetStrGLTextureFormat(floatType) +")");
	DD_GLTex->Allocate();
#endif
	_GPUCV_CLASS_GL_ERROR_TEST();

	//restore GL env after init
	PopAttribs();
#if _GPUCV_USE_TEMP_FBO_TEX
	if(TexFP32_1 && TexFP32_2 && TexFP16 && TexFP8_RGB)
#endif
		return true;

	return false;
}
//=================================================
void TextureRenderBufferFBO::SetContext(DataDsc_GLTex * _tex, int _width, int _height)
{
	CLASS_FCT_SET_NAME("SetContext(DataContainer)");
	CLASS_FCT_PROF_CREATE_START();
	PushAttribs();

	_GPUCV_CLASS_GL_ERROR_TEST();
	SetValues(_tex, _width, _height);

	if (FBOManager()->IsForced() == NO_FORCED)
	{
		FBOManager()->SetTexture(_tex->GetGLId(), m_width, m_height, _tex->_GetTexType());
	}
	_GPUCV_CLASS_GL_ERROR_TEST();
}
//=================================================
/*virtual*/
int  TextureRenderBufferFBO::SetContext(TextureGrp * _grp, int _width, int _height)
{
	CLASS_FCT_SET_NAME("SetContext(TextureGrp)");
	CLASS_FCT_PROF_CREATE_START();
	PushAttribs();
	SG_Assert(_grp, "TextureRenderBufferFBO::Force() => No texture group!");

	_GPUCV_CLASS_GL_ERROR_TEST();

	//loop into the texture grp and link them to the FBO
	int i = 0;//nbr of texture linked
	DataDsc_GLTex * GLTexture=NULL;
	TEXTURE_GRP_EXTERNE_DO_FOR_ALL(
		_grp,
		TEXT,
		GLTexture = TEXT->GetDataDsc<DataDsc_GLTex>();
	FBOManager()->AddAttachedTexture(GLTexture->GetGLId(), _width, _height, GLTexture->_GetTexType(), GLTexture->_GetColorAttachment());
	);
	//==================================================

	FBOManager()->ForceMultiTexture();

	_GPUCV_CLASS_GL_ERROR_TEST();
	return i;
}
//=================================================
void TextureRenderBufferFBO::UnSetContext()
{
	CLASS_FCT_SET_NAME("UnSetContext");
	CLASS_FCT_PROF_CREATE_START();
#if _GPUCV_DEBUG_FBO
	std::cout << "TextureRenderBufferFBO::UnsetContext()" << std::endl;
#endif

	_GPUCV_CLASS_GL_ERROR_TEST();
	if (FBOManager()->IsForced() == NO_FORCED)
		FBOManager()->UnsetTexture();

	PopAttribs();
	TextureRenderBuffer::UnSetContext();
	_GPUCV_CLASS_GL_ERROR_TEST();
}
//=================================================
/*virtual*/
void TextureRenderBufferFBO::Force(DataDsc_GLTex * _tex, int _width, int _height)
{
	CLASS_FCT_SET_NAME("Force(DataContainer)");
	CLASS_FCT_PROF_CREATE_START();
#if _GPUCV_DEBUG_FBO
	std::cout << "TextureRenderBufferFBO::Force(" <<_tex->_GetWidth()<< ", "<<_tex->_GetHeight() << ")" << std::endl;
	if(_tex->GetParentOption(DataContainer::DBG_IMG_FORMAT))
		_tex->Print();
#endif
	PushAttribs();

	_GPUCV_CLASS_GL_ERROR_TEST();
	SetValues(_tex, _width, _height);
	FBOManager()->ForceTexture(_tex->GetGLId(), m_width, m_height, _tex->_GetTexType());
	_GPUCV_CLASS_GL_ERROR_TEST();
}
//=================================================
int  TextureRenderBufferFBO::Force(TextureGrp * _grp,int _width, int _height)
{
	CLASS_FCT_SET_NAME("Force(TextureGrp)");
	CLASS_FCT_PROF_CREATE_START();
#if _GPUCV_DEBUG_FBO
	std::cout << "TextureRenderBufferFBO::ForceMultiTexture()" << std::endl;
#endif
	PushAttribs();
	SG_Assert(_grp, "TextureRenderBufferFBO::Force() => No texture group!");

	_GPUCV_CLASS_GL_ERROR_TEST();

	//loop into the texture grp and link them to the FBO
	int i = 0;//nbr of texture linked
	DataDsc_GLTex * GLTexture = NULL;
	TEXTURE_GRP_EXTERNE_DO_FOR_ALL(
		_grp,
		TEXT,
		GLTexture = TEXT->GetDataDsc<DataDsc_GLTex>();
	FBOManager()->AddAttachedTexture(GLTexture->GetGLId(),
		_width, _height, GLTexture->_GetColorAttachment());
	);
	//==================================================

	FBOManager()->ForceMultiTexture();

	_GPUCV_CLASS_GL_ERROR_TEST();
	return i;
}
//=================================================
void TextureRenderBufferFBO::UnForce(void)
{
	CLASS_FCT_SET_NAME("UnForce");
	CLASS_FCT_PROF_CREATE_START();
	_GPUCV_CLASS_GL_ERROR_TEST();

	FBOManager()->UnForce();
	PopAttribs();
	TextureRenderBuffer::UnForce();
	_GPUCV_CLASS_GL_ERROR_TEST();
}
//=================================================
int TextureRenderBufferFBO::IsForced(void)
{
	return FBOManager()->IsForced();
}
//=================================================
std::ostringstream & TextureRenderBufferFBO::operator << (std::ostringstream & _stream)const
{
	_stream << LogIndent() <<"TextureRenderBufferFBO==============" << std::endl;
	LogIndentIncrease();
	TextureRenderBuffer::operator <<(_stream);
	FBOManager()->operator <<(_stream);
	LogIndentDecrease();
	_stream << LogIndent() <<"TextureRenderBufferFBO==============" << std::endl;
	return _stream;
}

//=================================================
#if _GPUCV_USE_TEMP_FBO_TEX
GPUCV_TEXT_TYPE
TextureRenderBufferFBO::
SelectInternalTexture(RENDER_INTERNAL_OBJ _InternTextID)
{
	GPUCV_TEXT_TYPE ActiveText = NULL;

	switch(_InternTextID)
	{
		//case RENDER_FP8_RGBA :		ActiveText = TexFP16; break;
	case RENDER_FP16_RGBA :		ActiveText = TexFP16;break;
	case RENDER_FP32_RGBA_2 :	ActiveText = TexFP32_2;break;
	case RENDER_FP32_RGBA_1 :	ActiveText = TexFP32_1;break;
	case RENDER_FP8_RGB :		ActiveText = TexFP8_RGB;break;
		//case RENDER_FP16_RGB :  break;
		//case RENDER_FP32_RGB :  break;
	default : break;
	}
	return ActiveText;
}
//=================================================
#endif
}//namespace GCV


