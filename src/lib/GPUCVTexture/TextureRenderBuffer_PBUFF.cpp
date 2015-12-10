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
#include <GPUCVTexture/TextureRenderBuffer_PBUFF.h>
#include <GPUCVTexture/DataDsc_GLTex.h>

namespace GCV{

TextureRenderBufferPBUFF::TextureRenderBufferPBUFF(bool verbose/*=false*/)
:TextureRenderBuffer(verbose)
{
	CL_Profiler::SetClassName("TextureRenderBufferPBO");
	m_type = TextureRenderBuffer::RENDER_OBJ_PBUFF;
}

TextureRenderBufferPBUFF::~TextureRenderBufferPBUFF()
{
	delete(PBuff32);
	delete(PBuff16);
}

bool TextureRenderBufferPBUFF::InitRenderBuff()
{
	_GPUCV_CLASS_GL_ERROR_TEST();

	if (!ProcessingGPU()->m_glExtension.IsPBufferCompatible())
		return false;

	PBuff32 = new PBuffer("float=32 rgba", false);
	PBuff16 = new PBuffer("float=16 rgba", false);
	PBuff16->Initialize(_GPUCV_TEXTURE_MAX_SIZE_X, _GPUCV_TEXTURE_MAX_SIZE_Y, true, true);
	PBuff32->Initialize(_GPUCV_TEXTURE_MAX_SIZE_X, _GPUCV_TEXTURE_MAX_SIZE_Y, true, true);

	_GPUCV_CLASS_GL_ERROR_TEST();

	if (PBuff32 && PBuff16)
		return true;

	return false;
}

void TextureRenderBufferPBUFF::SetContext(DataDsc_GLTex * _tex, int _width, int _height)
{
	_GPUCV_CLASS_GL_ERROR_TEST();

	SetValues(_tex, _width, _height);

	if (PBuffersManager()->IsActive() == 0)// if no PBffer were forced
	{
		if (!PBuffersManager()->IsUnique())
			PBuffersManager()->ActivateFirst();
	}
	_GPUCV_CLASS_GL_ERROR_TEST();
}

void TextureRenderBufferPBUFF::UnSetContext()
{
	_GPUCV_CLASS_GL_ERROR_TEST();
	if (PBuffersManager()->IsActive() == 0) // if not forced
	{
		if (!PBuffersManager()->IsUnique())
			PBuffersManager()->Desactivate();
	}
	TextureRenderBuffer::UnSetContext();
	_GPUCV_CLASS_GL_ERROR_TEST();
}


void TextureRenderBufferPBUFF::Force(DataDsc_GLTex * _tex, int _width, int _height)
{
	_GPUCV_CLASS_GL_ERROR_TEST();
	SetValues(_tex, _width, _height);
	PBuffersManager()->ForceFirst();
	_GPUCV_CLASS_GL_ERROR_TEST();
}


void TextureRenderBufferPBUFF::UnForce(void)
{
	_GPUCV_CLASS_GL_ERROR_TEST();
	PBuffersManager()->Desactivate();
	TextureRenderBuffer::UnForce();
	_GPUCV_CLASS_GL_ERROR_TEST();
}

int TextureRenderBufferPBUFF::IsForced(void)
{
	return PBuffersManager()->IsActive();
}


void TextureRenderBufferPBUFF::GetResult(void)
{
	_GPUCV_CLASS_GL_ERROR_TEST();
	glActiveTextureARB(GL_TEXTURE0);
	m_textID->_Bind();

	_GPUCV_CLASS_GL_ERROR_TEST();
	glCopyTexSubImage2D(GetHardProfile()->GetTextType(), 0, 0, 0, 0, 0,m_width,m_height);
	glTexParameteri(GetHardProfile()->GetTextType(),GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GetHardProfile()->GetTextType(),GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	_GPUCV_CLASS_GL_ERROR_TEST();
}

#if _GPUCV_USE_TEMP_FBO_TEX
GPUCV_TEXT_TYPE
TextureRenderBufferPBUFF::
SelectInternalTexture(RENDER_INTERNAL_OBJ _InternTextID)
{
	/*
	GPUCV_TEXT_TYPE ActiveText = NULL;

	switch(_InternTextID)
	{
	//case RENDER_FP8_RGBA :		ActiveText = TexFP16; break;
	//case RENDER_FP16_RGBA :		PBuff16->Activate();break;
	//case RENDER_FP32_RGBA_2 :	ActiveText = TexFP32_2;break;
	case RENDER_FP32_RGBA_1 :	PBuff32->Activate();break;
	//case RENDER_FP8_RGA :		ActiveText = TexFP8_RGB;break;
	//case RENDER_FP16_RGB :  break;
	//case RENDER_FP32_RGB :  break;
	default : break;
	}

	SetContext(ActiveText, _width, _height);

	*/	std::cout << std::endl << "TextureRenderBufferPBUFF::SelectInternalTexture() not done yet..." << std::endl;
return NULL;
}
#endif


}//namespace GCV

