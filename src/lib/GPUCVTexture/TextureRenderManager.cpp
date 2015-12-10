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
#include "GPUCVTexture/TextureRenderManager.h"
#include "GPUCVTexture/TextureRenderBuffer_FBO.h"
#include "GPUCVTexture/TextureRenderBuffer_PBUFF.h"

#define TestGPGPU_FBO 0
#if TestGPGPU_FBO
#include "GPUCVTexture/TextureRenderBuffer_FBO_GPGPU.h"
#endif

namespace GCV{

//=================================================
TextureRenderBuffer * CreateRenderBufferManager(TextureRenderBuffer::RENDER_OBJ_TYPE _type/*=0*/, bool _verbose/*=false*/)
{
	TextureRenderBuffer * Renderer = NULL;

	if (_type== TextureRenderBuffer::RENDER_OBJ_AUTO)
	{//auto select

#if		TestGPGPU_FBO
		Renderer  = new TextureRenderBufferFBO_GPGPU(_verbose);
#else
		Renderer  = new TextureRenderBufferFBO(_verbose);
#endif
		if (Renderer->InitRenderBuff())
		{
			if (_verbose)std::cout << "FrameBufferObjects are compatible..." << endl;
		}
		else
		{
			if (_verbose)
				cout << "FrameBufferObjects are not available on your hardware configuration..." << endl;

			delete Renderer;
			Renderer  = new TextureRenderBufferPBUFF(_verbose);

			if (Renderer->InitRenderBuff())
			{
				if (_verbose)
					cout << "Using PBuffers instead" << endl;
			}
			else
			{
				cout << "PixelBuffer and FBO are compatible with you hardware..." << endl;
			}
		}
	}
	else
	{//force selection
		if (_type ==TextureRenderBuffer::RENDER_OBJ_FBO)
		{
			//type = _RENDER_BUFFER_TYPE_FBO;
#if		TestGPGPU_FBO
			Renderer  = new TextureRenderBufferFBO_GPGPU(_verbose);
#else
			Renderer  = new TextureRenderBufferFBO(_verbose);
#endif
			if (Renderer->InitRenderBuff())
			{
				if (_verbose)cout << "Forcing use of FrameBufferObjects..." << endl;
			}
			else
			{
				cout << "FrameBufferObjects are not available on your hardware configuration..." << endl;
				delete Renderer;
			}
		}
		else if (_type ==TextureRenderBuffer::RENDER_OBJ_PBUFF)
		{
			Renderer  = new TextureRenderBufferPBUFF(_verbose);
			if (Renderer->InitRenderBuff())
			{
				if (_verbose)cout << "Forcing use of PBuffers..." << endl;
			}
			else
			{
				cout << "PBuffers are not available on your hardware configuration..." << endl;
				delete Renderer;
			}
		}
		else if (_type ==TextureRenderBuffer::RENDER_OBJ_OPENGL_BASIC)
		{
			Renderer  = new TextureRenderBuffer(_verbose);
			Renderer->InitRenderBuff();
			if (_verbose)cout << "Forcing use of openGL basic render to texture..." << endl;
		}
		else
		{//nothing compatible
			cout << "Unkown TextureRendererType..." << endl;
		}
	}
	return Renderer;
}
//=================================================
TextureRenderBuffer * RenderBufferManager(TextureRenderBuffer::RENDER_OBJ_TYPE _type/*=0*/, bool _verbose/*=false*/)
{
	if(!TextureRenderBuffer::GetMainRenderer())
		return TextureRenderBuffer::SetMainRenderer(CreateRenderBufferManager(_type,_verbose));
	else
		return TextureRenderBuffer::GetMainRenderer();
}
//=================================================
}//namespace GCV

