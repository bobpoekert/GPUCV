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
#include "GPUCVTexture/TextureRecycler.h"

namespace GCV{
TextureRecycler *			TextureRecycler::m_instance = NULL;

//=================================================
/*static*/
GLuint TextureRecycler::GetNewTexture()
{
#if _GPUCV_TEXTURE_SUPPORT_TEXT_RECYCLING
	CreateInstance();

	GLuint ret;
	if (m_instance->m_freeTextures.size() > 0)
	{
		ret = m_instance->m_freeTextures[m_instance->m_freeTextures.size()-1];
		m_instance->m_freeTextures.pop_back();
		return ret;
	}
	else
	{
#endif

		glGenTextures(1,&ret);
		return ret;
#if _GPUCV_TEXTURE_SUPPORT_TEXT_RECYCLING
	}
#endif
}
//=================================================
/*static*/
bool			TextureRecycler::AddFreeTexture(GLuint _tex)
{
#if _GPUCV_TEXTURE_SUPPORT_TEXT_RECYCLING
	if (!_tex)
	{
		//GPUCV_WARNING("TextureRecycler::AddFreeTexture() => Freeing empty texture");
		return false;
	}

	CreateInstance();

	bool err=false;
	for (size_t  i=m_instance->m_freeTextures.size(); i--; )
	{
		if (m_instance->m_freeTextures[i] == _tex)
		{
			//GPUCV_WARNING("Warning : this texture was already marked as free.\n");
			err = true;
			break;
		}
	}

	if (!err)
		m_instance->m_freeTextures.push_back(_tex);

	return !err;
#else
	glDeleteTextures(_tex, 1);
	return true;
#endif

}
//=================================================
TextureRecycler::TextureRecycler()
{}
//=================================================
TextureRecycler::~TextureRecycler()
{}
//=================================================
TextureRecycler * TextureRecycler::CreateInstance()
{
#if _GPUCV_TEXTURE_SUPPORT_TEXT_RECYCLING
	if(m_instance == NULL)
		m_instance = new TextureRecycler();
	return m_instance;
#else
	return NULL;
#endif
}
//=================================================
}//namespace GCV

