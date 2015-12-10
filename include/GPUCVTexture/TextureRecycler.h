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
#ifndef __GPUCV_TEXTURE_RECYCLER_H
#define __GPUCV_TEXTURE_RECYCLER_H
/**
\author Yannick Allusse
*/
#include "GPUCVTexture/config.h"


namespace GCV{


/**
*	\brief This class it used to store create/free/store openGL texture.
It stores unused texture ID until they are requested by the library. If no textures are actually stored, they are created on request.
*/
class _GPUCV_TEXTURE_EXPORT TextureRecycler
{
public:
	/**
	*	\brief Get the first unused texture from the texture's vector. If no texture is available, create a new one.
	*	\return GLuint => ID of the texture.
	*/ 
	static GLuint GetNewTexture();

	/**
	*	\brief Set a OpenGL texture as unused and store it in the freeTexture manager
	*	\param  texid => OpenGL Texture id to store
	*	\return none
	*/ 
	static bool			AddFreeTexture(GLuint texid);

protected:
	static TextureRecycler *		m_instance; 	//!< Pointer to singleton.
	std::vector<GLuint>			m_freeTextures;		//!< Vector containing unused textures.

	/**
	*	\brief Constructor.
	*/
	TextureRecycler();

	/**
	*	\brief Destructor.
	*/
	~TextureRecycler();

	/**
	*	\brief CL_Singleton creator.
	*/
	static TextureRecycler * CreateInstance();
};
}//namespace GCV
#endif
