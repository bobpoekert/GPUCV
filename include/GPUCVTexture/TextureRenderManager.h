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



#ifndef __GPUCV_TEXTURE_RENDERBUFFER_MANAGER_H
#define __GPUCV_TEXTURE_RENDERBUFFER_MANAGER_H

#include "GPUCVTexture/TextureRenderBuffer.h"
namespace GCV{

/*!
*	\brief Create a TextureRenderBuffer object depending on specified options and hardware compatibilities.
*	\param _type => Define the type of TextureRenderBuffer, see TextureRenderBuffer::RENDER_OBJ_TYPE.
*	\param _verbose => If set to true, the type selected will be print to the console output.
*/
_GPUCV_TEXTURE_EXPORT __GPUCV_INLINE
TextureRenderBuffer * 
CreateRenderBufferManager(
						  TextureRenderBuffer::RENDER_OBJ_TYPE _type= TextureRenderBuffer::RENDER_OBJ_AUTO, 
						  bool _verbose=false
						  );

/*!
*	\brief Return the current TextureRenderBuffer object, if no one exists, create one by calling CreateRenderBufferManager().
*	\param _type => Define the type of TextureRenderBuffer, see TextureRenderBuffer::RENDER_OBJ_TYPE.
*	\param _verbose => If set to true, the type selected will be print to the console output.
*/
_GPUCV_TEXTURE_EXPORT __GPUCV_INLINE
TextureRenderBuffer* 
RenderBufferManager(
					TextureRenderBuffer::RENDER_OBJ_TYPE _type= TextureRenderBuffer::RENDER_OBJ_AUTO, 
					bool _verbose=false
					);
}//namespace GCV

#endif//_GPUCV_TEXTURE_RENDERBUFFER_H

