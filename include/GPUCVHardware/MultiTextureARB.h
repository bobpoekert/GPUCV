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


#ifndef __GPUCV_HARDWARE_MULTI_TEXTURE_H
#define __GPUCV_HARDWARE_MULTI_TEXTURE_H

#include <GPUCVHardware/config.h>
namespace GCV{

/*-----------------------------------------------
MultiDataContainer.h
Written by Steven Jones
http://www.kraftwrk.com/multi_texturing.htm
-----------------------------------------------*/

// Constants ////////////////////////////////////////////////////////////////////////////

// Multi texture constants
#define GL_TEXTURE0_ARB                     0x84C0
#define GL_TEXTURE1_ARB                     0x84C1

#define GL_COMBINE_ARB						0x8570
#define GL_RGB_SCALE_ARB					0x8573


// Functions ////////////////////////////////////////////////////////////////////////////
typedef void (APIENTRY * PFNGLMULTITEXCOORD2FARBPROC)     (GLenum target, GLfloat s, GLfloat t);
typedef void (APIENTRY * PFNGLACTIVETEXTUREARBPROC)       (GLenum target);
typedef void (APIENTRY * PFNGLCLIENTACTIVETEXTUREARBPROC) (GLenum target);
}//namespace GCV
#endif
