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
#ifndef __GPUCV_TEXTURE_CONFIG_H
#define __GPUCV_TEXTURE_CONFIG_H

#include <GPUCVHardware/include.h>

#ifdef _WINDOWS
#	ifdef _GPUCV_TEXTURE_DLL
#		define _GPUCV_TEXTURE_EXPORT		    __declspec(dllexport)
#	else
#		define _GPUCV_TEXTURE_EXPORT			__declspec(dllimport)
#	endif
#else//linux
#	define _GPUCV_TEXTURE_EXPORT
#endif

#define _GPUCV_USE_DATA_DSC 1
#define _GPUCV_MAX_TEXTURE_DSC 6
//#define USE_PBOjects 1
typedef GLvoid 				PIXEL_STORAGE_TYPE;//!< Define the type used to store the Texture data in ram.

// Compatibilities options	==========
#define _GPUCV_TEXTURE_SUPPORT_PBUFFER			1
#define _GPUCV_TEXTURE_SUPPORT_FBO				1
#define _GPUCV_TEXTURE_SUPPORT_TEXT_RECYCLING	1
//========================================

#endif //__GPUCV_TEXTURE_CONFIG_H
