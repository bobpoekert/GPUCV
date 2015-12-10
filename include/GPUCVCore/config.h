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
#ifndef __GPUCV_CORE_CONFIG_H
#define __GPUCV_CORE_CONFIG_H

#include <GPUCVTexture/include.h>

#ifdef _WINDOWS
#ifdef _GPUCV_CORE_DLL
#define _GPUCV_CORE_EXPORT					__declspec(dllexport)
#else
#define _GPUCV_CORE_EXPORT			__declspec(dllimport)
#endif
#else
#define _GPUCV_CORE_EXPORT
#endif

//features options.
#define _GPUCV_CORE_USE_FULL_SHADER_NAME	1
#define _GPUCV_CORE_CHECK_SHADER_CHANGE		1 //! Used to reload shader when the files have been modified from outside of the program
#define _GPUCV_CORE_DEBUG_FILTER 			0&_GPUCV_DEBUG_MODE
//========================================


#define GPUCV_GET_TEX(CVARR)				GetTextureManager()->Get<CvgArr>(CVARR)
//#define GPUCV_GET_TEX(CVARR, LOCATION)		GetTextureManager()->Get<CvgArr>(CVARR, LOCATION)
#define GPUCV_GET_TEX_ON_LOC(CVARR, LOCATION, DATA)GetTextureManager()->Get<CvgArr>(CVARR, LOCATION, DATA)

/** @defgroup GPUCV_SHADER_GRP GpuCV GLSL group
	@ingroup GPUCV_SDK_GRP
*  @{
*  @}*/

#endif//CVGPU_CONFIG_H

