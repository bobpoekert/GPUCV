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
#ifndef __GPUCV_HARDWARE_CONFIG_H
#define __GPUCV_HARDWARE_CONFIG_H

#include <definitions.h>

#ifdef _WINDOWS
#	ifdef _GPUCV_HARDWARE_DLL
#		define _GPUCV_HARDWARE_EXPORT		__declspec(dllexport)
#		define _GPUCV_HARDWARE_EXPORT_CLASS	__declspec(dllexport)
#	else
#		define _GPUCV_HARDWARE_EXPORT		__declspec(dllimport)
#		define _GPUCV_HARDWARE_EXPORT_CLASS	__declspec(dllimport)
#	endif
#elif defined MACOS
#pragma GCC visibility push(default)
#	ifdef _GPUCV_HARDWARE_DLL
#		define _GPUCV_HARDWARE_EXPORT		//__attribute__((__visibility__("default")))  //extern //"C"
#		define _GPUCV_HARDWARE_EXPORT_CLASS //__attribute__((__visibility__("default")))  //extern "C"
#	else
#		define _GPUCV_HARDWARE_EXPORT 		//__declspec(dllimport)
#		define _GPUCV_HARDWARE_EXPORT_CLASS //__declspec(dllimport)
#	endif
#else//Linux & MacOS
#	define _GPUCV_HARDWARE_EXPORT
#	define _GPUCV_HARDWARE_EXPORT_CLASS
#endif

#ifdef _MSC_VER
#	define __GPUCV_INLINE inline
#else
#	define __GPUCV_INLINE
#endif

#ifdef __cplusplus
	#define __GPUCV_THROW()throw()
#else
	#define __GPUCV_THROW()
#endif
#endif //__HARDWARE_CONFIG_H__
