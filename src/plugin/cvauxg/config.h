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
#ifndef __GPUCV_CVAUXG_CONFIG_H
#define __GPUCV_CVAUXG_CONFIG_H

#ifdef __cplusplus
#include <GPUCVCore/include.h>
#endif

#ifdef _WINDOWS
#	ifdef _GPUCV_CVAUXG_DLL
#		define _GPUCV_CVAUXG_EXPORT			__declspec(dllexport)
#		define _GPUCV_CVAUXG_EXPORT_C			extern "C"  _GPUCV_CVAUXG_EXPORT
#	else
#		ifdef __cplusplus
#			define _GPUCV_CVAUXG_EXPORT			__declspec(dllimport)
#			define _GPUCV_CVAUXG_EXPORT_C			extern "C" _GPUCV_CVAUXG_EXPORT
#		else
#		define _GPUCV_CVAUXG_EXPORT				__declspec(dllimport)
#		define _GPUCV_CVAUXG_EXPORT_C			_GPUCV_CVAUXG_EXPORT
#		endif
#	endif
#else
#	define _GPUCV_CVAUXG_EXPORT
#	define _GPUCV_CVAUXG_EXPORT_C
#endif

#endif//__GPUCV_CVAUXG_CONFIG_H
