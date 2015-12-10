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
#ifndef __HIGHGUI_SWITCH_CONFIG_H
#define __HIGHGUI_SWITCH_CONFIG_H

#ifdef _WINDOWS
#	ifdef _HIGHGUI_SWITCH_DLL
#		define _HIGHGUI_SWITCH_EXPORT			__declspec(dllexport)
#		define _HIGHGUI_SWITCH_EXPORT_C			extern "C" _HIGHGUI_SWITCH_EXPORT
#	else
#		ifdef __cplusplus
#			define _HIGHGUI_SWITCH_EXPORT		__declspec(dllimport)
#			define _HIGHGUI_SWITCH_EXPORT_C		extern "C" _HIGHGUI_SWITCH_EXPORT
#		else
#			define _HIGHGUI_SWITCH_EXPORT		__declspec(dllimport)
#			define _HIGHGUI_SWITCH_EXPORT_C		_HIGHGUI_SWITCH_EXPORT
#		endif
#	endif
#else
#		ifdef __cplusplus
#			define _HIGHGUI_SWITCH_EXPORT		
#			define _HIGHGUI_SWITCH_EXPORT_C		extern "C"
#		else
#			define _HIGHGUI_SWITCH_EXPORT	
#			define _HIGHGUI_SWITCH_EXPORT_C	
#		endif
#endif


#endif//__HIHGGUI_SWITCH_CONFIG_H
