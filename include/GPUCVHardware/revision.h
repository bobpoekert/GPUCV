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
#ifndef __GPUCV_REVISION_H
#define __GPUCV_REVISION_H

#define _GPUCV_VERSION_MAJOR	"1.0"
#define _GPUCV_REVISION_VAL		"588"
#ifdef _WINDOWS
#	define _GPUCV_REVISION_DATE	""##__DATE__
#else
#	define _GPUCV_REVISION_DATE	__DATE__
#endif
#define _GPUCV_REVISION_RANGE	588
#define _GPUCV_REVISION_MIXED	0
#define _GPUCV_REVISION_URL		"svn+ssh://guest@picoforge.int-evry.fr/gpucv/experimental/trunk/gpucv/"
#define _GPUCV_WEB_URL			"https://picoforge.int-evry.fr/cgi-bin/twiki/view/Gpucv/Web/WebHome"

#endif
