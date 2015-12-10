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
//		?????????????????????????????
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
/**
*	\brief Header file containg definitions for the GPU equivalent of ???????????? functions.
*	\author ????
*/
#ifndef __GPUCV_PLUGIN_TEMPLATE_H
#define __GPUCV_PLUGIN_TEMPLATE_H
#include <plugin_template/config.h>
//include some default GpuCV headers
#ifdef __cplusplus
#	include <GPUCV/misc.h>
#	include <GPUCVHardware/GlobalSettings.h>
#endif


//?? Declare your new operators here ?? like:
/*
_GPUCV_PLUGIN_TEMPLATE_EXPORT_C
void cvgXXXAdd( CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask CV_DEFAULT(NULL));
_GPUCV_PLUGIN_TEMPLATE_EXPORT_C
void cvgXXXDilate(CvArr* src, CvArr* dst, IplConvKernel* element, int iterations );
*/

#endif
