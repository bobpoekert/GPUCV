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
/**
*	\brief Header file containg definitions for the GPU equivalent OpenCV/Highgui functions.
*	\author Jean-Philippe Farrugia
*	\author Yannick Allusse
*/
#ifndef __GPUCV_GCVNPP_H
#define __GPUCV_GCVNPP_H
#include <gcvnpp/config.h>
#ifdef __cplusplus
#	include <GPUCV/misc.h>
#	include <GPUCVHardware/GlobalSettings.h>
#endif


//NVPP reference =============================================================
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @ingroup CVGXCORE_OPER_ARITHM_LOGIC_COMP_GRP
*  @{
*/

_GPUCV_GCVNPP_EXPORT_C
void cvgNppAdd( CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask CV_DEFAULT(NULL));
_GPUCV_GCVNPP_EXPORT_C
void cvgNppSub( CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask CV_DEFAULT(NULL));

_GPUCV_GCVNPP_EXPORT_C
void cvgNppDilate(CvArr* src, CvArr* dst, IplConvKernel* element, int iterations );

_GPUCV_GCVNPP_EXPORT_C
void cvgNppCanny(CvArr* image, CvArr* edges, double threshold1, double threshold2, int aperture_size CV_DEFAULT(3));
/** @}*///HIGHGUI__LOAD_SAVE_IMG_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @ingroup HIGHGUI__VIDEO_IO_GRP GPUCV
*  @{
/** @}*///HIGHGUI__VIDEO_IO_GRP
//_______________________________________________________________
//_______________________________________________________________

#endif
