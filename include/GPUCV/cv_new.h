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
/**	\file cv_new.h
* 	\brief Header file containg definitions for the CPU new OpenCV functions.
*	\author Yannick Allusse
*	\author David Gomez
*/
#ifndef __GPUCV_CV_NEW_H
#define __GPUCV_CV_NEW_H

#include <GPUCV/config.h>

#ifndef _MACOS
//!@addtogroup CVG_IMGPROC__CUSTOM_FILTER_GRP
//!@{
_GPUCV_EXPORT_C void cvDeriche(CvArr *src, CvArr *dst, float alpha);
_GPUCV_EXPORT_C void cvDericheDeriveX(CvArr* Ima, float alpha, CvMat* Y);
_GPUCV_EXPORT_C void cvDericheDeriveY(CvArr* Ima, float alpha, CvMat* Y);
_GPUCV_EXPORT_C void cvDericheLissageX(CvArr* Ima, CvArr *YImage, float alpha);
_GPUCV_EXPORT_C void cvDericheTranspose(CvArr *A, CvArr *AT);
_GPUCV_EXPORT_C void cvDericheLissageY(CvArr* Ima, CvArr* dst, float alpha);
_GPUCV_EXPORT_C void cvDericheExtrema(CvArr* B, CvArr* Gx, CvArr* Gy);
#endif // _MACOS
/**
*  \Brief - An operator which takes summed area tables as input and calculates local sum but on CPU
*  \param src1 -> The source summed area table.
*  \param dst ->  Output Image.
*  \param height -> The scalar which defines the height of the local area.
*  \param width -> The scalar which defines the Width of the local area.
*  \author Ankit Agarwal
*/
#if _GPUCV_DEVELOP_BETA
_GPUCV_EXPORT_C	
void  cvLocalSum(CvArr* src1 , CvArr* dst, int box_height , int box_width);
#endif
//!@} //CVG_IMGPROC__CUSTOM_FILTER_GRP
#endif
