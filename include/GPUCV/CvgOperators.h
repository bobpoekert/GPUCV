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
header file containing definitions for GpuCV functions in development.

Jean-Philippe Farrugia
Yannick Allusse
*/

#ifndef __GPUCV_OPERATORS_H
#define __GPUCV_OPERATORS_H
#include <GPUCV/misc.h>




_GPUCV_EXPORT_C void cvgTestGeometryShader(CvArr* src, CvArr* dst);
_GPUCV_EXPORT_C float cvgIsDiff(CvArr* src1, CvArr* src2);
//new CVG definitions :
//for cvDistTransform:
#define CVG_DIST_GEO 100 //does distance transform using geometry on GPU
#if _GPUCV_DEVELOP_BETA
_GPUCV_EXPORT_C void AcqAddRed( IplImage * A,  IplImage * B, IplImage* C);
_GPUCV_EXPORT_C void cvgHamed( IplImage* src, IplImage* dst);
_GPUCV_EXPORT_C void cvgGrabGLBuffer(IplImage* image, GLenum format, bool ForGpuUse=true);
//_GPUCV_EXPORT float cvgQueryHistValue(IplImage* src,int color);
//_GPUCV_EXPORT unsigned char cvgGetMaxValue( IplImage* src);
//_GPUCV_EXPORT unsigned char cvgGetMinValue( IplImage* src);





//==============================================================================

/** @defgroup CVG_NOT_VALIDATED_GRP GPUCV operators and functions waiting for beeing validated.
* This functions and operators are waiting to be validated. They were not validated yet for performance or stability issues. 
*  @{
*/

//performances issues
_GPUCV_EXPORT_C CvScalar AcqDistanceAvg(IplImage * carte, IplImage * mask);


_GPUCV_EXPORT_C void cvgCalcHistVBO( IplImage** img, CvHistogram* hist, int doNotClear=0,  CvArr* mask=0 );


#if _GPUCV_DEVELOP_BETA

_GPUCV_EXPORT_C void cvgConnectedComp(  IplImage* src, IplImage* dst);
_GPUCV_EXPORT_C void cvgConnectedComp2(  IplImage* src, IplImage* dst);
_GPUCV_EXPORT_C CvSeq* cvgHoughCircles(IplImage* image, void* circle_storage, int method, double dp, double min_dist, double param1, double param2);
_GPUCV_EXPORT_C void cvgCalcHist2(IplImage** src,CvHistogram* hist,GLuint liste, GLuint RGBA_Texture=0);
#endif

//under development
_GPUCV_EXPORT_C void cvgImageStat(  IplImage* src);
//_GPUCV_EXPORT_C void cvgDistTex(  IplImage* src, IplImage* dst, int distance_type=CV_DIST_L2, int mask_size=3,  float* mask=NULL, IplImage* labels=NULL );

/** @}*///CVG_INTERN_GRP
#endif

#endif//CVGPU_OPERATORS_H
