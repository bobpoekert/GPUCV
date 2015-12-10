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
/**	\file toolscvg.h
*	\brief Tools functions for cvg operators.
*	\author Jean-Philippe Farrugia               
*	\author Yannick Allusse
*/
#ifndef __GPUCV_TOOLSCVG_H
#define __GPUCV_TOOLSCVG_H

#include <GPUCV/config.h>
#include <includecv.h>

#ifdef __cplusplus
#include <sys/stat.h>
#include <time.h>
#endif

namespace GCV{



template <typename TType>
TType * GetCVData(CvArr * arr)
{
	SG_Assert(arr, "Array empty");
	if(CV_IS_IMAGE_HDR(arr))
		return (TType*) ((IplImage*)arr)->imageData;
	else if (CV_IS_MAT(arr))
		return (TType*) ((CvMat*)arr)->data.ptr;
	else
		SG_Assert(0, "GetCVData()Unknown type");
	return NULL;
}

/**	\brief Return depth information from a CvArr.
*	\return Depth in OpenCV format.
*	\note If arr is of type CvMat, it return type.*/
_GPUCV_EXPORT_C GLuint GetCVDepth(const CvArr * arr);
/**	\brief Return depth information from a CvArr.
*	\return Depth in OpenGL format.
*	\note If arr is of type CvMat, it return type.*/
_GPUCV_EXPORT_C GLuint GetGLDepth(const CvArr * arr);

/**	\brief Return width information from a CvArr.*/
_GPUCV_EXPORT_C GLuint GetWidth(const CvArr * arr);
/**	\brief Return height information from a CvArr.*/
_GPUCV_EXPORT_C GLuint GetHeight(const CvArr * arr);
/**	\brief Return channel information from a CvArr.
*	\warning 1 nChannels is return for CvMat type.*/
_GPUCV_EXPORT_C GLuint GetnChannels(const CvArr * arr);
/**	\brief Return channel sequence information from a CvArr.
*	\warning "R" nChannels is return for CvMat type.*/
_GPUCV_EXPORT_C const char * GetChannelSeq(const CvArr * arr);
/**	\brief Return a GpuCV size object from a CvArr.*/
_GPUCV_EXPORT GCV::GpuFilter::FilterSize GetSize(const CvArr * arr);
#if _GPUCV_DEPRECATED
GpuFilter::FilterSize *GetSize(const DataContainer * tex)
_GPUCV_EXPORT_C void cvgShowFrameBufferImage(const char* name, GLuint width, GLuint height, GLuint Format, GLuint PixelType);
#endif

#ifdef __cplusplus

#if _GPUCV_DEPRECATED
/*!
*	\brief get directly the openGL texture id of on image. Do conversion if necessary
*	\param img -> image to manage
*	\return none
*/
_GPUCV_EXPORT_C 
GCV::GPUCV_TEXT_TYPE GetTextureId(CvArr *img);
#endif

/*==============================================================================

Macros Definitions													

==============================================================================*/
#if _GPUCV_DEPRECATED
// macros used to check that destination image is different from source images
// and create new temp image if needed
#define __BackUpDstPointer(srcImg, dstImg, TempImgorig, FctName){\
	if (TestDstPointer(dstImg,srcImg)==false){\
	GPUCV_WARNING("Warning : Identical Src and Dst in '" << FctName << "'...creating temp image!(loose performance)\n");\
	TempImgorig = (IplImage *)dstImg;\
	dstImg=NULL;\
	dstImg = cvgCreateImage(cvGetSize(srcImg),srcImg->depth, srcImg->nChannels);\
	if (GetTextureManager()->IsCpuReturn(reinterpret_cast<const TextureManager::TypeIDPtr>(dstImg)))cvgSetCpuReturn((IplImage *)dstImg);\
   else cvgUnsetCpuReturn(dstImg);\
	}}

//other copy methods can be used, with rendering for example...???
#define __RestoreDstPointer(dstImg, TempImgorig){\
	if (TempImgorig!=NULL){\
	printf("__RestoreDstPointer()\n");\
	cvgCopy(dstImg, TempImgorig);\
	}}
_GPUCV_EXPORT_C bool TestDstPointer(const CvArr* dst,const CvArr* src1);
#endif



/*==============================================================================

OPENCV TOOLS                                                     

==============================================================================*/


_GPUCV_EXPORT_C int ConvertCvHaarFeatureIntoBuffer(CvHaarFeature* _inputHaarFeature, void * _outBuffer);
_GPUCV_EXPORT_C int ConvertCvHaarClassifierIntoBuffer(CvHaarFeature* _inputHaarClassifier, void * _outBuffer);
//return size of data written
_GPUCV_EXPORT_C int ConvertCvHaarStageClassifierIntoBuffer(CvHaarStageClassifier* _inputHaarStage, void * _outBuffer);
//haar tools....
_GPUCV_EXPORT_C IplImage * ConvertHaarClassifierIntoIplImage(CvHaarClassifierCascade* _inputHaar);

}//namespace GCV

#endif//cpp
#endif//#define CVGTOOLS_H
