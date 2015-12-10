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
#ifndef __GPUCV_CXCOREG_H
#define __GPUCV_CXCOREG_H

/**	\brief Header file containg definitions for the GPU equivalent open CV functions
*	\author Jean-Philippe Farrugia
*	\author Yannick Allusse
*/

#include <cxcoreg/config.h>
#include <GPUCV/misc.h>

//CXCore reference =============================================================
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup  CVGXCORE_OPER_ARRAY_INIT_GRP
*  @{
*/
/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvCreateImage" target=new>cvCreateImage</a> function.
*  Create one IplImage. You can use cvCreateImage instead, but this version is safer with GPUCV library
*	\param size -> image size
*	\param depth -> image depth
*	\param channels -> image channels number
*	\return New IplImage pointer.
*	\author Jean-Philippe Farrugia
*/
_GPUCV_CXCOREG_EXPORT_C
IplImage* cvgCreateImage(CvSize size, int depth, int channels);

//cvCreateImageHeader
//cvReleaseImageHeader

/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvReleaseImage" target=new>cvReleaseImage</a> function.
*  Release one IplImage. You can use cvReleaseImage instead, but this version is safer with GPUCV library
*	\param img -> Double pointer to the header of the deallocated image.
*	\author Jean-Philippe Farrugia
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgReleaseImage(IplImage **img);

//cvInitImageHeader

/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvCloneImage" target=new>cvCloneImage</a> function.
*  Clone one IplImage. You can use cvCloneImage instead, but this version is safer with GPUCV library
*	\param img => image to manage
*	\author Jean-Philippe Farrugia
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
IplImage* cvgCloneImage(IplImage *img);

//cvSetImageCOI
//cvGetImageCOI
//cvSetImageROI
//cvResetImageROI
//cvGetImageROI
#if _GPUCV_SUPPORT_CVMAT
/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvCreateMat" target=new>cvCreatemat</a> function.
*  Create one CvMat. You can use cvCreateMat instead, but this version is safer with GPUCV library
*	\param rows -> Number of rows in the matrix.
*	\param cols -> Number of columns in the matrix.
*	\param type -> Type of the matrix elements. Usually it is specified in form CV_<bit_depth>(S|U|F)C<number_of_channels>, for example:
CV_8UC1 means an 8-bit unsigned single-channel matrix, CV_32SC2 means a 32-bit signed matrix with two channels.
*	\return New CvMat*.
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
CvMat* cvgCreateMat( int rows, int cols, int type );

//cvCreateMatHeader

/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvReleaseMat" target=new>cvReleaseMat</a> function.
*  Release one CvMat. You can use cvReleaseMat instead, but this version is safer with GPUCV library.
*	\param mat -> Double pointer to the matrix.
*	\author Yannick Allusse.
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgReleaseMat(CvMat **mat);

//cvInitMatHeader
//cvMat
/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvCloneMat" target=new>cvCloneMat</a> function.
*  The function cvCloneMat creates a copy of input matrix and returns the pointer to it.
*	\param mat => matrix to copy.
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
CvMat* cvgCloneMat(CvMat* mat );
#endif
//cvCreateMatND
//cvCreateMatNDHeader
//cvReleaseMatND
//cvInitMatNDHeader
//cvCloneMatND
//cvDecRefData
//cvIncRefData
//cvCreateData		*
//cvReleaseData		*
/*!
*   \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvSetData" target=new>cvSetData</a> function.
*   \param arr -> Array header.
*	\param data -> User data.
*	\param step -> Full row length in bytes.
*   \author Yannick Allusse.
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgSetData( CvArr* arr, void* data, int step );
/*!
*   \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvGetRowData" target=new>cvGetRowData</a> function.
*   \param arr -> Array header.
*	\param data -> Output pointer to the whole image origin or ROI origin if ROI is set.
*	\param step -> Output full row length in bytes.
*	\param roi_size -> Output ROI size.
*   \author Yannick Allusse.
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgGetRawData(CvArr* arr, uchar** data, int* step CV_DEFAULT(NULL), CvSize* roi_size CV_DEFAULT(NULL));
//cvGetMat
//cvGetImage
//cvCreateSparseMat
//cvReleaseSparseMat
//cvCloneSparseMat

/** @}*///CVGXCORE_OPER_ARRAY_INIT_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVGXCORE_OPER_ARRAY_ACCESS_ELEM_GRP
*  @{
\todo fill with OpenCV functions list.
*/
/** @}*///CVGXCORE_OPER_ARRAY_ACCESS_ELEM_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVGXCORE_OPER_COPY_FILL_GRP
*  @{
\todo fill with OpenCV functions list.
*/
/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvCopy" target=new>cvCopy</a> function.
*	\param src -> The source array.
*	\param dst -> The destination array.
*	\param mask -> Operation mask, 8-bit single channel array; specifies elements of destination array to be changed.
*	\todo Now cvgCopy use shader even if there is no mask, do a simple render to texture when no mask!!
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgCopy(CvArr* src, CvArr* dst , CvArr* mask CV_DEFAULT(NULL));
/*
cvSet
*/
/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvSetZero" target=new>cvSetZero</a> function.
*	\param arr -> The source array.
*	\note This operator is only a compatibility operator, it reads back image in all cases.
*	\todo Make a GPU version.
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgSetZero( CvArr* arr );
/*
cvSetZero
cvSetIdentity
cvRange
*/

/** @}*///CVGXCORE_OPER_COPY_FILL_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVGXCORE_OPER_TRANSFORM_PERMUT_GRP
*  @{
*/
//cvReshape
//cvReshapeMatND
//cvRepeat
#if 1//_GPUCV_DEVELOP_BETA
/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvFlip" target=new>cvFlip</a> function.
*  The function cvgFlip flip a 2D array around vertical, horizontall or both axis.
*	\param src -> Source array.
*	\param dst -> Destination array. If dst = NULL the flipping is done inplace.
*	\param flip_mode -> Specifies how to flip the array. flip_mode = 0 means flipping around x-axis, flip_mode > 0 (e.g. 1) means flipping around y-axis and flip_mode < 0 (e.g. -1) means flipping around both axises. See also the discussion below for the formulas
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgFlip(  CvArr* src, CvArr* dst, int flip_mode CV_DEFAULT(0));
#define cvgMirror cvgFlip
#endif//beta

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvSplit" target=new>cvSplit</a> function
*	\param src -> The first source array.
*	\param dst0 -> first destination channels.
*	\param dst1 -> second destination channels.
*	\param dst2 -> third destination channels.
*	\param dst3 -> fourth destination channels.
*	\author Jean-Philippe Farrugia
*	\author Yannick Allusse
*	\note Test have been done with multiple render target MRT and it is faster.
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgSplit(  CvArr* src, CvArr* dst0, CvArr* dst1, CvArr* dst2, CvArr* dst3 );
#define cvgCvtPixToPlane cvgSplit//for compatibilities issues...

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvMerge" target=new>cvMerge</a> function
*	\param src0 -> First input channel.
*	\param src1 -> Second input channel.
*	\param src2 -> Third input channel.
*	\param src3 -> Forth input channel.
*	\param dst -> Dstination image.
*	\author Yannick Allusse
*	\note Test have been done with multiple render target MRT and it is faster.
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgMerge(CvArr* src0, CvArr* src1, CvArr* src2, CvArr* src3, CvArr* dst );
//cvMixChannels
//cvRandShuffle
/** @}*///CVGXCORE_OPER_TRANSFORM_PERMUT_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVGXCORE_OPER_ARITHM_LOGIC_COMP_GRP
*  @{
*/
#if _GPUCV_DEVELOP_BETA
/**
*   \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvLut" target=new>cvLut</a> function.
*	\param src => Source array of 8-bit elements.
*	\param dst => Destination array of arbitrary depth and of the same number of channels as the source array.
*	\param lut => Look-up table of 256 elements; should have the same depth as the destination array. In case of multi-channel source and destination arrays, the table should either have a single-channel (in this case the same table is used for all channels), or the same number of channels as the source/destination array.
*   \author Yannick Allusse
*	\note cvgLUT is based on GL_ARB_imaging extension, wich is not hardware accelerated...
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgLUT(CvArr* src, CvArr* dst, CvArr* lut );
#endif
/**
*   \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvConvertScale" target=new>cvConvertScale</a> function.
*   \author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgConvertScale(CvArr* src, CvArr* dst, double scale CV_DEFAULT(1), double shift CV_DEFAULT(0));
#define cvgCvtScale cvgConvertScale
#define cvgScale	cvgConvertScale
#define cvgConvert( src, dst )  cvgConvertScale( (src), (dst), 1, 0 )

/*
cvConvertScaleAbs
*/
//KEYTAG: TUTO_CREATE_OP_BASE__STP4__WRITE_DOC
/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvAdd" target=new>cvAdd</a> function.
*	\param src1 -> The first source array.
*	\param src2 -> The second source array.
*	\param dst -> The destination array.
*	\param mask -> mask image (optionnal)
*	\author Jean-Philippe Farrugia
*	\author Yannick Allusse
*/
//KEYTAG: TUTO_CREATE_OP_BASE__STP2__LAUNCHER
_GPUCV_CXCOREG_EXPORT_C
void cvgAdd( CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask CV_DEFAULT(NULL));

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvAdds" target=new>cvAdds</a> function.
*   \param src -> The source array.
*   \param value -> Added scalar.
*   \param dst -> The destination array.
*   \param mask -> Operation mask, 8-bit single channel array; specifies elements of destination array to be changed.
*	\author Songbo.
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgAddS(CvArr * src, CvScalar value, CvArr* dst, CvArr* mask CV_DEFAULT(NULL));


/**
*  \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvCmp" target=new>cvCmp</a> function
*  \param src1	->	The first source array.
*  \param src2	->	The second source array,Both source must be single channel.
*  \param dst	->	Output destination array must have 8uor 8s type.
*  \param op	->	The parameter which decides which operator to apply.
*  \author Ankit Agarwal
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgCmp(CvArr * src1,CvArr * src2, CvArr* dst,int op);
/**
*  \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvCmpS" target=new>cvCmpS</a> function
*  \param src	->	The first source array.
*  \param value	->	Double value.
*  \param dst	->	Output destination array must have 8uor 8s type.
*  \param op	->	The parameter which decides which operator to apply.
*  \author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgCmpS(CvArr * src1, double value, CvArr* dst,int op);



/**
*  \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvIntegral" target=new>cvIntegral</a> function
*  \param src1 -> The source image.
*  \param dst -> Output destination image. The Source qnd Destination must be of same type.
*  \author Ankit Agarwal
*/
#if _GPUCV_DEVELOP_BETA
_GPUCV_CXCOREG_EXPORT_C
void cvgIntegral(IplImage* src1,IplImage* dst);
#endif
//cvAddWeighted

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvSub" target=new>cvSub</a> function.
*	\param src1 -> The first source array.
*	\param src2 -> The second source array.
*	\param dst -> The destination array.
*	\param mask -> mask image (optionnal)
*	\author Jean-Philippe Farrugia
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgSub( CvArr* src1,  CvArr* src2, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvSubS" target=new>cvSubS</a> function.
*   \param src -> The source array.
*   \param value -> Added scalar.
*   \param dst -> The destination array.
*   \param mask -> Operation mask, 8-bit single channel array; specifies elements of destination array to be changed.
*	\author Songbo.
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgSubS( CvArr * src, CvScalar value, CvArr* dst,  CvArr* mask CV_DEFAULT(NULL));

/*
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvSubRS" target=new>cvSubsRS</a> function.
*   \param src -> The source array.
*   \param value -> Added scalar.
*   \param dst -> The destination array.
*   \param mask -> Operation mask, 8-bit single channel array; specifies elements of destination array to be changed.
*	\author Songbo.
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgSubRS( CvArr * src, CvScalar value, CvArr* dst,  CvArr* mask);

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvMul" target=new>cvMul</a> function.
*  The function cvgMul calculates per-element product of two arrays: dst(I)=scale * src1(I) * src2(I)
*  All the arrays must have same data type and the same size (or ROI size).
*	\param src1 -> The first source array.
*	\param src2 -> The second source array.
*	\param dst  -> The destination array.
*	\param scale -> Optional scale factor
*	\return none
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgMul(  CvArr* src1,  CvArr* src2, CvArr* dst, double scale CV_DEFAULT(1) );

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvDiv" target=new>cvDiv</a> function.
*  The function cvDiv divides one array by another:
dst(I)=scale•src1(I)/src2(I), if src1!=NULL
dst(I)=scale/src2(I),      if src1=NULL
*  All the arrays must have same data type and the same size (or ROI size).
*	\param src1 -> The first source array. If the pointer is NULL, the array is assumed to be all 1’s.
*	\param src2 -> The second source array.
*	\param dst  -> The destination array.
*	\param scale -> Optional scale factor
*	\return none
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgDiv(  CvArr* src1,  CvArr* src2, CvArr* dst, double scale CV_DEFAULT(1) );

/* Logic operators can not be done in GLSL and OpenGL 2.1
cvAnd
cvAndS
cvOr
cvOrS
cvXOr
cvXOrS
cvNot
cvCmp
cvCmpS
cvInRange
cvInRangeS
*/
/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvMax" target=new>cvMax</a> function.
*  The function cvMax calculates per-element maximum of two arrays: dst(I)=max(src1(I), src2(I))
*  All the arrays must have a single channel, the same data type and the same size (or ROI size).
*	\param src1 -> The first source array.
*	\param src2 -> The second source array.
*	\param dst -> The destination array.
*	\return none
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgMax(  CvArr* src1,  CvArr* src2, CvArr* dst);

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvMaxS" target=new>cvMaxS</a> function.
*   \param src -> The source array.
*   \param value -> Added double.
*   \param dst -> The destination array.
*	\author Songbo.
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgMaxS( CvArr * src, double value, CvArr* dst);

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvMin" target=new>cvMin</a> function.
*  The function cvgMin calculates per-element minimum of two arrays: dst(I)=min(src1(I), src2(I))
*  All the arrays must have a single channel, the same data type and the same size (or ROI size).
*	\param src1 -> The first source array.
*	\param src2 -> The second source array.
*	\param dst -> The destination array.
*	\return none
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgMin(  CvArr* src1,  CvArr* src2, CvArr* dst);

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvMins" target=new>cvMins</a> function.
*   \param src -> The source array.
*   \param value -> Added double.
*   \param dst -> The destination array.
*	\author Songbo.
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgMinS( CvArr * src, double value, CvArr* dst);


/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvAbs" target=new>cvAbs</a> function.
*   \param src -> The source array.
*   \param dst -> The destination array.
*	\author Yannick Allusse.
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgAbs( CvArr * src, CvArr* dst);

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvAbsDiff" target=new>cvAbsDiff</a> function.
*   \param src1 -> The source array 1.
*   \param src2 -> The source array 2.
*   \param dst -> The destination array.
*	\author Yannick Allusse.
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgAbsDiff(CvArr* src1, CvArr* src2, CvArr* dst );

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvAbsDiffS" target=new>cvAbsDiffS</a> function.
*   \param src -> The source array.
*   \param value -> The acalar value.
*   \param dst -> The destination array.
*	\author Yannick Allusse.
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgAbsDiffS(CvArr* src, CvArr* dst, CvScalar value );

/** @}*///CVGXCORE_OPER_ARITHM_LOGIC_COMP_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVGXCORE_OPER_STATS_GRP
*  @{
\todo fill with OpenCV functions list.
*/

//! Define the size x*x of the image that is read back to CPU to calculate AVG
#define GPUCV_CXCOREG_AVG_READBACK_SIZE 8

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvSum" target=new>cvSum</a> function
*	\param arr -> source array to sum
*	\return Result sum as a scalar.
*	\author Allusse Yannick
*	\note Is based on cvgAvg() that get an approximate value, sum calculated can have an error rate of about 1% compare to OpenCv, see GPUCV_CXCOREG_AVG_READBACK_SIZE to improve the error rate.
*	\sa cvgAvg().
*/
_GPUCV_CXCOREG_EXPORT_C
CvScalar cvgSum(CvArr* arr);
/*!
*   \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvMinMaxLoc" target=new>cvMinMaxLoc</a> function.
*	The function is calculated the average of the image using MipMapping on GPU. Size of the texture image that is read back to CPU to calculate average is defined by GPUCV_CXCOREG_AVG_READBACK_SIZE.
*   \param arr -> The array.
*   \param mask -> The optional operation mask.
*	\note Average value is an approximation, usual error rate is about 1%.
*	\note GPUCV_CXCOREG_AVG_READBACK_SIZE can be change, but having size inferior to default value 8 will introduce an error rate superior to 3% compare to OpenCv.
*	\todo Is mask is not NULL, we go back to OpenCv operator.
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
CvScalar cvgAvg(CvArr* arr, CvArr* mask CV_DEFAULT(NULL));//, bool NotNullPixels=false);

#if _GPUCV_DEVELOP_BETA
/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvMinMaxLoc" target=new>cvMinMaxLoc</a> function.
*	\param arr -> The source array, single-channel or multi-channel with COI set.
*	\param min_val -> Pointer to returned minimum value, if pointer is NULL, minimum value is not processed.
*	\param max_val -> Pointer to returned maximum value, if pointer is NULL, maximum value is not processed.
*	\param min_loc -> Pointer to returned minimum location.
*	\param max_loc -> Pointer to returned maximum location.
*	\param mask	  -> The optional mask that is used to select a subarray.
*	\warning If mask != NULL, original Opencv operator is called. TODO
*	\warning If source image nChannel != 1, original Opencv operator is called. TODO
*	\warning If source image depth != 8, original Opencv operator is called. TODO
*	\warning Result location values may differ from OpenCV when several maximum or minimum are found into the image.
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgMinMaxLoc(  CvArr* arr, double* min_val, double* max_val, CvPoint* min_loc CV_DEFAULT(NULL), CvPoint* max_loc CV_DEFAULT(NULL),  CvArr* mask CV_DEFAULT(NULL) );
#endif
/** @}*///CVGXCORE_OPER_STATS_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________

/** @addtogroup CVGXCORE_OPER_LINEAR_ALGEBRA_GRP
*  @{
\todo fill with OpenCV functions list.
*/
#if _GPUCV_DEVELOP_BETA
/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvScaleAdd" target=new>cvScaleAdd</a> function.
*	\param src1 -> The first source array.
*	\param src2 -> The second source array.
*	\param dst -> The destination array.
*	\param scale -> scaling factor
*	\author Yannick Allusse
*	\todo Check why the OpenCV operator use array of maximum 2 channels?
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgScaleAdd(CvArr* src1, CvScalar scale, CvArr* src2, CvArr* dst);
#endif
/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvGEMM" target=new>cvGEMM</a> function.
*	\param src1 -> The first source array.
*	\param src2 -> The second source array.
*	\param src3 -> The third source array (shift). Can be NULL, if there is no shift.
*	\param dst -> The destination array.
*	\param alpha -> multiplication coef for (src1*src2).
*	\param beta -> multiplication coef for src3.
*	\param tABC -> The operation flags that can be 0 or combination of the following values:
CV_GEMM_A_T - transpose src1
CV_GEMM_B_T - transpose src2
CV_GEMM_C_T - transpose src3
*	\author Yannick Allusse
* \todo Support Transpose option.
*/
_GPUCV_CXCOREG_EXPORT_C
void  cvgGEMM(CvArr* src1,CvArr* src2, double alpha,CvArr* src3, double beta, CvArr* dst, int tABC CV_DEFAULT(0) );
#define cvgMatMulAdd( src1, src2, src3, dst )	cvgGEMM( src1, src2, 1, src3, 1, dst, 0 )
#define cvgMatMul( src1, src2, dst )			cvgGEMM( src1, src2, 1, NULL, 0, dst, 0 )
/** @}*///CVGXCORE_OPER_LINEAR_ALGEBRA_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVGXCORE_OPER_MATH_FCT_GRP
*  @{
\todo fill with opencv functions list.
*/

/**
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvPow" target=new>cvPow</a> function.
*   \param src -> The source array.
*   \param dst -> The destination array.
*   \param power -> exponent of power.
*	\author Yannick Allusse
*   \todo Operator only working with GL_FLOAT/IPL_DEPTH_32F image format cause other format are stored in range [0..1] in video memory.
*	\note Operator work internaly with texture pixel format : FLOAT 32, to give best precision
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgPow(CvArr* src, CvArr* dst, double power );

/** @}*///CVGXCORE_OPER_MATH_FCT_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/*/* @addtogroup CVGXCORE_OPER_RANDOW_NUMBER_GRP
*  @{
\todo fill with opencv functions list.
*/
/** @}*///CVGXCORE_OPER_RANDOW_NUMBER_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/*/* @addtogroup CVGXCORE_OPER_DISCRETE_TRANS_GRP
*  @{
\todo fill with OpenCV functions list.
*/
/** @}*///CVGXCORE_OPER_DISCRETE_TRANS_GRP
//_______________________________________________________________
//_______________________________________________________________


//_______________________________________________________________
/** @addtogroup CVGXCORE_DRAWING_CURVES_SHAPES_GRP
*  @{
*/
/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvLine" target=new>cvLine</a> function.
*	\param img -> source image
*	\param pt1 -> first point of the line segment.
*	\param pt2 -> second point of the line segment.
*	\param color -> line color
*	\param thickness -> Line thickness
*	\param line_type -> Line type
*	\param shift -> shift
*	\return none
*	\warning There are differences between cvLine and cvgLine : there is maximum for glLineWidth and the thick lines are not drawn with rounding endings.
\note if shift is not 0, it uses opecv.
*   \author Song Songbo
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgLine( IplImage* img, CvPoint pt1, CvPoint pt2, CvScalar color,int thickness, int line_type, int shift );

/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvRectangle" target=new>cvRectangle</a> function.

\note if shift is not 0, it uses opencv.
*	\author Song Songbo
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgRectangle( IplImage* img, CvPoint pt1, CvPoint pt2, CvScalar color,int thickness, int line_type, int shift );

/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvCircle" target=new>cvCircle</a> function.
\note if shift is not 0, it uses opecv.
*	\author Song Songbo
*	\author Yannick Allusse
*/
_GPUCV_CXCOREG_EXPORT_C
void cvgCircle( IplImage* img, CvPoint center, int radius, CvScalar color,int thickness, int line_type, int shift );
//...more
/** @}*///CVGXCORE_DRAWING_CURVES_SHAPES_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/*/* @addtogroup CVGXCORE_DRAWING_TEXT_GRP
*  @{
*/
/** @}*///CVGXCORE_DRAWING_TEXT_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/*/* @addtogroup CVGXCORE_DRAWING_POINT_CONTOUR_GRP
*  @{
*/
/** @}*///CVGXCORE_DRAWING_POINT_CONTOUR_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________

#if _GPUCV_DEPRECATED
//matrix processing
//_GPUCV_CXCOREG_EXPORT_C CvArr * cvgMatrixAvg( CvArr* A,  CvArr* B, CvArr* C,  CvArr* D, CvArr* E,  CvArr* F, CvArr* G,  CvArr* H);

//to remove..??
_GPUCV_CXCOREG_EXPORT_C void customMultiply(CvArr *A,CvArr* B, double alpha, CvArr* C);
//========================================

#endif
//==============================================================================

void unitCircle(int slices);
void drawCircle(double radius, double x, double y,int slices);
void drawIntCircle(int i_radius, int i_x, int i_y,int width, int height,int slices);
void drawCircleCenter(double radius, double x, double y,int slices, CvScalar SlicesColor, CvScalar CenterColor);
void unitCircleCenter(int slices, CvScalar SlicesColor, CvScalar CenterColor);


#endif//CVGPU_CXCOREG_H
