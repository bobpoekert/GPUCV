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
/**	\file cvg.h
* 	\brief Header file containg definitions for the GPU equivalent open CV functions
* 	\author Jean-Philippe Farrugia
*	\author Yannick Allusse
*/
#ifndef __GPUCV_CVG_H
#define __GPUCV_CVG_H

#include <cvg/config.h>
#include <GPUCV/misc.h>
//Cv reference =============================================================
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVG_IMGPROC__GRAD_EDGE_CORNER_GRP
*  @{
*	\todo fill with OpenCV functions list.
*/
/*!
*   \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvSobel" target=new>cvSobel</a> function.
*	\param	src => Source image. 
*	\param	dst => Destination image. 
*	\param	xorder => Order of the derivative x . 
*	\param	yorder => Order of the derivative y . 
*	\param	aperture_size => Size of the extended Sobel kernel, must be 1, 3, 5 or 7.
*	\note  Aperture size can be 3 or -1(CV_SCHARR).
*	\note  xorder and yorder are limited to 0 or 1.
*    \note  Results have small differences with orignal operators.
*	\author Integrated in GPUCV by Yannick Allusse, original source from OpenGL SuperBible.
*/ 
_GPUCV_CVG_EXPORT_C 
void cvgSobel(CvArr* src, CvArr* dst, int xorder, int yorder, int aperture_size  CV_DEFAULT(3) );

/*!
*   \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvLaplace" target=new>cvLaplace</a> function.
*	\param	src => Source image. 
*	\param	dst => Destination image. 
*	\param	aperture_size => Aperture size (it has the same meaning as in cvSobel).
*	\author Integrated in GPUCV by Yannick Allusse, original source from OpenGL SuperBible.
*	\note  Aperture size can be 3 or 1.
*	\todo This operator is only using char/uchar as input format. Make it compatible with other format.
*/ 
_GPUCV_CVG_EXPORT_C
void cvgLaplace(CvArr* src, CvArr* dst, int aperture_size CV_DEFAULT(3) );

#if _GPUCV_DEVELOP_BETA
/*!
*   \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvLaplace" target=new>cvLaplace</a> function.
*	\param	image => Source image. 
*	\param	edges => Image to store the edges found by the function.
*	\param	threshold1 => The first threshold.
*	\param	threshold2 => The second threshold.
*	\param	aperture_size => Aperture parameter for Sobel operator (see cvSobel).
*	\author Integrated in GPUCV by Yannick Allusse.
*/ 
_GPUCV_CVG_EXPORT_C
void cvgCanny(CvArr* image, CvArr* edges, double threshold1, double threshold2, int aperture_size CV_DEFAULT(3) );
#endif
//...
/** @}*///CVG_IMGPROC__GRAD_EDGE_CORNER_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVG_IMGPROC__SAMPLE_INTER_GEO_GRP
*  @{
\todo fill with opencv functions list.
*/


/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvResize" target=new>cvResize</a> function.
*	\param src -> source image
*	\param dst -> destination image
*	\param interpolation -> Interpolation method:
* 	<ul><li>CV_INTER_NN - nearest-neigbor interpolation,</li>
* 	<li>CV_INTER_LINEAR - bilinear interpolation (used by default)</li>
* 	<li>CV_INTER_AREA - resampling using pixel area relation. It is preferred method for image decimation that gives moire-free results. In case of zooming it is similar to CV_INTER_NN method.</li>
* 	<li>CV_INTER_CUBIC - bicubic interpolation. {code of color conversino operator</li>
*	</ul>
*	\warning Only the interpolation method(CV_INTER_NN) have been ported to GPU yet.
*	\author Yannick Allusse
*/ 
_GPUCV_CVG_EXPORT_C 
void cvgResize(  CvArr* src, CvArr* dst, int interpolation CV_DEFAULT(CV_INTER_LINEAR));

/** @}*///CVG_IMGPROC__SAMPLE_INTER_GEO_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVG_IMGPROC__MORPHO_GRP
*  @{
\todo fill with OpenCV functions list.
*/

/*! 
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvDilate" target=new>cvDilate</a> function.
*	\param src -> source image
*	\param dst -> destination image
*	\param element-> kernel to use
*	\param iterations -> number of iterations (not avaible yet)
*	\warning cvgDilate uses GPU optimisation only with square structuring elements of odd-number size! For other sizes it uses the original cvErode operator.
*	\bug Correct bug that force iteration number to be iteration+1 to have the same restult as OpenCV.
*	\author Yannick Allusse
*	\author Jean-Philippe Farrugia
*/ 
_GPUCV_CVG_EXPORT_C 
void cvgDilate(CvArr* src, CvArr* dst, IplConvKernel* element CV_DEFAULT(NULL), int iterations CV_DEFAULT(1) );


/*! 
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvErode" target=new>cvErode</a> function.
*	\param src -> source image
*	\param dst -> destination image
*	\param element-> kernel to use
*	\param iterations -> number of iterations (not avaible yet)
*	\warning cvgErode uses GPU optimization only with square structuring elements of odd-number size! For other sizes it uses the original cvErode operator.
*	\bug
*	\bug Correct bug that force iteration number to be iteration+1 to have the same restult as OpenCV.
*	\author Yannick Allusse
*	\author Jean-Philippe Farrugia
*/ 
_GPUCV_CVG_EXPORT_C 
void cvgErode(CvArr* src, CvArr* dst, IplConvKernel* element CV_DEFAULT(NULL), int iterations CV_DEFAULT(1));

/*! 
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvMorphologyEx" target=new>cvMorphologyEx</a> function.
*	\param src -> source image
*	\param dst -> destination image
*	\param temp -> temporary image (may not be used)
*	\param B-> kernel to use
*	\param op -> operator to apply
*	\param iterations -> number of iterations
*	\author Yannick Allusse, Jean-Philippe Farrugia
*/ 
_GPUCV_CVG_EXPORT_C 
void cvgMorphologyEx(CvArr* src, CvArr* dst, CvArr* temp, IplConvKernel* B, CvMorphOp op, int iterations );

/** @}*///CVG_IMGPROC__MORPHO_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVG_IMGPROC__FILTERS_CLR_CONV_GRP
*  @{
\todo fill with opencv functions list.
*/

#if _GPUCV_DEVELOP_BETA
/*!
*   \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvSmooth" target=new>cvSmooth</a> function.
*	\param	src => Source image. 
*	\param	dst => Destination image. 
*	\param  smoothtype => Type of the smoothing (see OpenCV description)
*	\param	param1 => The first parameter of smoothing operation. 
*	\param	param2 => The second parameter of smoothing operation. In case of simple scaled/non-scaled and Gaussian blur if param2 is zero, it is set to param1. 
*	\param	param3 => (see OpenCV description)
*	\param	param4 => In case of non-square Gaussian kernel the parameter may be used to specify a different (from param3) sigma in the vertical direction.
*	\author Integrated in GPUCV by Yannick Allusse, code based on cvgSobel().
*/ 
_GPUCV_CVG_EXPORT_C 
void cvgSmooth(CvArr* src, CvArr* dst, int smoothtype CV_DEFAULT(CV_GAUSSIAN), int param1 CV_DEFAULT(3), int param2 CV_DEFAULT(0), double param3 CV_DEFAULT(0), double param4 CV_DEFAULT(0));
#endif//_GPUCV_DEVELOP_BETA

/*! 
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvCvtColor" target=new>cvCvtColor</a> function.
*	\param src -> source image
*	\param dst -> destination image
*	\param code -> code of color conversion operator
*	\warning There are small differences with cvCvtColor when using conversion code CV_YCrCb2BGR.
*	\author Jean-Philippe Farrugia
*/ 
_GPUCV_CVG_EXPORT_C 
void cvgCvtColor(  CvArr* src, CvArr* dst, int code );

/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvThreshold" target=new>cvThreshold</a> function.
*	\param src -> source image
*	\param dst -> destination image
*	\param threshold -> threshold value
*	\param maxValue -> maximum value for some kind of threshold types
*	\param thresholdType -> type of threshold operator to apply
*	\author Jean-Philippe Farrugia
*/ 
_GPUCV_CVG_EXPORT_C 
void cvgThreshold(  CvArr* src, CvArr* dst, double threshold, double maxValue, int thresholdType );

/** @}*///CVG_IMGPROC__FILTERS_CLR_CONV_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________

//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVG_IMGPROC__SPE_IMG_TRANS_GRP
*  @{
\todo fill with OpenCV functions list.
*/

#if _GPUCV_DEVELOP_BETA
/*! 
*  \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvDistTransform" target=new>cvDistTransform</a> function using geometric objects.
*  Calculates distance to closest zero pixel for all non-zero pixels of source image
*  \param src -> Source 8-bit single-channel (binary) image. 
*  \param dst -> Output image with calculated distances (32-bit floating-point, single-channel). 
*  \param distance_type -> Type of distance; can be CV_DIST_L1, CV_DIST_L2, CV_DIST_C, CV_DIST_USER or CVG_DIST_GEO(CVG_DIST_GEO is only available on GPUCV). 
*  \param mask_size -> Size of distance transform mask; can be 3 or 5. In case of CV_DIST_L1 or CV_DIST_C the parameter is forced to 3, because 3�3 mask gives the same result as 5�5 yet it is faster. 
*  \param mask -> User-defined mask in case of user-defined distance, it consists of 2 numbers (horizontal/vertical shift cost, diagonal shift cost) in case of 3�3 mask and 3 numbers (horizontal/vertical shift cost, diagonal shift cost, knight�s move cost) in case of 5�5 mask. 
*  \param labels -> The optional output 2d array of labels of integer type and the same size as src and dst. 
*  \return none.
*  \warning All features are disabled for now, gpuCV implementation of DistTransform is barely in alpha version. More to come...
*  \warning distance_type must be set to CVG_DIST_GEO to be runned on GPU. Else the operator uses original OpenCV operator.
*  \author Ankit Agarwal
*/
//see for depth : http://www.mevis.de/opengl/glDepthFunc.html
_GPUCV_CVG_EXPORT_C 
void cvgDistTransform(CvArr* src, CvArr* dst, int distance_type, int mask_size,  float* mask, CvArr* labels );
#endif
/** @}*///CVG_IMGPROC__SPE_IMG_TRANS_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________

//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVG_IMGPROC__HISTOGRAM_GRP
*  @{
\todo fill with OpenCV functions list.
*/

_GPUCV_CVG_EXPORT_C
void cvgCalcHist(IplImage ** _src, CvHistogram* hist, int accumulate CV_DEFAULT(0), const CvArr* mask CV_DEFAULT(NULL));

/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvQueryHistValue_*D" target=new>cvQueryHistValue_*D</a> function.
*	\author Jean-Philippe Farrugia
*/ 
_GPUCV_CVG_EXPORT_C 
float cvgQueryHistValue(CvArr* src,int color);

/** @}*///CVG_IMGPROC__HISTOGRAM_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________

#endif //CVGPU_CVG_H
