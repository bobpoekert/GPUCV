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
/** Contains CUDA correspondences of OpenCV cv operators.
*	\author Yannick Allusse
*/
#ifndef __GPUCV_CVG_CUDA_H
#define __GPUCV_CVG_CUDA_H
#include <cvgcu/config.h>

#if _GPUCV_COMPILE_CUDA
#include <GPUCVHardware/config.h>

#ifdef __cplusplus
#	include <GPUCV/include.h>
#	include <GPUCVCuda/DataDsc_CUDA_Buffer.h>
#endif

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
*	\author NVIDIA(CUDA SDK), integrated in GPUCV by Yannick Allusse
*/
_GPUCV_CVGCU_EXPORT_C
void cvgCudaSobel(CvArr* src, CvArr* dst, int xorder, int yorder, int aperture_size CV_DEFAULT(3) );

#if _GPUCV_DEVELOP_BETA
/*!
*   \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvlaplace" target=new>cvLaplace</a> function.
*	\param	src => Source image.
*	\param	dst => Destination image.
*	\param	aperture_size => Size of the extended Sobel kernel, must be 1, 3, 5 or 7.
*	\author Integrated in GPUCV by Yannick Allusse, code based on cvgCudaSobel().
*/
_GPUCV_CVGCU_EXPORT_C
void cvgCudaLaplace(CvArr* src, CvArr* dst, int aperture_size CV_DEFAULT(3) );
#endif// _GPUCV_DEVELOP_BETA
//...
/** @}*///CVG_IMGPROC__GRAD_EDGE_CORNER_GRP

//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVG_IMGPROC__MORPHO_GRP
*  @{
*/
#if _GPUCV_DEVELOP_BETA
/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvDilate" target=new>cvDilate</a> function.
*	\param src -> source image
*	\param dst -> destination image
*	\param element-> kernel to use
*	\param iterations -> number of iterations (not available yet)
*	\warning cvgCudaDilate uses GPU optimization only with square structuring elements of odd-number size! For other sizes it uses the original cvErode operator.
*	\warning Only working with single channel and unsigned char images.
*	\note cvgCudaDilate uses optimized filter is all the elements of the kernel are equal to 1.
*	\author Yannick Allusse
*	\bug Not working for iteration > 1 => trouble with the temp buffer
*/
_GPUCV_CVGCU_EXPORT_C
void cvgCudaDilate(CvArr* src, CvArr* dst, IplConvKernel* element CV_DEFAULT(NULL), int iterations CV_DEFAULT(1) );

_GPUCV_CVGCU_EXPORT_C
void cvgCudaErode(CvArr* src, CvArr* dst, IplConvKernel* element CV_DEFAULT(NULL), int iterations CV_DEFAULT(1) );
_GPUCV_CVGCU_EXPORT_C
void cvgCudaErodeNoShare(CvArr* src, CvArr* dst, IplConvKernel* element CV_DEFAULT(NULL), int iterations CV_DEFAULT(1) );
#endif// _GPUCV_DEVELOP_BETA
/** @}*///CVG_IMGPROC__MORPHO_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVG_IMGPROC__FILTERS_CLR_CONV_GRP
*  @{
*/


/*!
*   \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvThreshold" target=new>cvThreshold</a> function.
*	\param	src => Source array (single-channel, 8-bit of 32-bit floating point).
*	\param	dst => Destination array; must be either the same type as src or 8-bit.
*	\param  threshold => Threshold value (see the documention).
*	\param	max_value => Maximum value to use with CV_THRESH_BINARY and CV_THRESH_BINARY_INV thresholding types.
*	\param	threshold_type => Thresholding type.
*	\author Integrated in GPUCV by Clement Beausset, code based on cvgThreshold().
*	\todo Yannick:Test with format 32F, optimize by using reshape into multiple channels.
*/
_GPUCV_CVGCU_EXPORT_C
void cvgCudaThreshold(CvArr* srcArr, CvArr* dstArr, double threshold, double max_value, int threshold_type );

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
*	\author Integrated in GPUCV by Yannick Allusse, code based on cvgCudaSobel().
*/
_GPUCV_CVGCU_EXPORT_C
void cvgCudaSmooth(CvArr* src, CvArr* dst, int smoothtype CV_DEFAULT(CV_GAUSSIAN), int param1 CV_DEFAULT(3), int param2 CV_DEFAULT(0), double param3 CV_DEFAULT(0), double param4 CV_DEFAULT(0));
#endif// _GPUCV_DEVELOP_BETA
#ifdef _GPUCV_CUDA_SUPPORT_CUDPP
/*!
*   \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvIntegral" target=new>cvIntegral</a> function.
*	\param	image => The source image, W�H, 8-bit or floating-point (32f or 64f) image.
*	\param	sum => The integral image, W+1�H+1, 32-bit integer or double precision floating-point (64f).
*	\param	sqsum => The integral image for squared pixel values, W+1�H+1, double precision floating-point (64f).
*	\param	tilted_sum => The integral for the image rotated by 45 degrees, W+1�H+1, the same data type as sum.
*	\note  sqsum and tilted_sum aer not yet calculated.
*	\author Integrated in GPUCV by Yannick Allusse, based on CUDPPs.
*/
_GPUCV_CVGCU_EXPORT_C
void cvgCudaIntegral(CvArr* image, CvArr* sum, CvArr* sqsum CV_DEFAULT(NULL) , CvArr* tilted_sum CV_DEFAULT(NULL));
#endif

_GPUCV_CVGCU_EXPORT_C
void cvgCudaCvtColor(CvArr* src, CvArr* dst, int code);

/** @}*///CVG_IMGPROC__FILTERS_CLR_CONV_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVG_IMGPROC__HISTOGRAM_GRP
*  @{
\todo fill with opencv functions list.
*/
/*!
*   \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvCalcHist" target=new>cvCalcHist</a> function.
*	\param image => Source images (though, you may pass CvMat** as well), all are of the same size and type.
*	\param hist => Pointer to the histogram.
*	\param accumulate	=> Accumulation flag. If it is set, the histogram is not cleared in the beginning. This feature allows user to compute a single histogram from several images, or to update the histogram online.
*	\param mask => The operation mask, determines what pixels of the source images are counted.
*	\author NVIDIA(CUDA SDK), integrated in GPUCV by Yannick Allusse
*	\todo - Add support for mask image.
*	\todo - Only histogram of bins 64 and 256 are currently supported.
*	\todo - Only the first input image is processed.
*	\todo - Accumulate flag is not supported.
*	\todo - Histogram range value is currently fixed to 256.
*/
_GPUCV_CVGCU_EXPORT_C
void cvgCudaCalcHist( IplImage** image, CvHistogram* hist, int accumulate CV_DEFAULT(0), const CvArr* mask CV_DEFAULT(NULL));

/** @}*///CVG_IMGPROC__HISTOGRAM_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
_GPUCV_CVGCU_EXPORT_C
void cvgCudaDeriche(CvArr* srcArr, CvArr* dstArr, double alpha);

#endif
#endif
