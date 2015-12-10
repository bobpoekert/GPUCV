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



/** \brief Contains CUDA correspondences of OpenCV cxcore operators.
*	\author Yannick Allusse
*/
#ifndef __GPUCV_CXCORE_CUDA_H
#define __GPUCV_CXCORE_CUDA_H
#include <cxcoregcu/config.h>
#include <GPUCVHardware/config.h>

#if _GPUCV_COMPILE_CUDA
#ifdef __cplusplus
#	include <GPUCV/include.h>
#	include <GPUCVCuda/DataDsc_CUDA_Buffer.h>
#endif

/** \brief Function to call numerous arithmetic and logics operators like Add/Sub/Mul...using mask image, or scalar values for the second source image.
\author Yannick Allusse
\warning Image depth and channel number can heavily impact on operator performances due to CUDA memory access mechanisms. This operator will reshape images to fit with coalescent memory read and write(it requires width * nchannels multiple of 2 or 4).
\sa gcuUnsetReshapeObj(), gcuUnsetReshapeObj().
*/
#if GPUCV_DEPRECATED
_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaArithm(GPUCV_ARITHM_LOGIC_OPER OPERATOR_TYPE,
				   CvArr* src1,
				   CvArr* src2,
				   CvArr* dst,
				   CvArr* mask CV_DEFAULT(NULL),
				   float scale CV_DEFAULT(1),
				   CvScalar * _Scalar CV_DEFAULT(NULL),
				   double * _val CV_DEFAULT(NULL))__GPUCV_THROW();
#endif//GPUCV_DEPRECATED
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVGXCORE_OPER_COPY_FILL_GRP
*  @{
\todo fill with OpenCV functions list.
*/
_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaCopy(CvArr* src1,
				CvArr* dst,
				CvArr* mask CV_DEFAULT(NULL))__GPUCV_THROW();

_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaSet( CvArr* arr,  CvScalar value, CvArr* mask CV_DEFAULT(NULL));

_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaSetZero( CvArr* arr );
/** @}*///CVGXCORE_OPER_COPY_FILL_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________


//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVGXCORE_OPER_ARITHM_LOGIC_COMP_GRP
*  @{
*/
	/**
	*  \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvLUT" target=new>cvLUT</a> function
	*  \param src1 -> Source array of 8-bit elements.
	*  \param dst ->  Destination array of arbitrary depth and of the same number of channels as the source array.
	*  \param lut -> Look-up table of 256 elements; should have the same depth as the destination array. Use only a single channel LUT.
	*  \author Ankit Agarwal
	*  \todo Add support for LUT multiple channels.
	*  \todo Can be optimized.
	*/
	_GPUCV_CXCOREGCU_EXPORT_C
		void  cvgCudaLUT(CvArr* src1 , CvArr* dst, CvArr* lut)__GPUCV_THROW();

	/**
	*  \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvConvertScale" target=new>cvCudaConvertScale</a> function
	*  \param src1 -> The source array.
	*	\param scale -> The scalar which defines the Scaling factor.
	*	\param shift -> The scalar which defines the Shifting factor.
	*	\param dst -> Output destination array.
	*	\author Ankit Agarwal
	*	\todo Test with format 16u and 16s.
	*/
	_GPUCV_CXCOREGCU_EXPORT_C
		void  cvgCudaConvertScale(CvArr* src1,CvArr* dst,double scale,double shift)__GPUCV_THROW();

//Definitions of arithmetics operators
//KEYTAGS: TUTO_CUDA_ARITHM__STP6__ADD_CVG_DEFINITION

_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaAdd(CvArr* src1,
				CvArr* src2,
				CvArr* dst,
				CvArr* mask CV_DEFAULT(NULL))__GPUCV_THROW();


_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaAddS(CvArr* src1,
				 CvScalar scalar,
				 CvArr* dst,
				 CvArr* mask CV_DEFAULT(NULL)) __GPUCV_THROW();

/**
*  \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvAddWeighted" target=new>cvCudaAddWeighted</a> function
*  \param src1 -> The first source array.
*  \param s1 -> The scalar which defines the weightage for first image.
*  \param src2 -> The second source array,Both source must be single channel.
*  \param s2 -> The scalar which defines the weightage for Second image.All arrays must have same type
*  \param dst -> Output destination array.
*  \author Ankit Agarwal
*  \
*/
_GPUCV_CXCOREGCU_EXPORT_C
void  cvgCudaAddWeighted(CvArr* src1,double s1,CvArr* src2,double s2,double g,CvArr* dst)__GPUCV_THROW();


_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaSub(CvArr* src1,
				CvArr* src2,
				CvArr* dst,
				CvArr* mask CV_DEFAULT(NULL))__GPUCV_THROW();


_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaSubS(CvArr* src1,
				 CvScalar scalar,
				 CvArr* dst,
				 CvArr* mask CV_DEFAULT(NULL))__GPUCV_THROW();


_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaSubRS(CvArr* src1,
				 CvScalar scalar,
				 CvArr* dst,
				 CvArr* mask CV_DEFAULT(NULL))__GPUCV_THROW();

_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaMul(CvArr* src1,
				CvArr* src2,
				CvArr* dst,
				double scale
				)__GPUCV_THROW();


_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaDiv(CvArr* src1,
				CvArr* src2,
				CvArr* dst,
				double scale
				)__GPUCV_THROW();


_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaMin(CvArr* src1,
				CvArr* src2,
				CvArr* dst
				)__GPUCV_THROW();


_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaMinS(CvArr* src1,
				 double val,
				 CvArr* dst
				 )__GPUCV_THROW();

_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaMax(CvArr* src1,
				CvArr* src2,
				CvArr* dst
				)__GPUCV_THROW();

_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaMaxS(CvArr* src1,
				 double val,
				 CvArr* dst
				 )__GPUCV_THROW();

//logics
//AND
_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaAnd(CvArr* src1,
				CvArr* src2,
				 CvArr* dst,
				 CvArr* mask CV_DEFAULT(NULL))__GPUCV_THROW();
_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaAndS(CvArr* src1,
				CvScalar scalar,
				CvArr* dst,
				CvArr* mask CV_DEFAULT(NULL))__GPUCV_THROW();
//OR
_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaOr(CvArr* src1,
				CvArr* src2,
				CvArr* dst,
				CvArr* mask CV_DEFAULT(NULL))__GPUCV_THROW();
_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaOrS(CvArr* src1,
				 CvScalar scalar,
				 CvArr* dst,
				 CvArr* mask CV_DEFAULT(NULL))__GPUCV_THROW();
//XOR
_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaXor(CvArr* src1,
				CvArr* src2,
				CvArr* dst,
				CvArr* mask CV_DEFAULT(NULL))__GPUCV_THROW();
_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaXorS(CvArr* src1,
				 CvScalar scalar,
				 CvArr* dst,
				 CvArr* mask CV_DEFAULT(NULL))__GPUCV_THROW();
//NOT
_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaNot(CvArr* src1,
				CvArr* dst)__GPUCV_THROW();

//CMP, CMPS
	/**
	*  \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvCmp" target=new>cvCmp</a> function
	*  \param src1 -> The first source array.
	*  \param src2 -> The second source array,Both source must be single channel.
	*  \param dst -> Output destination array must have 8uor 8s type.
	*  \param cmp_op -> The parameter which decides which operator to apply.
	*  \author Ankit Agarwal
	*/
	_GPUCV_CXCOREGCU_EXPORT_C
		void  cvgCudaCmp(CvArr* src1,CvArr* src2, CvArr* dst,int cmp_op)__GPUCV_THROW();

	/**
	*  \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvCmpS" target=new>cvCmpS</a> function
	*  \param src -> The first source array.
	*  \param value -> The second double value.
	*  \param dst -> Output destination array must have 8uor 8s type.
	*  \param cmp_op -> The parameter which decides which operator to apply.
	*  \author Yannick Allusse
	*/
	_GPUCV_CXCOREGCU_EXPORT_C
		void  cvgCudaCmpS(CvArr* src,double value, CvArr* dst,int cmp_op)__GPUCV_THROW();

//ABS
_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaAbs(CvArr* src1,
				CvArr* dst)__GPUCV_THROW();

_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaAbsDiff(CvArr* src1,
				CvArr* src2,
				CvArr* dst)__GPUCV_THROW();

_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaAbsDiffS(CvArr* src1,
				 CvArr* dst,
				 CvScalar scalar)__GPUCV_THROW();
/** @}*///CVGXCORE_OPER_ARITHM_LOGIC_COMP_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________

_GPUCV_CXCOREGCU_EXPORT_C CvScalar cvgCudaSum(CvArr* arr )__GPUCV_THROW();
//_GPUCV_CXCOREGCU_EXPORT_C CvScalar cvgCudaAvg(CvArr* arr,CvArr* mask=NULL );
//_______________________________________________________________
/** @addtogroup CVGXCORE_OPER_LINEAR_ALGEBRA_GRP
*  @{
*/
_GPUCV_CXCOREGCU_EXPORT_C
void  cvgCudaScaleAdd(CvArr* src1, CvScalar scale, CvArr* src2, CvArr* dst)__GPUCV_THROW();
#define cvgCudaMulAddS cvgCudaScaleAdd


_GPUCV_CXCOREGCU_EXPORT_C
void  cvgCudaGEMM(CvArr* src1,
				  CvArr* src2,
				  double alpha,
				  CvArr* src3,
				  double beta,
				  CvArr* dst,
				  int tABC CV_DEFAULT(0))__GPUCV_THROW();

_GPUCV_CXCOREGCU_EXPORT_C
void  cvgCudaTranspose(CvArr* src, CvArr* dst)__GPUCV_THROW();
/** @}*///CVGXCORE_OPER_LINEAR_ALGEBRA_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVGXCORE_OPER_MATH_GRP
*  @{
*/
/**
*  \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvPOW" target=new>cvPOW</a> function
*  \param src1 -> Source array
*  \param dst ->  The destination array, should be the same type as the source.
*  \param power ->     The exponent of power.
*  \author Ankit Agarwal
*/
_GPUCV_CXCOREGCU_EXPORT_C
void  cvgCudaPow(CvArr* src1 , CvArr* dst, double power)__GPUCV_THROW();

/** @}*///CVGXCORE_OPER_MATH_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVGXCORE_OPER_TRANSFORM_PERMUT_GRP
*  @{
*/
_GPUCV_CXCOREGCU_EXPORT_C
void cvgCudaSplit(  CvArr* src, CvArr* dst0, CvArr* dst1, CvArr* dst2, CvArr* dst3 )__GPUCV_THROW();
	#define cvgCudaCvtPixToPlane cvgCudaSplit//for compatibilities issues...

_GPUCV_CXCOREGCU_EXPORT_C
	void cvgCudaMerge(CvArr* src0, CvArr* src1, CvArr* src2, CvArr* src3, CvArr* dst )__GPUCV_THROW();
#define cvgCudaCvtPlaneToPix cvgCudaMerge

_GPUCV_CXCOREGCU_EXPORT_C
	void cvgCudaFlip(CvArr* src, CvArr* dst, int flip_mode CV_DEFAULT(0))__GPUCV_THROW();
//cvMixChannels
//cvRandShuffle
/** @}*///CVGXCORE_OPER_TRANSFORM_PERMUT_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @addtogroup CVGXCORE_OPER_DESCRETE_TRANSFORMS_GRP
*  @{
*/


/**
*  \brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cxcore.htm#decl_cvDFT" target=new>cvDFT</a> function
*  \param src1 -> The source array must be 32f(float) array of 1,3 and 4 channels.
*  \param dst ->  Output destination array 32f.
*  \param flags -> The scalar which defines the direction(FORWARD|INVERSE) for the DFT.
*  \param flags -> The number of no zeros rows in input matrix.
*  \author Ankit Agarwal
*/
_GPUCV_CXCOREGCU_EXPORT_C
void  cvgCudaDFT(CvArr* src1, CvArr* dst, int flags, int nonzero_rows)__GPUCV_THROW();

/** @}*///CVGXCORE_OPER_DESCRETE_TRANSFORMS_GRP
//_______________________________________________________________
//_______________________________________________________________

/**
*  \Brief - An operator which takes summed area tables as input and calculates local sum on GPU with CUDA
*  \param src1 -> The source summed area table.
*  \param dst ->  Output Image.
*  \param height -> The scalar which defines the height of the local area.
*  \param width -> The scalar which defines the Width of the local area.
*  \author Ankit Agarwal
*/
_GPUCV_CXCOREGCU_EXPORT_C
void  cvgCudaLocalSum(CvArr* src1,CvArr* dst, int height , int width)__GPUCV_THROW();


#endif//_GPUCV_COMPILE_CUDA
#endif
