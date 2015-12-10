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

#include <includecv.h>
#include <cvgcu/cvgcu.h>


#if 0
_GPUCV_CVGCU_EXPORT_C CVAPI(CvHaarClassifierCascade*) cvLoadHaarClassifierCascade(
		const char* directory, CvSize orig_window_size);

_GPUCV_CVGCU_EXPORT_C CVAPI(void) cvReleaseHaarClassifierCascade( CvHaarClassifierCascade** cascade );

#define CV_HAAR_DO_CANNY_PRUNING 1
#define CV_HAAR_SCALE_IMAGE      2


	/** 
	*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#cv_pattern_objdetection" target=new>cvgHaarDetectObjects</a> function.
	*	\param image -> Image to detect objects in. 
	*	\param cascade -> Haar classifier cascade in internal representation. 
	*	\param storage -> Memory storage to store the resultant sequence of the object candidate rectangles. 
	*	\param scale_factor -> The factor by which the search window is scaled between the subsequent scans, for example, 1.1 means increasing window by 10%. 
	*	\param min_neighbors -> Minimum number (minus 1) of neighbor rectangles that makes up an object. All the groups of a smaller number of rectangles than min_neighbors-1 are rejected. If min_neighbors is 0, the function does not any grouping at all and returns all the detected candidate rectangles, which may be useful if the user wants to apply a customized grouping procedure. 
	*	\param flags -> Mode of operation. Currently the only flag that may be specified is CV_HAAR_DO_CANNY_PRUNING. If it is set, the function uses Canny edge detector to reject some image regions that contain too few or too much edges and thus can not contain the searched object. The particular threshold values are tuned for face detection and in this case the pruning speeds up the processing. 
	*	\param min_size -> Minimum window size. By default, it is set to the size of samples the classifier has been trained on (~20×20 for face detection).
	*	\todo 1. Implement the Tree Data Structure on GPU.
	2. Implement the cvRunHaarClassifierCascade function as a CUDA kernel,so the pixels can be processed in parallel
	3. While processing the pixels , use a shared block of memory for each block to store the rectangles.
	4. After processing, Do the neigbhour check for rectangles and returnthe final ist to CPU.
	*	\author Ankit Agarwal.
	*/

	/*_GPUCV_CUDA_EXPORT_CU _GPUCV_CUDA_EXPORT*/
		_GPUCV_CVGCU_EXPORT_C CvSeq* cvgHaarDetectObjects( const CvArr* image,
		CvHaarClassifierCascade* cascade,
		CvMemStorage* storage, double scale_factor CV_DEFAULT(1.1),
		int min_neighbors CV_DEFAULT(3), int flags CV_DEFAULT(0),
		CvSize min_size CV_DEFAULT(cvSize(0,0)));

	IPCVAPI_EX( CvStatus, icvIntegral_8u32s_C1R,
		"ippiIntegral_8u32s_C1R", CV_PLUGINS1(CV_PLUGIN_IPPCV),
		( const uchar* pSrc, int srcStep, int* pDst, int dstStep,
		CvSize roiSize, int val ))

		IPCVAPI_EX( CvStatus, icvSqrIntegral_8u32s64f_C1R,
		"ippiSqrIntegral_8u32s64f_C1R", CV_PLUGINS1(CV_PLUGIN_IPPCV),
		( const uchar* pSrc, int srcStep,
		int* pDst, int dstStep, double* pSqr, int sqrStep,
		CvSize roiSize, int val, double valSqr ))

		IPCVAPI_EX( CvStatus, icvRectStdDev_32s32f_C1R,
		"ippiRectStdDev_32s32f_C1R", CV_PLUGINS1(CV_PLUGIN_IPPCV),
		( const int* pSrc, int srcStep,
		const double* pSqr, int sqrStep, float* pDst, int dstStep,
		CvSize roiSize, CvRect rect ))

		IPCVAPI_EX( CvStatus, icvHaarClassifierInitAlloc_32f,
		"ippiHaarClassifierInitAlloc_32f", CV_PLUGINS1(CV_PLUGIN_IPPCV),
		( void **pState, const CvRect* pFeature, const float* pWeight,
		const float* pThreshold, const float* pVal1,
		const float* pVal2, const int* pNum, int length ))

		IPCVAPI_EX( CvStatus, icvHaarClassifierFree_32f,
		"ippiHaarClassifierFree_32f", CV_PLUGINS1(CV_PLUGIN_IPPCV),
		( void *pState ))

		IPCVAPI_EX( CvStatus, icvApplyHaarClassifier_32s32f_C1R,
		"ippiApplyHaarClassifier_32s32f_C1R", CV_PLUGINS1(CV_PLUGIN_IPPCV),
		( const int* pSrc, int srcStep, const float* pNorm,
		int normStep, uchar* pMask, int maskStep,
		CvSize roi, int *pPositive, float threshold,
		void *pState ))

#endif



