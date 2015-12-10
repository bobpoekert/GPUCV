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
#ifdef __cplusplus
	#include <GPUCVSwitch/macro.h>
	#include <GPUCVCore/GpuTextureManager.h>
	#include <GPUCVSwitch/Cl_Dll.h>
	#include <GPUCVSwitch/switch.h>
	using namespace std;
	using namespace GCV;
#endif

#define _GPUCV_FORCE_OPENCV_NP 1
#include <includecv.h>
#define CVAPI(MSG) MSG
#ifndef __HIGHGUI_SWIT_H
#define __HIGHGUI_SWIT_H


#include <highgui_switch/highgui_switch.h>

#define cvConvertImage	 cvgswConvertImage
#define cvCreateCameraCapture	 cvgswCreateCameraCapture
#define cvCreateFileCapture	 cvgswCreateFileCapture
#define cvCreateTrackbar	 cvgswCreateTrackbar
#define cvCreateTrackbar2	 cvgswCreateTrackbar2
#define cvCreateVideoWriter	 cvgswCreateVideoWriter
#define cvDecodeImage	 cvgswDecodeImage
#define cvDecodeImageM	 cvgswDecodeImageM
#define cvDestroyAllWindows	 cvgswDestroyAllWindows
#define cvDestroyWindow	 cvgswDestroyWindow
#define cvEncodeImage	 cvgswEncodeImage
#define cvGetCaptureDomain	 cvgswGetCaptureDomain
#define cvGetCaptureProperty	 cvgswGetCaptureProperty
#define cvGetTrackbarPos	 cvgswGetTrackbarPos
#define cvGetWindowHandle	 cvgswGetWindowHandle
#define cvGetWindowName	 cvgswGetWindowName
#define cvGetWindowProperty	 cvgswGetWindowProperty
#define cvGrabFrame	 cvgswGrabFrame
#define cvInitSystem	 cvgswInitSystem
#define cvLoadImage	 cvgswLoadImage
#define cvLoadImageM	 cvgswLoadImageM
#define cvMoveWindow	 cvgswMoveWindow
#define cvNamedWindow	 cvgswNamedWindow
#define cvQueryFrame	 cvgswQueryFrame
#define cvReleaseCapture	 cvgswReleaseCapture
#define cvReleaseVideoWriter	 cvgswReleaseVideoWriter
#define cvResizeWindow	 cvgswResizeWindow
#define cvRetrieveFrame	 cvgswRetrieveFrame
#define cvSaveImage	 cvgswSaveImage
#define cvSetCaptureProperty	 cvgswSetCaptureProperty
#define cvSetMouseCallback	 cvgswSetMouseCallback
#define cvSetPostprocessFuncWin32	 cvgswSetPostprocessFuncWin32
#define cvSetPreprocessFuncWin32	 cvgswSetPreprocessFuncWin32
#define cvSetTrackbarPos	 cvgswSetTrackbarPos
#define cvSetWindowProperty	 cvgswSetWindowProperty
#define cvShowImage	 cvgswShowImage
#define cvStartWindowThread	 cvgswStartWindowThread
#define cvWaitKey	 cvgswWaitKey
#define cvWriteFrame	 cvgswWriteFrame
/*........End Declaration.............*/


#endif //__HIGHGUI_SWIT_H