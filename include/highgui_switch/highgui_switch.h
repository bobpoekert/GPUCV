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
#ifndef __HIGHGUI_SWITCH_H
#define __HIGHGUI_SWITCH_H

#include <highgui_switch/config.h>
#ifdef __cplusplus
_HIGHGUI_SWITCH_EXPORT  void cvg_highgui_switch_RegisterTracerSingletons(SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> * _pAppliTracer, SG_TRC::CL_TRACING_EVENT_LIST *_pEventList);
#endif
_HIGHGUI_SWITCH_EXPORT_C void cvgswConvertImage( CvArr* src, CvArr* dst, int flags CV_DEFAULT(0));
_HIGHGUI_SWITCH_EXPORT_C CvCapture* cvgswCreateCameraCapture(int index);
_HIGHGUI_SWITCH_EXPORT_C CvCapture* cvgswCreateFileCapture(const  char* filename);
_HIGHGUI_SWITCH_EXPORT_C int cvgswCreateTrackbar(const  char* trackbar_name, const  char* window_name, int* value, int count, CvTrackbarCallback on_change);
_HIGHGUI_SWITCH_EXPORT_C int cvgswCreateTrackbar2(const  char* trackbar_name, const  char* window_name, int* value, int count, CvTrackbarCallback2 on_change, void* userdata CV_DEFAULT(0));
_HIGHGUI_SWITCH_EXPORT_C CvVideoWriter* cvgswCreateVideoWriter(const  char* filename, int fourcc, double fps, CvSize frame_size, int is_color CV_DEFAULT(1));
_HIGHGUI_SWITCH_EXPORT_C IplImage* cvgswDecodeImage( CvMat* buf, int iscolor CV_DEFAULT(CV_LOAD_IMAGE_COLOR));
_HIGHGUI_SWITCH_EXPORT_C CvMat* cvgswDecodeImageM( CvMat* buf, int iscolor CV_DEFAULT(CV_LOAD_IMAGE_COLOR));
_HIGHGUI_SWITCH_EXPORT_C void cvgswDestroyAllWindows();
_HIGHGUI_SWITCH_EXPORT_C void cvgswDestroyWindow(const  char* name);
_HIGHGUI_SWITCH_EXPORT_C CvMat* cvgswEncodeImage(const  char* ext,  CvArr* image, const  int* params CV_DEFAULT(0));
_HIGHGUI_SWITCH_EXPORT_C int cvgswGetCaptureDomain(CvCapture* capture);
_HIGHGUI_SWITCH_EXPORT_C double cvgswGetCaptureProperty(CvCapture* capture, int property_id);
_HIGHGUI_SWITCH_EXPORT_C int cvgswGetTrackbarPos(const  char* trackbar_name, const  char* window_name);
_HIGHGUI_SWITCH_EXPORT_C void* cvgswGetWindowHandle(const  char* name);
_HIGHGUI_SWITCH_EXPORT_C const char* cvgswGetWindowName(void* window_handle);
_HIGHGUI_SWITCH_EXPORT_C double cvgswGetWindowProperty(const  char* name, int prop_id);
_HIGHGUI_SWITCH_EXPORT_C int cvgswGrabFrame(CvCapture* capture);
_HIGHGUI_SWITCH_EXPORT_C int cvgswInitSystem(int argc, char** argv);
_HIGHGUI_SWITCH_EXPORT_C IplImage* cvgswLoadImage(const  char* filename, int iscolor CV_DEFAULT(CV_LOAD_IMAGE_COLOR));
_HIGHGUI_SWITCH_EXPORT_C CvMat* cvgswLoadImageM(const  char* filename, int iscolor CV_DEFAULT(CV_LOAD_IMAGE_COLOR));
_HIGHGUI_SWITCH_EXPORT_C void cvgswMoveWindow(const  char* name, int x, int y);
_HIGHGUI_SWITCH_EXPORT_C int cvgswNamedWindow(const  char* name, int flags CV_DEFAULT(CV_WINDOW_AUTOSIZE));
_HIGHGUI_SWITCH_EXPORT_C IplImage* cvgswQueryFrame(CvCapture* capture);
_HIGHGUI_SWITCH_EXPORT_C void cvgswReleaseCapture(CvCapture** capture);
_HIGHGUI_SWITCH_EXPORT_C void cvgswReleaseVideoWriter(CvVideoWriter** writer);
_HIGHGUI_SWITCH_EXPORT_C void cvgswResizeWindow(const  char* name, int width, int height);
_HIGHGUI_SWITCH_EXPORT_C IplImage* cvgswRetrieveFrame(CvCapture* capture, int streamIdx CV_DEFAULT(0));
_HIGHGUI_SWITCH_EXPORT_C int cvgswSaveImage(const  char* filename,  CvArr* image, const  int* params CV_DEFAULT(0));
_HIGHGUI_SWITCH_EXPORT_C int cvgswSetCaptureProperty(CvCapture* capture, int property_id, double value);
_HIGHGUI_SWITCH_EXPORT_C void cvgswSetMouseCallback(const  char* window_name, CvMouseCallback on_mouse, void* param CV_DEFAULT(NULL));
#ifdef _WINDOWS
_HIGHGUI_SWITCH_EXPORT_C void cvgswSetPostprocessFuncWin32(CvWin32WindowCallback on_postprocess);
_HIGHGUI_SWITCH_EXPORT_C void cvgswSetPreprocessFuncWin32(CvWin32WindowCallback on_preprocess);
#endif
_HIGHGUI_SWITCH_EXPORT_C void cvgswSetTrackbarPos(const  char* trackbar_name, const  char* window_name, int pos);
_HIGHGUI_SWITCH_EXPORT_C void cvgswSetWindowProperty(const  char* name, int prop_id, double prop_value);
_HIGHGUI_SWITCH_EXPORT_C void cvgswShowImage(const  char* name,  CvArr* image);
_HIGHGUI_SWITCH_EXPORT_C int cvgswStartWindowThread();
_HIGHGUI_SWITCH_EXPORT_C int cvgswWaitKey(int delay CV_DEFAULT(0));
_HIGHGUI_SWITCH_EXPORT_C int cvgswWriteFrame(CvVideoWriter* writer,  IplImage* image);
/*........End Declaration.............*/


#endif //__HIGHGUI_SWITCH_H
