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
#include "StdAfx.h"
#include <GPUCVSwitch/macro.h>
#include <GPUCVCore/GpuTextureManager.h>
#include <GPUCVSwitch/Cl_Dll.h>
#include <GPUCVSwitch/switch.h>

#define _GPUCV_FORCE_OPENCV_NP 1
#include <includecv.h>
#include <highgui.h>


using namespace std;
using namespace GCV;
#define CVAPI(MSG) MSG

#include <highgui_switch/highgui_switch.h>
#include <GPUCVSwitch/switch.h>
/*====================================*/
void cvg_highgui_switch_RegisterTracerSingletons(SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> * _pAppliTracer, SG_TRC::CL_TRACING_EVENT_LIST *_pEventList)
{
SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance().RegisterNewSingleton(_pAppliTracer);
SG_TRC::CL_TRACING_EVENT_LIST::Instance().RegisterNewSingleton(_pEventList);
}
/*====================================*/

/*====================================*/
int cvgswInitSystem(int argc, char** argv)
{
	return cvInitSystem((int) argc, (char**) argv);
}


/*====================================*/
int cvgswStartWindowThread()
{
	return cvStartWindowThread();
}


/*====================================*/
int cvgswNamedWindow(const  char* name, int flags)
{
	return cvNamedWindow((const  char*) name, (int) flags);
}


/*====================================*/
void cvgswSetWindowProperty(const  char* name, int prop_id, double prop_value)
{
	cvSetWindowProperty((const  char*) name, (int) prop_id, (double) prop_value);
}


/*====================================*/
double cvgswGetWindowProperty(const  char* name, int prop_id)
{
	return cvGetWindowProperty((const  char*) name, (int) prop_id);
}


/*====================================*/
void cvgswShowImage(const  char* name,  CvArr* image)
{
	typedef void(*_ShowImage) (const  char*,  CvArr* ); 
	GPUCV_FUNCNAME("cvShowImage");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((const  char*) name, ( CvArr*) image), _ShowImage, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswResizeWindow(const  char* name, int width, int height)
{
	cvResizeWindow((const  char*) name, (int) width, (int) height);
}


/*====================================*/
void cvgswMoveWindow(const  char* name, int x, int y)
{
	cvMoveWindow((const  char*) name, (int) x, (int) y);
}


/*====================================*/
void cvgswDestroyWindow(const  char* name)
{
	cvDestroyWindow((const  char*) name);
}


/*====================================*/
void cvgswDestroyAllWindows()
{
	cvDestroyAllWindows();
}


/*====================================*/
void* cvgswGetWindowHandle(const  char* name)
{
	return cvGetWindowHandle((const  char*) name);
}


/*====================================*/
const char* cvgswGetWindowName(void* window_handle)
{
	return cvGetWindowName((void*) window_handle);
}


/*====================================*/
int cvgswCreateTrackbar(const  char* trackbar_name, const  char* window_name, int* value, int count, CvTrackbarCallback on_change)
{
	return cvCreateTrackbar((const  char*) trackbar_name, (const  char*) window_name, (int*) value, (int) count, (CvTrackbarCallback) on_change);
}


/*====================================*/
int cvgswCreateTrackbar2(const  char* trackbar_name, const  char* window_name, int* value, int count, CvTrackbarCallback2 on_change, void* userdata)
{
	return cvCreateTrackbar2((const  char*) trackbar_name, (const  char*) window_name, (int*) value, (int) count, (CvTrackbarCallback2) on_change, (void*) userdata);
}


/*====================================*/
int cvgswGetTrackbarPos(const  char* trackbar_name, const  char* window_name)
{
	return cvGetTrackbarPos((const  char*) trackbar_name, (const  char*) window_name);
}


/*====================================*/
void cvgswSetTrackbarPos(const  char* trackbar_name, const  char* window_name, int pos)
{
	cvSetTrackbarPos((const  char*) trackbar_name, (const  char*) window_name, (int) pos);
}


/*====================================*/
void cvgswSetMouseCallback(const  char* window_name, CvMouseCallback on_mouse, void* param)
{
	cvSetMouseCallback((const  char*) window_name, (CvMouseCallback) on_mouse, (void*) param);
}


/*====================================*/
IplImage* cvgswLoadImage(const  char* filename, int iscolor)
{
	typedef IplImage*(*_LoadImage) (const  char*, int ); 
	GPUCV_FUNCNAME("cvLoadImage");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	IplImage* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((const  char*) filename, (int) iscolor), _LoadImage,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswLoadImageM(const  char* filename, int iscolor)
{
	typedef CvMat*(*_LoadImageM) (const  char*, int ); 
	GPUCV_FUNCNAME("cvLoadImageM");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((const  char*) filename, (int) iscolor), _LoadImageM,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
int cvgswSaveImage(const  char* filename,  CvArr* image, const  int* params)
{
	typedef int(*_SaveImage) (const  char*,  CvArr*, const  int* ); 
	GPUCV_FUNCNAME("cvSaveImage");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((const  char*) filename, ( CvArr*) image, (const  int*) params), _SaveImage,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
IplImage* cvgswDecodeImage( CvMat* buf, int iscolor)
{
	typedef IplImage*(*_DecodeImage) ( CvMat*, int ); 
	GPUCV_FUNCNAME("cvDecodeImage");
	CvArr* SrcARR[] = { (CvArr*) buf};
	CvArr** DstARR = NULL;
	IplImage* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvMat*) buf, (int) iscolor), _DecodeImage,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswDecodeImageM( CvMat* buf, int iscolor)
{
	typedef CvMat*(*_DecodeImageM) ( CvMat*, int ); 
	GPUCV_FUNCNAME("cvDecodeImageM");
	CvArr* SrcARR[] = { (CvArr*) buf};
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvMat*) buf, (int) iscolor), _DecodeImageM,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswEncodeImage(const  char* ext,  CvArr* image, const  int* params)
{
	typedef CvMat*(*_EncodeImage) (const  char*,  CvArr*, const  int* ); 
	GPUCV_FUNCNAME("cvEncodeImage");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((const  char*) ext, ( CvArr*) image, (const  int*) params), _EncodeImage,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswConvertImage( CvArr* src, CvArr* dst, int flags)
{
	typedef void(*_ConvertImage) ( CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvConvertImage");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) flags), _ConvertImage, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswWaitKey(int delay)
{
	return cvWaitKey((int) delay);
}


/*====================================*/
CvCapture* cvgswCreateFileCapture(const  char* filename)
{
	return cvCreateFileCapture((const  char*) filename);
}


/*====================================*/
CvCapture* cvgswCreateCameraCapture(int index)
{
	return cvCreateCameraCapture((int) index);
}


/*====================================*/
int cvgswGrabFrame(CvCapture* capture)
{
	return cvGrabFrame((CvCapture*) capture);
}


/*====================================*/
IplImage* cvgswRetrieveFrame(CvCapture* capture, int streamIdx)
{
	typedef IplImage*(*_RetrieveFrame) (CvCapture*, int ); 
	GPUCV_FUNCNAME("cvRetrieveFrame");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	IplImage* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((CvCapture*) capture, (int) streamIdx), _RetrieveFrame,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
IplImage* cvgswQueryFrame(CvCapture* capture)
{
	typedef IplImage*(*_QueryFrame) (CvCapture* ); 
	GPUCV_FUNCNAME("cvQueryFrame");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	IplImage* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((CvCapture*) capture), _QueryFrame,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswReleaseCapture(CvCapture** capture)
{
	cvReleaseCapture((CvCapture**) capture);
}


/*====================================*/
double cvgswGetCaptureProperty(CvCapture* capture, int property_id)
{
	return cvGetCaptureProperty((CvCapture*) capture, (int) property_id);
}


/*====================================*/
int cvgswSetCaptureProperty(CvCapture* capture, int property_id, double value)
{
	return cvSetCaptureProperty((CvCapture*) capture, (int) property_id, (double) value);
}


/*====================================*/
int cvgswGetCaptureDomain(CvCapture* capture)
{
	return cvGetCaptureDomain((CvCapture*) capture);
}


/*====================================*/
CvVideoWriter* cvgswCreateVideoWriter(const  char* filename, int fourcc, double fps, CvSize frame_size, int is_color)
{
	return cvCreateVideoWriter((const  char*) filename, (int) fourcc, (double) fps, (CvSize) frame_size, (int) is_color);
}


/*====================================*/
int cvgswWriteFrame(CvVideoWriter* writer,  IplImage* image)
{
	typedef int(*_WriteFrame) (CvVideoWriter*,  IplImage* ); 
	GPUCV_FUNCNAME("cvWriteFrame");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((CvVideoWriter*) writer, ( IplImage*) image), _WriteFrame,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswReleaseVideoWriter(CvVideoWriter** writer)
{
	cvReleaseVideoWriter((CvVideoWriter**) writer);
}


/*====================================*/
#ifdef _WINDOWS
void cvgswSetPreprocessFuncWin32(CvWin32WindowCallback on_preprocess)
{
	cvSetPreprocessFuncWin32((CvWin32WindowCallback) on_preprocess);
}


/*====================================*/
void cvgswSetPostprocessFuncWin32(CvWin32WindowCallback on_postprocess)
{
	cvSetPostprocessFuncWin32((CvWin32WindowCallback) on_postprocess);
}
#endif
/*........End Code.............*/

