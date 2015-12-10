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




/** @defgroup GPUCV_CAMDEMO_GRP GpuCVCamDemo
	@ingroup GPUCV_EXAMPLE_LIST_GRP
	@{
This application is used to show real time processing of video stream using different processing
implementations (Native OpenCV, GpuCV with GLSL, GpuCV with CUDA). See \ref parseCommandLine() for command line options and \ref CBkeys() for live key commands.

	\warning ALL cv* function used into this demo application are rederected directly to GpuCV operators unless _DEMO_FORCE_NATIVE_OPENCV is set to 1.
	using the files "GPUCV/cv_wrapper.h", "GPUCV/cxcore_wrapper.h", "GPUCV/highgui_wrapper.h".
	\warning ALL the cvg* operator are pure GpuCV operators that have no equivalent in OpenCV.
	\author Yannick Allusse
	\version GpuCV 1.0.0 rev 562
*/

/** \brief Force the GpuCVCam demo to compile without using any GPU functions.
*/
#ifndef _GPUCV_SUPPORT_SWITCH
#	define _DEMO_FORCE_NATIVE_OPENCV 1
#else
#	define _DEMO_FORCE_NATIVE_OPENCV 0
#endif

#define _CAMDEBUG 0
#define _GPUCV_CAM_DEMO__USE_THREAD 0
 
#if _DEMO_FORCE_NATIVE_OPENCV
	#include <cv.h>
	#include <cxcore.h>
	#include <highgui.h>
	#include <cvaux.h>
	#include <GPUCV/config.h>//does not include any GpuCV operators, but only usefull macro for debugging.
#else	
	#include <GPUCVSwitch/switch.h>
	//#include <GPUCVCuda/cuda_misc.h>
	#include <GPUCV/misc.h>
#endif

#include <GPUCV/cv_new.h>
//include webcam
#if  CV_MAJOR_VERSION < 2
	#include <cvcam.h>
#else
	#include <highgui.h>
#endif
#include "SugoiTools/MultiPlat.h"


#if !_DEMO_FORCE_NATIVE_OPENCV
#include <cv_switch/cv_switch_wrapper.h>//they must be the latest GpuCV files to be included....
#include <cxcore_switch/cxcore_switch_wrapper.h>
#include <highgui_switch/highgui_switch_wrapper.h>
#endif

//defines
#define CAM_DEMO_PROFILE 0
//================================================================================


//GLUT callback functions
//================================================================================
	void CBDisplay();
	void CBIdle();
	void CBkeys(unsigned char key, int x, int y);
	void CBChangeSize(int w, int h);
//================================================================================



//Application functions definition
//================================================================================
	void InitGlut(int argc, char **argv);
	void OpenVideoInput(const std::string _videoFile, char _cCameraID=-1);
	void InitProcessing();
	IplImage * GrabInputFrame(IplImage * inputImage);
//	void RemoveBackground (IplImage* im, IplImage* bg, IplImage* output);


//typedef bool (*)(void) FctType_DftCB;

struct GpuCVProcessing{
	bool (*ProcessInit)(void);
	bool (*Process)(void);
	bool (*ProcessClean)(void);
	bool (*ProcessSwitch)(GCV::BaseImplementation _type);
};

void CallDemoFilter();

//processing function declarattion
//Morphology
	bool MorphoProcess();
	bool MorphoInit();
	bool MorphoClean();
	bool MorphoSwitch(GCV::BaseImplementation _type);
//background substraction
	bool BackGroundProcess();
	bool BackGroundInit();
	bool BackGroundClean();
//Deriche filter
	bool DericheProcess();
	bool DericheInit();
	bool DericheClean();
	bool DericheSwitch(GCV::BaseImplementation _type);
//Lut filter
	bool LutProcess();
	bool LutInit();
	bool LutClean();
	bool LutSwitch(GCV::BaseImplementation _type);
//Lut filter
	bool ArithmProcess();
	bool ArithmInit();
	bool ArithmClean();
	bool ArithmSwitch(GCV::BaseImplementation _type);
//Sobel filter
	bool SobelProcess();
	bool SobelInit();
	bool SobelClean();
	bool SobelSwitch(GCV::BaseImplementation _type);
//
//

//================================================================================
//Macros
#define DEBUG_FCT(TXT) GPUCV_DEBUG(GPUCV_GET_FCT_NAME() << ":"<< TXT);
//================================================================================
//All images
extern IplImage* imageCam;
extern IplImage* imageSrc;
extern IplImage* imageDst;
extern IplImage* imageTemp;
extern IplImage *BackGrnd_Img;
extern IplImage *ForGrnd_Img;
//================================================================================
//State variables 
extern bool bVideoFinished;
extern bool bPerformProcessing;
extern std::string strLastFctCalled;
//================================================================================



/** @}*/ //GPUCV_CAMDEMO_GRP






