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

/** @defgroup GPUCV_SIMPLE_C_INTEGRATION_GRP Simple integration of GpuCV into a C application
	@ingroup GPUCV_EXAMPLE_LIST_GRP
@{
This small example is used as a test application to control the C compatibility of GpuCV.
The application is currently empty but control GpuCV include files.
\author Yannick Allusse
\version GpuCV 1.0 rev 540
\todo Add example of algorithm in C.
*/

#include "StdAfx.h"

#ifdef _GPUCV_SUPPORT_SWITCH //switch and wrapper
	#include <cxcore_switch/cxcore_switch_wrapper.h>
	#include <cv_switch/cv_switch_wrapper.h>
	#include <highgui_switch/highgui_switch_wrapper.h>
	#include <GPUCV/misc.h>
#else

//opencv classic
#include <cxcore.h>
#include <cv.h>
#include <highgui.h>

//OpenGL
#include <highguig/highguig.h>
#include <cxcoreg/cxcoreg.h>

#include <cvg/cvg.h>

//CUDA
#include <cxcoregcu/cxcoregcu.h>
#include <cvgcu/cvgcu.h>
#endif
//switch
/*
#include <highgui_switch/highgui_switch_wrapper.h>
#include <cxcore_switch/cxcore_switch_wrapper.h>
#include <cv_switch/cv_switch_wrapper.h>
*/
int main(int argc, char** argv)
{
#if 1
	//declare variable first
		IplImage * Img1 = NULL;
		IplImage * Img2 = NULL;
		IplImage * Dst  = NULL;

	//init gpucv
	
		//GPUCV_NOTICE("Set shader application path: " << AppPath);
		printf("Shader path: %s\n", cvgRetrieveShaderPath(argv[0]));
		//init GpuCV
		cvgswInit(1, 1);
		//GPUCV, init CUDA version. GLSL operator are still available
		
		//Init images
		Img1 = cvCreateImage(cvSize(256,256), IPL_DEPTH_8U, 3);
		Img2 = cvCreateImage(cvSize(256,256), IPL_DEPTH_8U, 3);
		Dst  = cvCreateImage(cvGetSize(Img1), Img1->depth, Img1->nChannels);

		cvSet(Img1, cvScalar(100,0,0,0),NULL);
		cvSet(Img2, cvScalar(0,100,0,0),NULL);
		cvSet(Dst, cvScalar(0,0,0,0),NULL);
		
		//perform operation		
		//GPUCV_NOTICE("Start processing...");
		//for(int i =0; i < 20; i+=1)
		{
			//GPUCV_NOTICE("\tAdd...");
			cvAdd(Img1, Img2, Dst, NULL);//GPUCV_SW
		}
		//GPUCV_NOTICE("Stop processing...");
		//GPUCV_NOTICE("Show images...");
		cvNamedWindow("Img1", 1);
		cvNamedWindow("Img2", 1);
		cvNamedWindow("Result", 1);
		cvShowImage("Img1", Img1);//GPUCV_SW
		cvShowImage("Img2", Img2);//GPUCV_SW
		cvShowImage("Result", Dst);//GPUCV_SW
 		cvWaitKey(0);
		//release data
//		GPUCV_NOTICE("Release images...");
		cvReleaseImage(&Img1);//GPUCV_SW
		cvReleaseImage(&Img2);//GPUCV_SW
		cvReleaseImage(&Dst);//GPUCV_SW

		//release windows
	//	GPUCV_NOTICE("Release windows...");
		cvDestroyWindow("Img1");
		cvDestroyWindow("Img2");
		cvDestroyWindow("Result");
#endif
	return 0;
}
/**@}*/ //GPUCV_SIMPLE_C_INTEGRATION_GRP
