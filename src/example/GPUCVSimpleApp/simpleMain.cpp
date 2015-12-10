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

/** @defgroup GPUCV_SIMPLEAPP_GRP Simple GpuCV application
	@ingroup GPUCV_EXAMPLE_LIST_GRP
@{
This small example is used as a test and template application to use GpuCV.
\author Yannick Allusse
\version GpuCV 0.4.2 rev 290

\warning Application working directory must be set to : './gpucv/bin'
\note Lines added to use GPUCV are marked with "//GPUCV" tag.
\note Lines changed to use GPUCV are marked with //GPUCV_SW tag. 
These changes could be avoided by including some of the "GpuCV/c*_wrapper.h" files such as
[cv_wrapper.h/cvaux_wrapper.hcxcore_wrapper.h/highgui_wrapper.h].
\note To enable GPUCV acceleration, set GPUCVSAMPLE_USE_GPUCV to 1. This flag indicates wich part of the the example application to change to have GpuCV ready to work.
*/

#include "StdAfx.h"


#define GPUCVSAMPLE_USE_GPUCV 0

#if GPUCVSAMPLE_USE_GPUCV

#if defined _GPUCV_SUPPORT_SWITCH 
	#include <highgui_switch/highgui_switch_wrapper.h>
	#include <GPUCVSwitch/switch.h>
	#include <cxcore_switch/cxcore_switch_wrapper.h>
	#include <cv_switch/cv_switch_wrapper.h>
	#include <GPUCVSwitch/Cl_FctSw_Mngr.h>
	#include <GPUCVSwitch/Cl_Dll.h>
#else
//#warning ("_GPUCV_SUPPORT_SWITCH not defined...Running on OpenCV only")
	#include <cxcore.h>
	#include <cv.h>
	#include <highgui.h>
#endif
	#include <GPUCV/misc.h>
	using namespace GCV;
	#define EXCEPTION_OBJ SGE::CAssertException
#else
	#include <GPUCV/misc.h>
	using namespace GCV;

	#include <cxcore.h>
	#include <cv.h>
	#include <highgui.h>
	//#define GPUCV_NOTICE(MSG) std::cout << MSG << std::endl;
	#define EXCEPTION_OBJ std::exception
#endif

using namespace std;



#define IMG1_FILE 
#define IMG2_FILE 

CvSize ProcessSize = {256,256};	//!< Image size requested for processing. We will resize input image if smaller.
std::string strImage1 = "pictures/lena512.jpg";
std::string strImage2 = "pictures/plage512.jpg";

bool printError(const char* _errorMsg) 
{
	std::cout << _errorMsg << std::endl;
	return false;
}
/** \brief Parse command line parameters.
Command line options:
<table>
<tr>
<td>Option</td>
<td>Values</td>
<td>Description</td>
</tr>
<tr>
<td>-f videofilname</td>
<td>videofilname: Path to a video file</td>
<td>Open the given video file as input stream</td>
</tr>
<tr>
<td>-c cameraid</td>
<td>cameraid: id of the camera(from 0 to x)</td>
<td>Open the given video camera as input stream</td>
</tr>
<td>-w width - height</td>
<td>width & height: size of the video input</td>
<td>Try to open the video input using the given size, otherwise resize input to fit the given size</td>
</tr>
</table>
*/
bool parseCommandLine(int argc, char * argv[]) 
{
	std::string strCurrentCmd; 
	for(int i = 1; i< argc; i++)
	{
		strCurrentCmd= argv[i];
		
		if(strCurrentCmd=="-q")//quit application, used to see if it compiles and run. Further tests must be done
		{
			//printError("Exiting application");
			exit(0);
		}
		if(strCurrentCmd=="-w")//input width
		{
			if(i+1<argc)
			{
				ProcessSize.width = atoi(argv[++i]);
				continue;
			}
			else
				return printError("Error while parsing arguments: missing width value");
		}
		if(strCurrentCmd=="-h")//input height
		{
			if(i+1<argc)
			{
				ProcessSize.height = atoi(argv[++i]);
				continue;
			}
			else
				return printError("Error while parsing arguments: missing height value");
		}
		if(strCurrentCmd=="-f1")//video file
		{
			if(i+1<argc)
			{
				strImage1 = argv[++i];
				continue;
			}
			else
				return printError("Error while parsing arguments: missing video filename");
		}
		else if(strCurrentCmd=="-f2")//video file
		{
			if(i+1<argc)
			{
				strImage2 = argv[++i];
				continue;
			}
			else
				return printError("Error while parsing arguments: missing video filename");
		}
		else //we do not know the parameter, we suppose it is a filename
			strImage1 = argv[i];
	}
	return true;
}


int main(int argc, char** argv)
{
#if defined _GPUCV_SUPPORT_SWITCH 
#else
#pragma warning ("_GPUCV_SUPPORT_SWITCH not defined...Running on OpenCV only")
#endif
	if(parseCommandLine(argc,argv)==false)
	{
		GPUCV_ERROR("Exiting application...");
		exit(-1);
	}
	try//exception handling is recommended...
	{
		//get application path
		std::string AppPath;
		GPUCV_NOTICE("Current application path: " << argv[0]);
		AppPath = cvgRetrieveShaderPath(argv[0]);
		GPUCV_NOTICE("GpuCV shader and data path: "<< AppPath);
		//=============================================

		//init GpuCV
#if GPUCVSAMPLE_USE_GPUCV 
	#if GPUCVSAMPLE_USE_CUDA
			cvgcuInit(true, true); //GPUCV, init CUDA version. GLSL operator are still available
	#else
			cvgInit(true, false); //GPUCV, init GLSL version only.
	#endif

		GetGpuCVSettings()->SetOption(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG,1);//GPUCV
		GetGpuCVSettings()->SetOption(GpuCVSettings::GPUCV_SETTINGS_SWITCH_LOG,1);//GPUCV
		GetGpuCVSettings()->SetOption(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_NOTICE,1);//GPUCV
		GetGpuCVSettings()->SetOption(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_WARNING,1);//GPUCV
#endif

		//Init images
		std::string ImagePath=AppPath;
		ImagePath+=strImage1;		
		GPUCV_NOTICE("Loading input images");
		IplImage * Img1 = cvLoadImage(ImagePath.data());
		SG_AssertFile(Img1,ImagePath, "Could not load file");

		ImagePath=AppPath;
		ImagePath+=strImage2;
		IplImage * Img2 = cvLoadImage(ImagePath.data());
		SG_AssertFile(Img2,ImagePath, "Could not load file");

		//choose to resize or not
		GPUCV_NOTICE("Resizing images");
		CvSize InputSize;
		InputSize.width = Img1->width;
		InputSize.height = Img1->height;

		if(ProcessSize.width!=InputSize.width 
			||
			ProcessSize.height!=InputSize.height)
		{
			IplImage * TmpImage = cvCreateImage(ProcessSize, Img1->depth, Img1->nChannels);
			cvResize(Img1, TmpImage);//, CV_INTER_CUBIC);//CV_INTER_CUBIC is done on CPU...
			cvReleaseImage(&Img1);
			Img1 = TmpImage;

			TmpImage = cvCreateImage(ProcessSize, Img2->depth, Img2->nChannels);
			cvResize(Img2, TmpImage);//, CV_INTER_CUBIC);//CV_INTER_CUBIC is done on CPU...
			cvReleaseImage(&Img2);
			Img2 = TmpImage;
		}

		GPUCV_NOTICE("Create output image...");
		IplImage * Dst = cvCreateImage(cvGetSize(Img1), Img1->depth, Img1->nChannels);
		IplImage * Dst2 = cvCreateImage(cvGetSize(Img1), Img1->depth, Img1->nChannels);

		//perform operation		
		GPUCV_NOTICE("Start processing...");
		cvNamedWindow("Img1", 1);
		cvNamedWindow("Img2", 1);
		cvNamedWindow("Result", 1);
		
		for(int i =0; i < 20; i++)
		{

			GPUCV_NOTICE("\tAdd...");
			cvAdd(Img1, Img2, Dst);//GPUCV_SW
			GPUCV_NOTICE("\tCvtColor...");
			cvCvtColor(Dst, Dst2, CV_BGR2XYZ);
		}

		GPUCV_NOTICE("Stop processing...");
		GPUCV_NOTICE("Show images...");
		cvShowImage("Img1", Img1);//GPUCV_SW
		cvShowImage("Img2", Img2);//GPUCV_SW
		cvShowImage("Result", Dst2);//GPUCV_SW

		
		cvWaitKey(0);


		//release data
		GPUCV_NOTICE("Release images...");
		cvReleaseImage(&Img1);//GPUCV_SW
		cvReleaseImage(&Img2);//GPUCV_SW
		cvReleaseImage(&Dst);//GPUCV_SW
		cvReleaseImage(&Dst2);//GPUCV_SW

		//release windows
		GPUCV_NOTICE("Release windows...");
		cvDestroyWindow("Img1");
		cvDestroyWindow("Img2");
		cvDestroyWindow("Result");
	}

	catch(EXCEPTION_OBJ &e)
	{
		GPUCV_NOTICE("=================== Exception catched Start =================");
		GPUCV_NOTICE(e.what());
		GPUCV_NOTICE("=================== Exception catched End =================");
		GPUCV_NOTICE("Press ENTER to continue...");
		getchar();
	}
	
	GPUCV_NOTICE("Shut down application...");
	
#if GPUCVSAMPLE_USE_GPUCV 
#if defined _GPUCV_SUPPORT_SWITCH 
	cvgswPrintAllFctStats();
	cvgswPrintFctStats("cvAdd");
	CL_FctSw_Mngr::GetSingleton()->XMLSaveToFile("gcv_FctSwManager.xml");
	DllManager::GetSingleton()->XMLSaveToFile("gcv_dlls.xml");
#endif
	cvgTerminate();//GPUCV
#endif
	GPUCV_NOTICE("Press ENTER to end...");
	getchar();
	return 0;
}
/**@}*/ //GPUCV_SIMPLEAPP_GRP
