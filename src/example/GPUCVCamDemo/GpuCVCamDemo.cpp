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
#include "GpuCVCamDemo.h"
#include <pthread.h>
#include <SugoiPThread/mutex.h>
#ifndef _GPUCV_GL_USE_GLUT
#	include <GLUT/glut.h> 
#endif

using namespace GCV;

#if _CAMDEBUG
CvSize ImageSize = {256,256};	//!< Image size requested for processing. We will resize input image if smaller.
#else
CvSize ImageSize = {0,0};	//!< Image size requested for processing. We will resize input image if smaller.
#endif
CvSize ImageSize_Next = {ImageSize.width,ImageSize.height};	//!< Store the image size defined by the window size. Changing the window size will also change default image resolution.


//All images
//================================================================================
IplImage* imageCam = NULL;	//!< Image from the video input.
IplImage* imageSrc = NULL;	//!< Image resized and used as source image for morphology.
IplImage* imageDst = NULL;	//!< Destination image for morphology.
IplImage* imageTemp = NULL; //!< Temp image for morphology.


// a thread retrieve images from video input and copy them into vInputImages up to INPUT_IMAGE_STACK_SIZE images, when vInputImages is full -> the thread wait.
#if _GPUCV_CAM_DEMO__USE_THREAD
	#define INPUT_IMAGE_STACK_SIZE 10
	std::vector <IplImage*>	vInputImages;	//vector of input images to process
	std::vector <IplImage*>	vFreeImages;	//vector of free image that will be used again to copie video input frame
	pthread_mutex_t MutexInputImages;		//mutex to manage access to both vectors of images
	void *ThreadLoopFunction(void * _ptr);
#endif


/** Contains the list of all algorithm implemented in the CamDemo and their respective functions*/
GpuCVProcessing CamDemoFilters[]={
	{MorphoInit,MorphoProcess,MorphoClean, MorphoSwitch}
	,{LutInit,LutProcess,LutClean, LutSwitch}
	,{ArithmInit,ArithmProcess,ArithmClean, ArithmSwitch}
	,{SobelInit,SobelProcess,SobelClean, SobelSwitch}
	,{DericheInit,DericheProcess,DericheClean, DericheSwitch}

	//	,{BackGroundInit,BackGroundProcess,BackGroundClean}
};
int FilterNbr = sizeof(CamDemoFilters)/sizeof(GpuCVProcessing);
int FilterID = 0;		
pthread_t  ThreadPtr;//thread used to grab frames

//State variables =================================
bool bVideoFinished		= false;	//!< Flag used to know when the video is finished.
bool bPerformProcessing	= true;
bool bGLoutput			= true;		//!< If enable draw the GPUCV result images to the OpenGL windows, if false get back GPUCV GPU images to ram to show with cvShowImage
bool bEnableGLWindowsResize = false;	//!< Resizing the OpenGL windows will resize input image, this must not be allowed on the first frames cause we do not know the input image size before drawing the first frame.
bool bEnableGrabFrame = true;		//!< Use to disable grabbing input frame, when false processing continues on the same image
BaseImplementation CVG_STATUS = GPUCV_IMPL_OPENCV;//!< Flag to know if we start with OpenCV/GpuCV GLSL / GpuCV CUDA implementations.
bool bNeedResize = false;
bool bEnableResize = false;
bool bFileIsVideo = true;
int iterac = 1;						//!< Iteration number for morphology.
//=================================================




//================================================================================
//Application main variables
CvCapture* VideoCapture = NULL;	//!< Main capture input source.
std::string AppPath="";			//!< Path of the executable file.
std::string VideoSeqFile = "";	//!< Filepath of the video to load.
char iCameraID = -1;
std::string LabelOpenCVFile = "pictures/opencv-logo2.png";//"pictures/processed_by_opencv.bmp";//!< Image file use to show that we are using OpenCV.
std::string LabelGpuCVFile = "pictures/ogl.jpg";//"pictures/processed_by_gpucv.bmp";//!< Image file use to show that we are using GpuCV.
std::string LabelGCUDAFile = "pictures/nvidia_cuda_logo.jpg";//"pictures/processed_by_gpucv_cuda.bmp";//!< Image file use to show that we are using GpuCV.
IplImage *MSG_CV = NULL;		//!< OpenCV label image.
IplImage *MSG_CVG = NULL;		//!< GpuCV-Glsl label image.
IplImage *MSG_CVGCU = NULL;		//!< GpuCV-Cuda label image.

IplImage* CurrentImplLabel = NULL;
std::string strLastFctCalled = "";


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
				ImageSize.width = atoi(argv[++i]);
				ImageSize_Next.width = ImageSize.width;
				continue;
			}
			else
				return printError("Error while parsing arguments: missing width value");
		}
		if(strCurrentCmd=="-h")//input height
		{
			if(i+1<argc)
			{
				ImageSize.height = atoi(argv[++i]);
				ImageSize_Next.height = ImageSize.height;
				continue;
			}
			else
				return printError("Error while parsing arguments: missing height value");
		}
		if(strCurrentCmd=="-c")//camera id
		{
			if(i+1<argc)
			{
				iCameraID = atoi(argv[++i]);
				continue;
			}
			else
				return printError("Error while parsing arguments: missing camera id value");
		}
		if(strCurrentCmd=="-f")//input file
		{
			if(i+1<argc)
			{
				VideoSeqFile = argv[++i];
				if(VideoSeqFile.find(".avi")!=std::string::npos)
				{
					bFileIsVideo=true;
				}
				else if(VideoSeqFile.find(".mpg")!=std::string::npos)
				{
					bFileIsVideo=true;
				}
				else if(VideoSeqFile.find(".wmv")!=std::string::npos)
				{
					bFileIsVideo=true;
				}
				else
				{
					bFileIsVideo=false;
				}
				continue;
			}
			else
				return printError("Error while parsing arguments: missing video filename");
		}
		else //we do not know the parameter, we suppose it is a filename
			VideoSeqFile = argv[i];
	}
	return true;
}

//================================================================================
/** \brief Process input keys on the OpenGL window.
Input keys:
<table>
<tr>
<td>Key</td>
<td>Description</td>
</tr>
<tr>
<td>from '1' to '9'</td>
<td>Select the correponding algorithm to run on video stream</td>
</tr>
<tr>
<td>ESC / 'Q' / 'q'</td>
<td>Exit the demo</td>
</tr>
<tr>
<td>'C' / 'c'</td>
<td></td>
</tr>
<tr>
<td>'p'</td>
<td>Enable/disable processing of the input video stream</td>
</tr>
<tr>
<td>'g'</td>
<td>Enable rendering the video ouput with OpenGL. Default is OPENCV render to CV window and GPUCV render to OpenGL window.</td>
</tr>
<tr>
<td>SPACE</td>
<td>Cycle processing implementation from OPENCV->GLSL->CUDA->OPENCV->...</td>
</tr>
<tr>
<td>'d'</td>
<td>Enable/Disable debugging log output. It impacts on performances.</td>
</tr>
<tr>
<td>'s'</td>
<td>Enable/Disable switching log output. It impacts on performances.</td>
</tr>
<tr>
<td>'+'/'-'</td>
<td>Increase/Decrease loop number used by some processing algorithms.</td>
</tr>
</table>
*/
void CBkeys(unsigned char key, int x, int y) 
{	
	GPUCV_FUNCNAME("CBkeys");
	DEBUG_FCT("");

	if(key<='9' && key >='0')
	{
		int NewFilterID = key-'0'-1;
		if(NewFilterID >=0 && NewFilterID < FilterNbr)
		{
			CamDemoFilters[FilterID].ProcessClean();
			FilterID  = NewFilterID;
			CamDemoFilters[FilterID].ProcessInit();
			CamDemoFilters[FilterID].ProcessSwitch(CVG_STATUS);
		}		
	}
	switch (key)
	{
	case 27  ://escape Key
	case 'Q' :
	case 'q' : 
#if !_DEMO_FORCE_NATIVE_OPENCV
		cvgswPrintAllFctStats();
#endif
		CamDemoFilters[FilterID].ProcessClean();
		GPUCV_NOTICE("Press a key to exit");
		getchar();
		exit(0);
		break;
	case 'C' : 	
	case 'c' : 	
#if CAM_DEMO_PROFILE
		if(AppliTracer()->GetConsoleStatus()) 
			AppliTracer()->DisableConsole();
		else
			AppliTracer()->EnableConsole();
		break;
#endif
	case 'p' :
		bPerformProcessing = !bPerformProcessing;
		break;
	case 'g' :
		bGLoutput = !bGLoutput;
		break;
	case 'r' :
		bEnableResize = !bEnableResize;
		break;
	case 'i':
		bEnableGrabFrame = !bEnableGrabFrame;
		break;
	case 'a' ://auto-switch
		CVG_STATUS=GPUCV_IMPL_AUTO;
		std::cout << std::endl << "Using Auto-Switch" << std::endl;
		#if !_DEMO_FORCE_NATIVE_OPENCV
			CamDemoFilters[FilterID].ProcessSwitch(CVG_STATUS);
			cvgswSetGlobalImplementation(CVG_STATUS);
		#endif
		break;
	case ' ' :
#if 1
		if (CVG_STATUS==GPUCV_IMPL_GLSL)
		{
			CVG_STATUS=GPUCV_IMPL_CUDA;
			std::cout << std::endl << "Using GpuCV-CUDA" << std::endl;
			CurrentImplLabel = MSG_CVGCU;
		}
		else if((CVG_STATUS==GPUCV_IMPL_CUDA) || (CVG_STATUS==GPUCV_IMPL_AUTO))
		{
			CVG_STATUS=GPUCV_IMPL_OPENCV;
			std::cout << std::endl << "Using OpenCV" << std::endl;
			CurrentImplLabel = MSG_CV;
		}
		else if(CVG_STATUS==GPUCV_IMPL_OPENCV)
		{
			CVG_STATUS=GPUCV_IMPL_GLSL;
			std::cout << std::endl << "Using GpuCV-GLSL" << std::endl;
			CurrentImplLabel = MSG_CVG;
		}
#else//disable opengl
		if(CVG_STATUS==GPUCV_IMPL_CUDA)
		{
			CVG_STATUS=GPUCV_IMPL_OPENCV;
			std::cout << std::endl << "Using OpenCV" << std::endl;
			CurrentImplLabel = MSG_CV;
		}
		else if(CVG_STATUS==GPUCV_IMPL_OPENCV)
		{
			CVG_STATUS=GPUCV_IMPL_CUDA;
			std::cout << std::endl << "Using GpuCV-CUDA" << std::endl;
			CurrentImplLabel = MSG_CVGCU;
		}
#endif
		#if !_DEMO_FORCE_NATIVE_OPENCV
			CamDemoFilters[FilterID].ProcessSwitch(CVG_STATUS);
			cvgswSetGlobalImplementation(CVG_STATUS);
		#endif
		break;
	case 'd':
		//inverse debug option
		GetGpuCVSettings()->SetOption(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG, 
				!GetGpuCVSettings()->GetOption(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG));
		GetGpuCVSettings()->SetOption(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_WARNING, 
				!GetGpuCVSettings()->GetOption(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_WARNING));
		
		break;
	case 's':
		//inverse switch log option
		GetGpuCVSettings()->SetOption(GpuCVSettings::GPUCV_SETTINGS_SWITCH_LOG, 
				!GetGpuCVSettings()->GetOption(GpuCVSettings::GPUCV_SETTINGS_SWITCH_LOG));
		break;
	case '+' : 
		if(++iterac > 10) iterac = 10;
		GPUCV_NOTICE("Nbr of iterations:" << iterac);
		break;
	case '-' : 
		if(--iterac < 1) iterac = 1;
		GPUCV_NOTICE("Nbr of iterations:" << iterac);
		break;
	}
}
//================================================================================
int main(int argc, char **argv) 
{
	GPUCV_FUNCNAME("GpuCVCamDemo");
	DEBUG_FCT("");
	//parse command line options...
	if(parseCommandLine(argc,argv)==false)
	{
		GPUCV_ERROR("Exiting application...");
		exit(-1);
	}

	try
	{
 
		InitGlut(argc, argv);	//Init OpenGL and GLUT
#if !_DEMO_FORCE_NATIVE_OPENCV
		//First step: get application path
		std::string AppPath;
		GPUCV_NOTICE("Current application path: " << argv[0]);
		AppPath = cvgRetrieveShaderPath(argv[0]);
		GPUCV_NOTICE("GpuCV shader and data path: "<< AppPath);
		//=============================================

		//then init gpucv
		cvg_cv_switch_RegisterTracerSingletons(&SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance(), &SG_TRC::CL_TRACING_EVENT_LIST::Instance());
		cvg_cxcore_switch_RegisterTracerSingletons(&SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance(), &SG_TRC::CL_TRACING_EVENT_LIST::Instance());
		cvg_highgui_switch_RegisterTracerSingletons(&SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance(), &SG_TRC::CL_TRACING_EVENT_LIST::Instance());
		//cvg_CVAUX_SWITCH_RegisterTracerSingletons(&SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance(), &SG_TRC::CL_TRACING_EVENT_LIST::Instance());
		cvg_switch_RegisterTracerSingletons(&SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance(), &SG_TRC::CL_TRACING_EVENT_LIST::Instance());
		
		bool bMultithreading = false;
                cvgswInit(false, bMultithreading);//Init GPUCV
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_RISE_EXCEPTION, true);
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_CHECK, true);

		
		//Select processing mode CV/CVG, default is auto.
		cvgswSetGlobalImplementation(CVG_STATUS);
#endif
                //enable/disable switch log
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_SWITCH_LOG, (0)?true:false);
                //enable/disable global log
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG 
						|GpuCVSettings::GPUCV_SETTINGS_GLOBAL_WARNING
						, (0)?true:false);

		cvUseOptimized(true);

		OpenVideoInput(VideoSeqFile, iCameraID);//open video input from file or webcam

		imageSrc = GrabInputFrame(NULL);
		InitProcessing();//init OpenCV objects

		glutMainLoop();//start main GLUT loop
	}
	catch(SGE::CAssertException &e)
	{//catch any exceptions and log message
		GPUCV_NOTICE("=================== Exception catched Start =================");
		GPUCV_NOTICE(e.what());
		GPUCV_NOTICE("=================== Exception catched End =================");
		GPUCV_NOTICE("Press a key to continue...");
//		SGE::Sleep(5000);
		getchar();	
#if !_DEMO_FORCE_NATIVE_OPENCV
		cvgTerminate();
	}
	cvgTerminate();
#else
	}
	cvReleaseCapture(&VideoCapture);
	cvDestroyWindow("MainWindow");
#endif
}
//================================================================================
void InitGlut(int argc, char **argv)
{
	GPUCV_FUNCNAME("InitGlut");
	DEBUG_FCT("");
	//Init GLUT and OpenGL
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA| GLUT_ALPHA);
	glutInitWindowPosition(100,100);
	if(ImageSize.width==0)
		glutInitWindowSize(256,256);
	else
		glutInitWindowSize(ImageSize.width, ImageSize.height);
	glutCreateWindow("Morphology on CPU vs GPU");
	//callback settings
	glutDisplayFunc(CBDisplay);
	glutIdleFunc(CBIdle);	
	glutReshapeFunc(CBChangeSize);
	glutKeyboardFunc(CBkeys);
	//==========================
#if _DEMO_FORCE_NATIVE_OPENCV
	//using only opencv, we create a window to see the result image
	cvNamedWindow("MainWindow", 1);
#endif
}
//================================================================================
//open video input...[WEBCAM|VIDEO_FILE]
//if video file is specified, load it
//else check if a webcam is available, if not load default video sequence.
void OpenVideoInput(const std::string _videoFile, char _cCameraID/*=-1*/)
{
	GPUCV_FUNCNAME("OpenVideoInput");
	
	if(VideoCapture)
	{
		GPUCV_NOTICE("Closing video input...");
		cvReleaseCapture(&VideoCapture);
		bVideoFinished=false;
	}
	GPUCV_NOTICE("Opening video input...");
		CvCapture* capture = 0;
	    
	    if( _cCameraID>-1)//argc == 1 || (argc == 2 && strlen(argv[1]) == 1 && isdigit(argv[1][0])))
	        VideoCapture = cvCaptureFromCAM(_cCameraID);
	    else if( _videoFile!="" )
		{
			if(bFileIsVideo)
			{
			    VideoCapture = cvCaptureFromAVI(_videoFile.data()); 
				if(VideoCapture==NULL)
				{//using relative path name
					std::string TmpFileName = GetGpuCVSettings()->GetShaderPath() + "/" + VideoSeqFile;
					SGE::ReformatFilePath(TmpFileName);
					VideoCapture = cvCaptureFromAVI(TmpFileName.data()); 
				}
			}
			else//image file
			{
				imageCam = cvLoadImage(_videoFile.data());
				if(imageCam==NULL)
				{//using relative path name
					std::string TmpFileName = GetGpuCVSettings()->GetShaderPath() + "/" + VideoSeqFile;
					SGE::ReformatFilePath(TmpFileName);
					imageCam = cvLoadImage(TmpFileName.data());
					SG_AssertFile(imageCam, VideoSeqFile,"Could not initialize input file...\n");
				}
				return;				
			}
		}
		else
		{
			GPUCV_ERROR("No video input selected");
		}

	    SG_Assert(VideoCapture,"Could not initialize video input...Check syntaxe and use -f or -c parameters.\n");
}
//================================================================================
//grab video frame from current capture input source
IplImage * GrabInputFrame(IplImage * inputImage)
{
	static int frameId = 0;
	GPUCV_FUNCNAME("GrabInputFrame");
	DEBUG_FCT("");
	if( (bEnableGrabFrame==true) && (bFileIsVideo==true))
	{//we grab a new frame...

		imageCam = cvQueryFrame(VideoCapture);
		if(!imageCam)
		{
			GPUCV_ERROR("Could not open input frame");
			GPUCV_ERROR("Trying to re-openfile");
			OpenVideoInput(VideoSeqFile, iCameraID);
			SG_Assert(VideoCapture, "Could not open video input.");
		}
	}

	if(!imageCam)
	{	
		bVideoFinished = true;
		GPUCV_NOTICE("Video file finished");
	}
	else
	{//resize to required one
		#if _CAMDEBUG
				cvNamedWindow("Cam Image",1);
				cvShowImage("Cam Image",imageCam);
		//		cvWaitKey(0);
		#endif

		//if start size is 0, then use real video size
		if(ImageSize.width ==0)
		{
			ImageSize		= cvGetSize(imageCam);
			ImageSize_Next	= cvGetSize(imageCam);
			CBChangeSize(ImageSize.width, ImageSize.height);
		}
		//check if resize size has changed:
		//bool ImageResize=false;
		if(!inputImage)
			inputImage = cvCreateImage(ImageSize, imageCam->depth, imageCam->nChannels);

		
	/*	if(frameId>10)
		{
			bEnableGLWindowsResize = true;
			//CBChangeSize(ImageSize.width, ImageSize.height);
		}		
		else
			CBChangeSize(ImageSize.width, ImageSize.height);
	*/

		frameId++;

#if !_DEMO_FORCE_NATIVE_OPENCV
		cvgSetLabel(imageCam, "ImageCam");
		cvgSetLabel(inputImage, "ImageSrc");
#endif

		//resize image to fit defined camera size
		if(bEnableResize == true && 
			(ImageSize.width!=imageCam->width || ImageSize.height!=imageCam->height))
			cvResize(imageCam, inputImage);
		else if (inputImage->width!=imageCam->width || inputImage->height!=imageCam->height)
			cvResize(imageCam, inputImage);
		else
			cvCopy(imageCam, inputImage, NULL);

#if _CAMDEBUG
		cvNamedWindow("Input Image",1);
		cvShowImage("Input Image",inputImage);
		cvNamedWindow("Cam Image",1);
		cvShowImage("Cam Image",imageCam);
#endif
	}
	GPUCV_DEBUG("\n\n\n\n======================>>>frame ID:" <<frameId);
	return inputImage;
}
//================================================================================
/**
Init OpenCV structures and element for morphology processing.
*/
void InitProcessing()
{
	GPUCV_FUNCNAME("InitProcessing");
	DEBUG_FCT("");
	static bool InitDone = false;

	if (!InitDone)
	{
#if ! _DEMO_FORCE_NATIVE_OPENCV
		//loading GPUCV/OPENCV label (text)images files 
		//reformat file pathdft
		LabelOpenCVFile = GetGpuCVSettings()->GetShaderPath()+"/"+LabelOpenCVFile;
		SGE::ReformatFilePath(LabelOpenCVFile);
		LabelGpuCVFile = GetGpuCVSettings()->GetShaderPath()+"/"+LabelGpuCVFile;
		SGE::ReformatFilePath(LabelGpuCVFile);
		LabelGCUDAFile = GetGpuCVSettings()->GetShaderPath()+"/"+LabelGCUDAFile;
		SGE::ReformatFilePath(LabelGCUDAFile);
		

		GPUCV_DEBUG("AppPath:"<< GetGpuCVSettings()->GetShaderPath());
		GPUCV_DEBUG("Open Image "<< LabelOpenCVFile);
		//open files
		MSG_CV = cvLoadImage(LabelOpenCVFile.data(),1);
		SG_AssertFile(MSG_CV, LabelOpenCVFile, "Could not open");

		GPUCV_DEBUG("Open Image "<< LabelGpuCVFile);
		MSG_CVG = cvLoadImage(LabelGpuCVFile.data(),1);
		SG_AssertFile(MSG_CVG, LabelGpuCVFile, "Could not open");
		
		GPUCV_DEBUG("Open Image "<< LabelGCUDAFile);
		MSG_CVGCU = cvLoadImage(LabelGCUDAFile.data(),1);
		SG_AssertFile(MSG_CVGCU, LabelGCUDAFile, "Could not open");

		//Default label:
		CurrentImplLabel = MSG_CV;

		//load to GPU
#if 0//!_DEMO_FORCE_NATIVE_OPENCV
		cvgSetLocation<DataDsc_GLTex>(MSG_CV,  true);
		cvgSetLocation<DataDsc_GLTex>(MSG_CVG, true);
		cvgSetLocation<DataDsc_GLTex>(MSG_CVGCU, true);

		MSG_CV->origin = 1;	
		MSG_CVG->origin = 1;
		MSG_CVGCU->origin = 1;
#endif

#if _GPUCV_CAM_DEMO__USE_THREAD
		//init a thread to grab video frame:
		pthread_mutexattr_t attr;
		pthread_mutexattr_init (&attr);
		pthread_mutex_init(&MutexInputImages, &attr);
		SG_Assert(pthread_create(&ThreadPtr, NULL, (void*(*)(void *))ThreadLoopFunction, (void*)NULL)==0, "Too many threads");
#endif

#endif
		InitDone=true;
		//=======================================
	}
}
//================================================================================
void CBDisplay()
{
	GPUCV_FUNCNAME("CBDisplay");
	DEBUG_FCT("");

	if(bNeedResize)
	{
#if _GPUCV_CAM_DEMO__USE_THREAD
		//we need to discard all previous frames...
		//cause they do not have correct size
		pthread_mutex_lock (&MutexInputImages);
		std::vector<IplImage *>::iterator iterImage;
			for(iterImage = vFreeImages.begin();iterImage != vFreeImages.end(); iterImage++)
			{
				cvReleaseImage(& (*iterImage));
			}
			for(iterImage = vInputImages.begin();iterImage != vInputImages.end(); iterImage++)
			{
				cvReleaseImage(& (*iterImage));
			}
			vFreeImages.clear();
			vInputImages.clear();
			if(imageSrc)
			{
				cvReleaseImage(&imageSrc);
				imageSrc=NULL;
			}
			bNeedResize =false;
		pthread_mutex_unlock(&MutexInputImages);
#endif
		//======================================
	}
	if ((bEnableGrabFrame==true) || (imageSrc==NULL))
	{
#if _GPUCV_CAM_DEMO__USE_THREAD
		//get input image from vector list:
		pthread_mutex_lock (&MutexInputImages);
		if(vInputImages.size() >0)
		{	
			imageSrc =  *vInputImages.begin();
			vInputImages.erase(vInputImages.begin());//, vInputImages.begin());
		}
		pthread_mutex_unlock (&MutexInputImages);
#else
		imageSrc= GrabInputFrame(imageSrc);
#endif
	}
	
//	if(imageSrc)
	//we might process some input frame twice... so we discard previous loading.?
//		cvgSetDataFlag<DataDsc_CPU>(imageSrc, true, true);

	
	//beginning of image processing
	if(bPerformProcessing)
	{
		if(imageSrc)
		{
			GPUCV_DEBUG("Process frame");
			CamDemoFilters[FilterID].Process();
		}
	}

	if (bVideoFinished || imageDst ==NULL || imageSrc == NULL)
	{
		return;
	}

	
#if _CAMDEBUG//0//_DEBUG
	cvNamedWindow("Output Image",1);
	cvShowImage("Output Image",imageDst);
#endif

	IplImage * pImageToDraw = imageDst;//default
	if(!bPerformProcessing)
		pImageToDraw = imageSrc;
		

	//beginning of drawing
#if _DEMO_FORCE_NATIVE_OPENCV
	//we don't use opengl to show the results...no need
	//we use classic OpenCV window
	cvShowImage("MainWindow", imageDst);
	InitGLView(0,imageDst->width, 0, imageDst->height);
	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT);
	glFlush();
	glutSwapBuffers();
#else
	
	if(0)//(CVG_STATUS==GPUCV_IMPL_OPENCV))
	{
		cvNamedWindow("Output Image",1);
		cvShowImage("Output Image",pImageToDraw);
		//we don't use opengl to show the results...no need
		//we use classic OpenCV window
		InitGLView(0,pImageToDraw->width, 0, pImageToDraw->height);
		glClearColor(0,0,0,0);
		glClear(GL_COLOR_BUFFER_BIT);
	}
	else
	{
		cvgFlush(pImageToDraw);//force synchonization on destination image
		if(bGLoutput==true)
		{
			cvgInitGLView(pImageToDraw);
		}
		glClearColor(0,0,0,0);
		glClear(GL_COLOR_BUFFER_BIT);
		
		glScalef(1.,-1.,1.);//flip image
		if(bGLoutput==true)
		{
			//draw img Dst
			cvgDrawGLQuad(pImageToDraw);
			//=================
		}
		glScalef(1.,-1.,1.);//flip back
	}


	//draw implementation flags
	if(bPerformProcessing)
	{
		int lastImplID = cvgswGetLastCalledImpl(strLastFctCalled);
		switch(lastImplID)
		{
			case GPUCV_IMPL_GLSL:	CurrentImplLabel = MSG_CVG;break;
			case GPUCV_IMPL_CUDA:	CurrentImplLabel = MSG_CVGCU;break;
			case GPUCV_IMPL_OPENCV:	CurrentImplLabel = MSG_CV;break;
		}
		//Draw labels : "Processed with OpenCV/GpuCV/CUDA"
		if(CurrentImplLabel)
		{
			glPushMatrix();
			glRotatef(180, 1.,0.,0.);
			//calculate scale factor depending on image size and window size
			CvSize LogoSize = cvGetSize(CurrentImplLabel);
			float ScaleX = 1.;
			float ScaleY = 1.;
			ScaleX = 1/8.;//LogoSize.width / (float)(ImageSize.width);
			ScaleY = 1/8.;//LogoSize.height / (float)(ImageSize.height);
			
			//rescale
			if(ScaleX > 1./4.)
				ScaleX = (1./4.);
			else if (ScaleX < 1./10.)
				ScaleX = (1./10.);
			if(ScaleY > 1./4.)
				ScaleY = (1./4.);
			else if (ScaleY < 1./10.)
				ScaleY = (1./10.);
			
			glTranslatef(-1 + ScaleX, 1 - ScaleY,0);
			glScalef(ScaleX, ScaleY, 1);	
			cvgDrawGLQuad(CurrentImplLabel);
			glPopMatrix();
		}
		#if _GPUCV_CAM_DEMO__USE_THREAD
			if(vInputImages.size() <1)
			{	
				//if input image table is almost empty, we draw a red square in the corner
				InitGLView(0, 10, 0, 10);
				glClearColor(1.,0,0,0);
				glClear(GL_COLOR_BUFFER_BIT);
			}
		#endif
	}	
		
	glFlush();
	glutSwapBuffers();
#endif	
	if(imageSrc)
	{
		#if _GPUCV_CAM_DEMO__USE_THREAD
		//recycle image after processing
		if(imageSrc->width!=ImageSize.width || imageSrc->height!=ImageSize.height)
		{
			cvReleaseImage(&imageSrc);
			imageSrc = NULL;
		}
		if (bEnableGrabFrame==true)
		{//we release Input Image, otherwise we keep it.
			pthread_mutex_lock (&MutexInputImages);
			vFreeImages.push_back(imageSrc);
			imageSrc = NULL;
			pthread_mutex_unlock (&MutexInputImages);
		}
		else
		{
			//we might process some input frame twice... so we discard previous loading.?
			//cvgSetDataFlag<DataDsc_CPU>(imageSrc, true, true);
		}
		#endif
	}
	_GPUCV_GL_ERROR_TEST();
}
//================================================================================
void CBChangeSize(int w, int h)
{
	GPUCV_FUNCNAME("CBChangeSize");
	DEBUG_FCT("");
	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if(h == 0)
		h = 1;

	float ratio = 1.0* w / h;

	// Reset the coordinate system before modifying
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(45,ratio,1,1000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0,0.0,5.0, 
		0.0,0.0,-1.0,
		0.0f,1.0f,0.0f);
	if(bEnableResize)
	{
		ImageSize_Next.width=w;
		ImageSize_Next.height=h;
		if(ImageSize.width!=ImageSize_Next.width || ImageSize.height!=ImageSize_Next.height)
		{
			bNeedResize =true;
			ImageSize.width=ImageSize_Next.width;
			ImageSize.height=ImageSize_Next.height;
			
	//		GPUCV_NOTICE("Resizing source image to: " << ImageSize.width <<" x " << ImageSize.height);
		}
	}
}
//================================================================================

//================================================================================
void CBIdle()
{
	GPUCV_FUNCNAME("CBIdle");
	DEBUG_FCT("");

	//show results
	glutPostRedisplay();
}
//==============================================================
#if _GPUCV_CAM_DEMO__USE_THREAD
void *ThreadLoopFunction(void * _ptr)
{
	IplImage * pNewImage = NULL;
	while (1)//!currAppliTracer->GetExitFlag())
	{
		if((vInputImages.size()<INPUT_IMAGE_STACK_SIZE) && (bNeedResize==false))
		{
			if (bVideoFinished)
			{//try to reload video file
				OpenVideoInput(VideoSeqFile, iCameraID);//open video input from file or webcam
			}
			pNewImage = NULL;
			//try to recycle images
			if(vFreeImages.size()>0)
			{
				pthread_mutex_lock (&MutexInputImages);
				pNewImage = *vFreeImages.begin();
				vFreeImages.erase(vFreeImages.begin());//, vFreeImages.begin());
				pthread_mutex_unlock(&MutexInputImages);
			}
			pNewImage = GrabInputFrame(pNewImage);
			if(pNewImage)
			{
				pthread_mutex_lock (&MutexInputImages);
				vInputImages.push_back(pNewImage);
				pthread_mutex_unlock(&MutexInputImages);
			}
		}
#ifdef _WINDOWS
		else
			Sleep(1);
#endif
	//	pthread_yield();
	}	
#if SGPT_USE_PTHREAD
 	pthread_exit(NULL);
#endif
	return NULL;
}
#endif