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
/**
Main Sample application to test all the function developed with GpuCV
*/
#include "StdAfx.h"
#include "mainSampleTest.h"
#include "commands.h"


#ifdef _WINDOWS
//#	define _WIN32_WINNT 0x0800
#	include <Windows.h>
#	include <Winuser.h>
#endif

using namespace GCV;

IplImage *GlobSrc1,*GlobSrc2, *GlobMask, *MaskBackup;

int NB_ITER_BENCH;
bool benchmark		= false;
bool AvoidCPUReturn	= true;
bool CmdOperator	= true;
bool DataPreloading = true;
bool ControlOperators = true;
bool ShowImage	=true;
bool AskForExit =false;
bool SelectIpp	=false;
int GpuCVSelectionMask = OperALL;// - OperGLSL;// - OperCuda ;// |
void callback();
void Idle();
extern std::string ConsoleCommand;
float fEpsilonOperTest = 0.5;//1e-2f;//1e-4f
std::string CurLibraryName;

//=============================================================================
void runIsDiff(IplImage * src1,IplImage * src2)
{
	GPUCV_FUNCNAME("IsDiff");
	__CreateImages__(cvGetSize(src1) ,IPL_DEPTH_32F, 1, OperGLSL);

	bool isdiff=false;
	std::string localParams="";
	_GPU_benchLoop(isdiff=cvgIsDiff(src1,src2),NULL , localParams);

	if (isdiff)
	{ GPUCV_ERROR("\ncvgIsDiff : image are not equal");}
	else
	{   GPUCV_NOTICE("\ncvgIsDiff : image are equal");}
	__ReleaseImages__();
}
//=============================================================================
void LoadDefaultImages(std::string _AppPath)
{
	std::string NameImg1 = _AppPath + "/pictures/lena512.jpg";
	//std::string NameImg1 = "data/pictures/test_avg.jpg";
	std::string NameImg2 = _AppPath +"/pictures/plage512.jpg";

	SGE::ReformatFilePath(NameImg1);
	SGE::ReformatFilePath(NameImg2);


	if(GlobSrc1)
		cvgReleaseImage(&GlobSrc1);
	if(GlobSrc2)
		cvgReleaseImage(&GlobSrc2);
	if(MaskBackup)
		cvgReleaseImage(&MaskBackup);


	GPUCV_DEBUG("Start Loading image 1:" << NameImg1);

	if( (GlobSrc1 = cvLoadImage(NameImg1.data(),1)) == 0 )
		SG_Assert(GlobSrc1, "Can't load image source 1");


	GPUCV_DEBUG("Start Loading image 2:" << NameImg2);
	if( (GlobSrc2 = cvLoadImage(NameImg2.data(),1)) == 0 )
		SG_Assert(GlobSrc1, "Can't load image source 2");

	//create global mask
	MaskBackup = cvCreateImage(cvGetSize(GlobSrc2), IPL_DEPTH_8U, 1);
	cvCvtColor(GlobSrc2, MaskBackup, CV_BGR2GRAY);
	cvThreshold(MaskBackup, MaskBackup, 150, 250, 1);

#if GPUCVCONSOLE_FORCE_BW_SOURCE
	//convert into b/w
	IplImage * tempImg = NULL;
	tempImg = GlobSrc1;
	GlobSrc1 = cvgCreateImage(cvGetSize(tempImg), tempImg->depth, 1);
	cvCvtColor(tempImg, GlobSrc1, CV_BGR2GRAY);

	tempImg = GlobSrc2;
	GlobSrc2 = cvgCreateImage(cvGetSize(tempImg), tempImg->depth, 1);
	cvCvtColor(tempImg, GlobSrc2, CV_BGR2GRAY);
	cvgRelease(&tempImg);
#endif
}
//=============================================================================

void ReleaseAll()
{//Memleak 11/03/10
	cvgReleaseImage(&GlobSrc1);
	cvgReleaseImage(&GlobSrc2);
	cvgReleaseImage(&MaskBackup);
	cvDestroyAllWindows();
	/*
	GPUCV_DEBUG("GPU_NVIDIA_CUDA::~GPU_NVIDIA_CUDA(): cudaThreadExit()");
	cudaError err = cudaThreadSynchronize();                                 \
	if( cudaSuccess != err)
	{                                                \
		fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
		__FILE__, __LINE__, cudaGetErrorString( err) );              \
		fprintf(stderr, "Press a key to exit application.\n");					\
		getchar();																\
      }                                       \
	cudaThreadExit();
	*/
}

/**
\todo add optional parameter to select IMG1, IMG2, IMGMask, and launch benchmarks automatoically from command prompt.
*/
int main(int argc, char **argv)
{
	try
	{
		if (argc > 1)
		{
			ParseCommands(argc, argv);//fill the command queue
			GPUCV_NOTICE("Press a key to continue...");
		}

		//set GPUCV main settings.... before cvgInit
		SET_GPUCV_OPTION(
			GpuCVSettings::GPUCV_SETTINGS_PROFILING,
			true);
		//set to check opengl error and rise exceptions
		SET_GPUCV_OPTION(
			GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_RISE_EXCEPTION
			| GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_CHECK
			| GpuCVSettings::GPUCV_SETTINGS_PLUG_THROW_EXCEPTIONS,
			true);
		//set to Use Call opencv function inside GpuCV operators
		SET_GPUCV_OPTION(
			GpuCVSettings::GPUCV_SETTINGS_USE_OPENCV,
			false);
		//set to checking shader changes
		SET_GPUCV_OPTION(
			GpuCVSettings::GPUCV_SETTINGS_CHECK_SHADER_UPDATE,
			true);
		//set to disable error resolution inside the switch operators
		SET_GPUCV_OPTION(
			GpuCVSettings::GPUCV_SETTINGS_PLUG_AUTO_RESOLVE_ERROR,
			false);
		//=======================================
		
		//get application path
		std::string strDataPath = cvgRetrieveShaderPath(argv[0]);

		//choose to use IPP or not.
		cvUseOptimized(SelectIpp);

		// GPU lib initialisation
		//GetGpuCVSettings()->EnableGlutDebug();
#ifdef _GPUCV_SUPPORT_SWITCH
		cvg_switch_RegisterTracerSingletons(&SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance(), &SG_TRC::CL_TRACING_EVENT_LIST::Instance());	
		cvgswInit(true);
	#if defined _GPUCV_SUPPORT_CUDA && !_GCV_CUDA_EXTERNAL
			cvgCudaSetProcessingDevice();
			cvgcuInit(true);
	#endif
		cvg_cv_switch_RegisterTracerSingletons(&SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance(), &SG_TRC::CL_TRACING_EVENT_LIST::Instance());
		cvg_cxcore_switch_RegisterTracerSingletons(&SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance(), &SG_TRC::CL_TRACING_EVENT_LIST::Instance());
		cvg_highgui_switch_RegisterTracerSingletons(&SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance(), &SG_TRC::CL_TRACING_EVENT_LIST::Instance());

		//disable previous benchmarks such as CUDA/OPENCV/GLSL, every thing is done threw the switch
		EnableDisableSettings("opencv",false);
		EnableDisableSettings("glsl",false);
		EnableDisableSettings("cuda",false);
		EnableDisableSettings("ipp",false);
#else
	#ifdef _GPUCV_SUPPORT_CUDA
			cvgCudaSetProcessingDevice();
			cvgcuInit(true);
	#else
			cvgInit(true);
	#endif
		//disable the switch macros 
		EnableDisableSettings("switch",false);
#endif
		
		GPUCV_NOTICE("\nInitDone");
		GPUCV_NOTICE("\nloading default Images...");

	

		LoadDefaultImages(strDataPath);

		if(GCV::ProcessingGPU()->GetGLSLProfile()==GCV::GenericGPU::HRD_PRF_0)
		{
			GPUCV_WARNING("Your GPU is not compatible with the minimum profile required to run GLSL shaders or CUDA operators. GLSL and CUDA plugins will be disabled!");
			EnableDisableSettings("cuda", false);
			EnableDisableSettings("glsl", false);
		}
		else
		{
			EnableDisableSettings("cuda", true);
			EnableDisableSettings("glsl", true);
		}



		cvgSetLabel(GlobSrc1,"GlobSrc1");
		cvgSetLabel(GlobSrc2,"GlobSrc2");
		cvgSetLabel(MaskBackup,"GlobMask");

		GPUCV_NOTICE("Init done!");

	/*	if (argc > 1)
		{
			ParseCommands(argc, argv);//fill the command queue
			GPUCV_NOTICE("Press a key to continue...");
		}
	*/	//init app
		NB_ITER_BENCH = 1;
		ShowImage = true;
		AvoidCPUReturn = true;

		GPUCV_NOTICE("GpuCVConsole Application: enter 'h' to get the list of available commands.");

		std::cout << "\n>";
		//==================

		//we perform normal console
#if _GPUCV_GL_USE_GLUT
		if(GetGpuCVSettings()->GetGlutDebug())
		{
			glutIdleFunc(Idle);
			glutDisplayFunc(ViewerWindowDisplay);
			glutReshapeFunc(ViewerWindowReshape);
			ViewerWindowReshape(512,512);
			glutMainLoop();
		}
		else
#endif
		{
			while(!AskForExit)
				Idle();
		}
		//=========================
		ReleaseAll();
	}
	catch(SGE::CAssertException &e)
	{
		GPUCV_NOTICE("=================== Exception catched Start =================");
		GPUCV_NOTICE(e.what());
		GPUCV_NOTICE("=================== Exception catched End =================");
		GPUCV_NOTICE("Press a key to continue...");
		getchar();
	}
	cvgTerminate();
}
//=============================================================================

void Idle()
{
	std::string Command;
	CmdOperator = true;

	if(ConsoleCommand!="")
	{
		Command = ConsoleCommand;//we still have command to process
	}
	else
		std::cin>> Command;//get command from the queue

#if _GPUCV_GL_USE_GLUT
	if(GetGpuCVSettings()->GetGlutDebug())
		glutSwapBuffers();
#endif

	if(Command!="")
	{
		Command = SGE::StrTrimLR(Command);//remove first and last space of the line
		int result = ProcessCommand(Command);
		//Command might contains other operators to process
		ConsoleCommand = Command;

		if(result == -1)
		{
			//	ViewerWindowIdle();
			AskForExit = true;
			return;
		}
		std::cout << "\n>";
	}
	else
	{
		//Sleep(10);
		//ViewerWindowIdle();
		return;
	}
	//====================

	std::cout << "\n";
	//ViewerWindowIdle();
}

