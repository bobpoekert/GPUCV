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
#ifndef _MAINSAMPLETEST_H_
#define _MAINSAMPLETEST_H_
 

//#define _GPUCV_SUPPORT_CUDA//must be defined before including other files

#define _GCV_CUDA_EXTERNAL 1
#include "StdAfx.h"

#include "macro.h"

//include opencv
#include <cxcore.h>
#include <cv.h>
#include <highgui.h>

//include cvg
#include <highguig/highguig.h>
#include <cvg/cvg.h>
#include <cxcoreg/cxcoreg.h>

//include all cvgsw*()
#ifdef _GPUCV_SUPPORT_SWITCH
#	include <cv_switch/cv_switch.h>
#	include <cxcore_switch/cxcore_switch.h>
#	include <highgui_switch/highgui_switch.h>
#endif
//include GpuCV objects
#include <GPUCVHardware/GlobalSettings.h>
#include <GPUCVHardware/config.h>
#include <GPUCV/include.h>
#include <GPUCV/cxtypesg.h>

/*#include "cvg_test.h"
#include "cxcoreg_test.h"
#include "misc_test.h"
*/
#if defined _GPUCV_SUPPORT_CUDA && !_GCV_CUDA_EXTERNAL
	#include <GPUCVCuda/include.h>
	#include <GPUCVCuda/GPU_NVIDIA_CUDA.h>
	#include <cublas.h>
	#include <cutil_inline.h>
	#include <cuda_gl_interop.h>
#endif



//! All input image will be converted to GRAY to be compatible with CUDA operators(it doesn't support 3 channels image yet).
#if defined _GPUCV_SUPPORT_CUDA && !_GCV_CUDA_EXTERNAL
#define GPUCVCONSOLE_FORCE_BW_SOURCE 0
#else
#define GPUCVCONSOLE_FORCE_BW_SOURCE 0
#endif


#if NDEBUG
#undef	GPUCV_FUNCNAME
/** In GPUCV libraries, we disable GPUCV_FUNCNAME in release mode for faster processing
but it is used by the console application to perform profiling. So we redefine it locally.*/
#define GPUCV_FUNCNAME(FCT_NAME)std::string FctName=FCT_NAME;
#endif

bool cxcoreg_processCommand(std::string & CurCmd, std::string & nextCmd);
#if TEST_GPUCV_CV
bool cvg_processCommand(std::string & CurCmd, std::string & nextCmd);
#endif


//test functions
void testBenchLoop(IplImage * src1);
void runVBO(IplImage * src1);
void runStats(IplImage * src1);

void runIsDiff(IplImage * src1,IplImage * src2);
void runTestGeometry(IplImage * src1,IplImage * src2);


//bool resizeImage(IplImage ** Img, int width, int height, int interpolation=CV_INTER_LINEAR);

//==============
//tools functions
template<typename TType>
void PrintMatrix(std::string _name, int width, int height, TType * data)
{
	int p,q;
	TType * tempData = data;
	std::cout << "Matrix Name : "<< _name;
	for (p=0;p<height;p++)
	{
		std::cout << std::endl << "___________________________________________" << std::endl;
		std::cout << "|" << std::endl;
		for (q=0;q<width;q++)
		{
			printf("%.1f ",*tempData++);
			std::cout << "\t|";
		}
	}
	std::cout << std::endl << "___________________________________________" << std::endl;
}
void PrintImageInfo(std::string _imgID);

#if _GPUCV_DEPRECATED
void showbenchResult(std::string  Cmd);
#endif
void PrintMsg();


bool loadImage(IplImage * Img, char * filename="");
void LoadDefaultImages(std::string _AppPath);
void testImageFormat(IplImage * src1, int channels, int depth, float scale);
bool changeImageFormat2(IplImage ** Img, int channels, int depth, float scale);
bool changeImageFormat(std::string & Command);
void SynchronizeOper(enum OperType _operType, CvArr * _pImage);
//===============================


/** @defgroup GPUCVCONSOLE_GRP GpuCV console test application
	@ingroup GPUCV_EXAMPLE_LIST_GRP
	@{
		@defgroup GPUCVCONSOLE_GRP_PARAMETERS Parameters
		@defgroup GPUCVCONSOLE_GRP_COMMANDS Commands available from the console
		@defgroup GPUCVCONSOLE_GRP_OPERATORS Operators
		@defgroup GPUCVCONSOLE_VARIABLES_GRP Global variables
	@}
*/

/**
*   @addtogroup GPUCVCONSOLE_VARIABLES_GRP
@{*/

/** \brief Global image used as source 1.
*/
extern IplImage *GlobSrc1;
/** \brief Global image used as source 2.
*/
extern IplImage *GlobSrc2;
/** \brief Global image used as optional mask.
*/
extern IplImage *GlobMask;
/** \brief Global image used to backup optional mask.
*/
extern IplImage *MaskBackup;
/** \brief Global flag to specify that we use scalar values, affects operators such as cvAddS.
*	\sa use_mask, global_scalar.
*/
extern bool use_scalar;
/** \brief Global flag to specify that we use the optional mask, affects operators such as cvAddS.
*	\sa use_scalar, global_scalar.
*/
extern bool use_mask;

/** \brief Global value that is used when use_scalar flag is true.
*	\sa use_mask, global_scalar.
*/
extern CvScalar * global_scalar;

/** \brief Number of iteration used for benchmarking.
*	\sa Macros _CV_benchLoop / _GPU_benchLoop / _CUDA_benchLoop.
*/
extern int NB_ITER_BENCH;
extern bool benchmark;
extern bool AvoidCPUReturn;
extern bool DataPreloading;
extern bool CmdOperator;
extern float fEpsilonOperTest;
extern std::string CurLibraryName; //!< Used to set the class of operators when benchmarking using cv_all/cxcore_all/...

/** \brief Flag to set if we want to show result image or not.
*	\sa __ShowImages__, __CreateWindows__.
*/
extern bool ShowImage;

/** \brief Flag to set that we use IPP instead of OpenCV.
*	\sa __ShowImages__, __CreateWindows__.
*/
extern bool SelectIpp;

/** \brief Global mask that specify which implementation to use between OpenCV and GpuCV(OperType).
*	\sa _CV_benchLoop/_GPU_benchLoop/_CUDA_benchLoop, __CreateImages__, __CreateMatrixes__, __CreateWindows__.
*/
extern int GpuCVSelectionMask;


/** \brief Flag to set if we want to compare OpenCV and GpuCV results.
*	\sa __ShowImages__, __CreateWindows__.
*/
extern bool ControlOperators;

extern bool cxcore_test_enable;
extern bool cv_test_enable;
extern bool transfer_test_enable;
/** @}*/

#ifndef _WINDOWS
typedef long int LONGLONG;
#endif

#endif
