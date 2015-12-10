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

using namespace GCV;

extern	CvSize ImageSize;
extern	std::string LabelOpenCVFile;
extern  std::string LabelGpuCVFile;
extern std::string AppPath;
extern IplImage *MSG_CV;
extern IplImage *MSG_CVG;
extern int iterac;
extern enum BaseImplementation CVG_STATUS;

bool SobelInitDone = false;
IplImage *imageSrc_1C, *imageSrc_1C32F;

bool SobelInit()
{
	if (!SobelInitDone && imageSrc)
	{
		CvSize LutSize;
		LutSize.height = 1;
		int scale = 1;
		LutSize.width = 256;

		imageSrc_1C = cvCreateImage(ImageSize, imageSrc->depth, 1);
		imageSrc_1C32F  = cvCreateImage(ImageSize, IPL_DEPTH_32F, 1);
		imageDst  = cvCreateImage(ImageSize, IPL_DEPTH_32F, 1);
		
#if !_DEMO_FORCE_NATIVE_OPENCV
		cvgSetLabel(imageSrc_1C,"Sobel - 1C");
		cvgSetLabel(imageSrc_1C32F,"Sobel - 1C32F");
		cvgSetLabel(imageDst, "Sobel - Image Destination");
#endif
		SobelInitDone=true;
		return true;
		//=======================================
	}
	return false;
}

bool SobelClean()
{
	if(SobelInitDone)
	{
		SobelInitDone = false;
		cvReleaseImage(&imageSrc_1C);
		cvReleaseImage(&imageSrc_1C32F);
#if !_DEMO_FORCE_NATIVE_OPENCV
		cvgswPrintFctStats("cvSobel");
#endif
		return true;
	}
	return false;
}

bool SobelSwitch(BaseImplementation _type)
{
	if(_type==GPUCV_IMPL_CUDA)
	{
		CVG_STATUS=GPUCV_IMPL_OPENCV;
		std::cout << std::endl << "Using OpenCV" << std::endl;
	}
	return true;
}

bool SobelProcess()
{
	if (!SobelInitDone)
		SobelInit();
	if(imageSrc->width!=imageDst->width || imageSrc->height!=imageDst->height)
	{//need to refresh image size on the algorithm...
		SobelClean();
		SobelInit();
	}

	if(imageSrc_1C32F && imageDst)
	{
		cvCvtColor(imageSrc, imageSrc_1C, CV_RGB2GRAY);
		cvConvertScale(imageSrc_1C, imageSrc_1C32F, 1./256);

		cvSobel(imageSrc_1C32F, imageDst, 0, 1, 3);
		return true;
	}
	else
		return false;
}
