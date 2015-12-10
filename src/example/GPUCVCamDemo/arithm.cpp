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
bool ArithmInitDone = false;
IplImage * imageArithmGrey=NULL;
IplImage * imageArithmGreyMask=NULL;

bool ArithmInit()
{
	if (!ArithmInitDone &&  imageSrc)
	{
		imageDst  = cvCreateImage(ImageSize, imageSrc->depth, imageSrc->nChannels);
		imageArithmGrey = cvCreateImage(ImageSize, imageSrc->depth, 1);
		imageArithmGreyMask = cvCreateImage(ImageSize, imageSrc->depth, 1);
		

#if !_DEMO_FORCE_NATIVE_OPENCV
		cvgSetLabel(imageDst, "Arithm - Image Destination");
		cvgSetLabel(imageArithmGrey, "imageArithmGrey - Image Destination");
		cvgSetLabel(imageArithmGrey, "imageArithmGreyMask - Image Destination");
#endif

		ArithmInitDone=true;
		return true;
		//=======================================
	}
	return false;
}

bool ArithmClean()
{
	if(ArithmInitDone)
	{
		ArithmInitDone = false;
		cvReleaseImage(&imageArithmGrey);
		cvReleaseImage(&imageArithmGreyMask);
#if !_DEMO_FORCE_NATIVE_OPENCV
		cvgswPrintFctStats("cvArithm");
#endif
		return true;
	}
	return false;
}

bool ArithmSwitch(GCV::BaseImplementation _type)
{
	return true;
}

bool ArithmProcess()
{
	if (!ArithmInitDone)
		ArithmInit();
	if(imageSrc->width!=imageDst->width || imageSrc->height!=imageDst->height)
	{//need to refresh image size on the algorithm...
		ArithmClean();
		ArithmInit();
	}

	if(imageSrc && imageDst)
	{
#if 1
		cvCvtColor(imageSrc, imageArithmGrey, CV_BGR2GRAY);
		cvThreshold(imageArithmGrey, imageArithmGreyMask, 100, 150, 0);
		cvMerge(imageArithmGreyMask, imageArithmGreyMask, imageArithmGreyMask, NULL, imageDst);
#else
		cvCvtColor(imageSrc, imageArithmGrey, CV_BGR2GRAY);
		cvThreshold(imageArithmGrey, imageArithmGreyMask, 100, 150, 0);
		cvAddWeighted(imageArithmGrey, 0.5, imageSrc, 0.5, 0., imageDst);
#endif			
	//cvThreshold(imageSrc, imageDst, 100, 150, 0);
		//cvDiv(imageSrc, imageDst, NULL, 2);
		strLastFctCalled = "cvArithm";
		return true;
	}
	else
		return false;
}
