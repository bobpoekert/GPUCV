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
extern int iterac;
extern enum BaseImplementation CVG_STATUS;

IplConvKernel* elemStrut3 = NULL;		//!< Structuring element of size 3*3
IplConvKernel* elemStrut5 = NULL;		//!< Structuring element of size 5*5
const CvElementShape 	shp = CV_SHAPE_RECT; //!< Structuring element shape 1
const CvElementShape 	shp2 = CV_SHAPE_RECT;//!< Structuring element shape 2
bool MorphoInitDone = false;
IplImage * imageTemp2=NULL;
//=======================================

bool MorphoInit()
{
	if (!MorphoInitDone && imageSrc)
	{
		//create structuring elements of size 3 and 5
		elemStrut3 = cvCreateStructuringElementEx(3,3, 1, 1,shp, NULL);
		elemStrut5 = cvCreateStructuringElementEx(5,5, 1, 1,shp, NULL);

		//create temporary images to be processed
		imageDst  =		cvCreateImage(ImageSize, imageSrc->depth, imageSrc->nChannels);
		imageTemp =		cvCreateImage(ImageSize, imageSrc->depth, imageSrc->nChannels);
		imageTemp2 =	cvCreateImage(ImageSize, imageSrc->depth, imageSrc->nChannels);

#if !_DEMO_FORCE_NATIVE_OPENCV
		cvgSetLabel(imageDst, "Morpho - Image Destination");
		cvgSetLabel(imageTemp, "Image Temp");
		cvgSetLabel(imageTemp2, "Image Temp2");
#endif
		imageDst->origin	= 0;
		imageTemp->origin	= 0;
		imageSrc->origin	= 0;

		MorphoInitDone=true;
		return true;
	}
	return false;
}
//=======================================

bool MorphoClean()
{
	if(MorphoInitDone)
	{
		cvReleaseStructuringElement(&elemStrut3);
		cvReleaseStructuringElement(&elemStrut5);

		MorphoInitDone = false;
		cvReleaseImage(&imageDst);
		cvReleaseImage(&imageTemp);
		cvReleaseImage(&imageTemp2);
#if !_DEMO_FORCE_NATIVE_OPENCV
		cvgswPrintFctStats("cvMorphologyEx");
#endif
		return true;
	}
	return false;
}
//=======================================
bool MorphoSwitch(BaseImplementation _type)
{
	return true;
}
//=======================================
bool MorphoProcess()
{
	if (!MorphoInitDone)
		MorphoInit();
	if(imageSrc->width!=imageDst->width || imageSrc->height!=imageDst->height)
	{//need to refresh image size on the algorithm...
		MorphoClean();
		MorphoInit();
	}

	if(imageSrc && imageDst && imageTemp)
	{
		//cvErode(imageSrc, imageDst, elemStrut3); 
		//size 3
		cvMorphologyEx( imageSrc, imageTemp2, imageTemp,elemStrut3,CV_MOP_OPEN,iterac);
		cvMorphologyEx( imageTemp2, imageDst, imageTemp,elemStrut3,CV_MOP_CLOSE,iterac);
		//size 5
		cvMorphologyEx( imageDst, imageTemp2, imageTemp,elemStrut5,CV_MOP_OPEN,iterac);
		cvMorphologyEx( imageTemp2, imageDst, imageTemp,elemStrut5,CV_MOP_CLOSE,iterac);
		
		strLastFctCalled ="cvMorphologyEx";//used to get the last implementation called
		return true;
	}
	else
		return false;
}
//=======================================
