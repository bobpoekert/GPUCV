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

bool LutInitDone = false;
IplImage * LUT_table;

bool LutInit()
{
	if (!LutInitDone &&  imageSrc)
	{
		CvSize LutSize;
		LutSize.height = 1;
		int scale = 1;
		LutSize.width = 256;
		LUT_table = cvCreateImage(LutSize,IPL_DEPTH_8U,imageSrc->nChannels);
		imageDst  = cvCreateImage(ImageSize, imageSrc->depth, imageSrc->nChannels);
		

		if (LUT_table->depth == IPL_DEPTH_8U)
			scale = 1 ; 
		else
			scale = 256;

		char * Data = LUT_table->imageData;
		for(int i =0; i < LutSize.width; i++)
		{
			for(int j =0; j <LUT_table->nChannels ; j++)
			{
				*Data = (255*scale)-i;
				Data++;
			}
		}

#if !_DEMO_FORCE_NATIVE_OPENCV
		cvgSetLabel(LUT_table,"LUT - table");
		cvgSetLabel(imageDst, "LUT - Image Destination");
		cvgSetOptions(LUT_table, DataContainer::UBIQUITY, true);
#endif
//		LUT_table->origin	= 0;

#if !_DEMO_FORCE_NATIVE_OPENCV
		cvgUnsetCpuReturn(LUT_table);
#endif	
		LutInitDone=true;
		return true;
		//=======================================
	}
	return false;
}

bool LutClean()
{
	if(LutInitDone)
	{
		LutInitDone = false;
		cvReleaseImage(&LUT_table);
#if !_DEMO_FORCE_NATIVE_OPENCV
		cvgswPrintFctStats("cvLut");
#endif
		return true;
	}
	return false;
}

bool LutSwitch(BaseImplementation _type)
{
	return true;
}

bool LutProcess()
{
	if (!LutInitDone)
		LutInit();
	if(imageSrc->width!=imageDst->width || imageSrc->height!=imageDst->height)
	{//need to refresh image size on the algorithm...
		LutClean();
		LutInit();
	}

	if(imageSrc && imageDst)
	{
		cvLUT(imageSrc, imageDst, LUT_table);
		//cvCvtColor(imageSrc, imageDst, CV_BGR2YCrCb);
		//cvThreshold(imageSrc, imageDst, 100, 150, 0);
		//cvDiv(imageSrc, imageDst, NULL, 2);
		strLastFctCalled = "cvLUT";
		return true;
	}
	else
		return false;
}
