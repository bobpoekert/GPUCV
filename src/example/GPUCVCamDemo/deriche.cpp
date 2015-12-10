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
extern BaseImplementation CVG_STATUS;
IplImage * DericheGreyInput8u=NULL;
IplImage * DericheGreyInput32f=NULL;


bool DericheInitDone = false;
//=======================================
bool DericheInit()
{
	if (!DericheInitDone)
	{
		//grab first frame for init
		if(imageDst)
		{
			cvReleaseImage(&imageDst);
		}
		imageDst  = cvCreateImage(ImageSize, IPL_DEPTH_32F, 1);
		if(!DericheGreyInput8u)
		{
			DericheGreyInput8u = cvCreateImage(ImageSize, imageSrc->depth, 1);
			DericheGreyInput32f = cvCreateImage(ImageSize, IPL_DEPTH_32F, 1);

#if 0//!_DEMO_FORCE_NATIVE_OPENCV
			cvgUnsetCpuReturn(DericheGreyInput8u);
			//cvgUnsetCpuReturn(DericheGreyInput32f);
			cvgUnsetCpuReturn(imageSrc);
			cvgUnsetCpuReturn(imageDst);
#endif
		}

		DericheInitDone=true;
		return true;
		//=======================================
	}
	return false;
}
//=======================================
bool DericheClean()
{
	if(DericheInitDone)
	{
		if(imageDst)
		{
			cvReleaseImage(&imageDst);
		}
		if(!DericheGreyInput8u)
		{
			cvReleaseImage(&DericheGreyInput8u);
			cvReleaseImage(&DericheGreyInput32f);
		}

		DericheInitDone = false;
#if !_DEMO_FORCE_NATIVE_OPENCV
		cvgswPrintFctStats("cvDeriche");
#endif


		return true;
	}
	return false;
}
//=======================================
bool DericheSwitch(BaseImplementation _type)
{
	/*
#if !_DEMO_FORCE_NATIVE_OPENCV
	if (_type==GPUCV_IMPL_OPENCV)
	{
		cvgSynchronize(imageSrc);
		cvgSynchronize(imageCam);
		cvgSynchronize(DericheGreyInput8u);
		cvgSynchronize(DericheGreyInput32f);
		cvgSynchronize(imageDst);
		return true;
	}
#endif

	return false;
	*/
	return true;

}
//=======================================
bool DericheProcess()
{
	if(imageSrc &&  imageDst)
	{
		cvCvtColor(imageSrc, DericheGreyInput8u, CV_BGR2GRAY);
		cvConvertScale(DericheGreyInput8u, DericheGreyInput32f,1./256.);
#if 0//_DEBUG
		cvNamedWindow("DericheGreyInput8u",1);
		cvShowImage("DericheGreyInput8u",DericheGreyInput8u);
		cvNamedWindow("DericheGreyInput32f",1);
		cvShowImage("DericheGreyInput32f",DericheGreyInput32f);
#endif
		cvDeriche(DericheGreyInput32f, imageDst, 1.45);	
/*
#if 1	
		if(CVG_STATUS==GPUCV_IMPL_OPENCV)
		{
			
		}
		else if(CVG_STATUS==GPUCV_IMPL_CUDA)
		{
//			cvgCudaDeriche(DericheGreyInput32f, imageDst, 1.45);
		}

		//cvScale(imageDst, imageDst, 1024);
#endif
*/
#if 1
#if !_DEMO_FORCE_NATIVE_OPENCV
		if(CVG_STATUS==GPUCV_IMPL_OPENCV)
		{
			cvgSetDataFlag<DataDsc_IplImage>(imageDst, true,true);
			//cvgSetDataFlag<DataDsc_IplImage>(imageSrc, true,true);
		}
		else if(CVG_STATUS==GPUCV_IMPL_CUDA)
		{
//			cvgSetDataFlag<DataDsc_CUDA_Buffer>(imageDst, true,true);
			//cvgSetDataFlag<DataDsc_IplImage>(imageSrc, true,true);
		}
#if 1//_DEBUG
	cvNamedWindow("TestDst",1);
	cvShowImage("TestDst",imageDst);
#endif
#endif		
#endif
		return true;
	}
	else
		return false;
}
