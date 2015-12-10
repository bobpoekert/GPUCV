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



/** \brief C++ File containg definitions GPU equivalents for openCV functions: Image Processing -> Integral (Summed Area Tables)
\author Ankit Agarwal
*/
#include "StdAfx.h"
#include <cvg/cvg.h>
#include <GPUCV/toolscvg.h>
#include <cxcoreg/cxcoreg.h>
#include <highguig/highguig.h>
#include <GPUCV/misc.h>
#include <math.h>


using namespace GCV;


int powera(int a,int b)
{	int r=1;
for (int i=0;i<b;i++)
{r=r*a;}
return r;
}
int loga(int a)
{	double d = (double) a;
int r = (int) ceil(log10(d)/log10(2.0));
return r;
}

#if _GPUCV_DEVELOP_BETA
void cvgIntegral(CvArr* src1,CvArr* dst)
{

	GPUCV_START_OP(cvIntegral(src1,dst,NULL,NULL),
		"cvgIntegral",
		src1,
		GenericGPU::HRD_PRF_3);

	int width	=GetWidth(src1);
	int height	=GetHeight(src1);
	int wpass = loga(width);
	int hpass = loga(height);
	float param[4];
	param[0]=1.;
	param[1]=width;
	param[2]=height;


	IplImage* TmpImage=cvgCreateImage(cvGetSize(src1), GetCVDepth(src1), GetnChannels(src1));
	cvgSetLabel(TmpImage, "tempImage");
	IplImage* TmpDest=NULL;
	IplImage* TmpSrc=src1;
	IplImage* TmpSwitch=NULL;
	TmpDest = dst;

	//first call that use src1
	param[3]=(float) powera(2,0);
	TemplateOperator("cvgIntegral", "FShaders/satf", "",
		src1, NULL, NULL,
		TmpDest,param,4,
		TextureGrp::TEXTGRP_SAME_ALL_FORMAT,"");

	if(TmpDest == dst)
		TmpSrc = TmpImage;
	else
		TmpSrc = dst;

	param[0]=1.;

	for (int i=1;i<wpass;i++)
	{
		SWITCH_VAL(IplImage*,TmpSrc,TmpDest);
		param[3]=(float) powera(2,i);
		TemplateOperator("cvgIntegral", "FShaders/satf", "",
			TmpSrc, NULL, NULL,
			TmpDest,param,4,
			TextureGrp::TEXTGRP_SAME_ALL_FORMAT,"");

	}
	for (int i=0;i<hpass;i++)
	{
		SWITCH_VAL(IplImage* , TmpSrc,TmpDest);
		param[3]=(float) powera(2,i);
		TemplateOperator("cvgIntegral", "FShaders/satfv", "",
			TmpSrc, NULL, NULL,
			TmpDest,param,4,
			TextureGrp::TEXTGRP_SAME_ALL_FORMAT,"");
	}

	cvgReleaseImage(&TmpImage);
	GPUCV_STOP_OP(cvIntegral(src1,dst,NULL,NULL),src1,NULL,NULL,dst);
}
#endif


