#include "StdAfx.h"

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
#include "mainSampleTest.h"
#include "cvg_test.h"
#include "commands.h"

#if !_GCV_CUDA_EXTERNAL
#ifdef _GPUCV_SUPPORT_CUDA
#	include <cvgcu/harr.h>
#	include <cvgcu/cvgcu.h>
#	ifdef _GPUCV_SUPPORT_NPP
#		include <gcvnpp/gcvnpp.h>
#	endif
#endif
#endif
#include <GPUCV/cv_new.h>

using namespace GCV;

//CVG_IMGPROC__GRAD_EDGE_CORNER_GRP
//CVG_IMGPROC__SAMPLE_INTER_GEO_GRP
//CVG_IMGPROC__MORPHO_GRP
//CVG_IMGPROC__FILTERS_CLR_CONV_GRP
//CVG_IMGPROC__PYRAMIDS_GRP
//CVG_IMGPROC__IMGSEGM_CC_CR_GRP
//CVG_IMGPROC__IMG_CONT_MOMENT_GRP
//CVG_IMGPROC__SPE_IMG_TRANS_GRP
//CVG_IMGPROC__HISTOGRAM_GRP
//CVG_IMGPROC__MATCHING_GRP
//CVG_IMGPROC__CUSTOM_FILTER_GRP

bool cv_test_enable = true;
void cv_test_print()
{
#if TEST_GPUCV_CV
	GPUCV_NOTICE("===<cv.h> test operators ========");
	GPUCV_NOTICE(" 'cv_all'\t : run all cv operators implemented.");
	GPUCV_NOTICE(" 'disable/enable cv'\t : enable benchmarking of cv* libraries(default=true).");
	GPUCV_NOTICE("-------------------------------------");
	GPUCV_NOTICE(" 'cvtcolor %code[1-?]'\t : cvgCvtColor()");
	GPUCV_NOTICE(" 'dilate'\t : dilate()");
	GPUCV_NOTICE(" 'erode'\t : erode()");
	GPUCV_NOTICE(" 'resize ()'\t : Test cvgResize operator()");
	GPUCV_NOTICE(" 'resizefct ()'\t : Test cvgResizeGLSLFct operator()");
	GPUCV_NOTICE(" 'sobel'\t : Sobel()");
	GPUCV_NOTICE(" 'laplace'\t : Laplace()");
	GPUCV_NOTICE(" 'threshold %threshold_value %max_value %type(1-5)'\t : threshold()");
#ifdef _GPUCV_SUPPORT_CUDA
	GPUCV_NOTICE(" 'cudahist'\t : calcHist() with CUDA");
#endif
#if _GPUCV_DEVELOP
	GPUCV_NOTICE(" 'dist'\t : distance transform cvgDistTransform()");
	GPUCV_NOTICE(" 'histo'\t : cvgHisto()");
#endif
	GPUCV_NOTICE("===============================");
#endif
}

bool cvg_processCommand(std::string & CurCmd, std::string & nextCmd)
{
	CurLibraryName="cv";
	
	if (CurCmd=="")
	{
		return false;
	}
	else if (cv_test_enable==false)
	{
		return false;
	}
#if TEST_GPUCV_CV
	else if (CurCmd=="cv_all")
	{
		cvg_runAll(&GlobSrc1, &GlobSrc2, &GlobMask);
		CmdOperator = false;
	}
	//CVG_IMGPROC__GRAD_EDGE_CORNER_GRP
	else if (CurCmd=="sobel")
	{
		GPUCV_NOTICE("\nXorder, Yorder and Aperture size?");
		int X, Y, Aperture;
		SGE::GetNextCommand(nextCmd, X);
		SGE::GetNextCommand(nextCmd, Y);
		SGE::GetNextCommand(nextCmd, Aperture);
		runSobel(GlobSrc1,X, Y, Aperture);
	}
	else if (CurCmd=="laplace")
	{
		int Aperture;
		GPUCV_NOTICE("\nAperture size?");
		SGE::GetNextCommand(nextCmd, Aperture);
		runLaplace(GlobSrc1, Aperture);
	}
	else if (CurCmd=="canny")
	{
		GPUCV_NOTICE("\nThresh1, Thresh2 and Aperture size?");
		int X, Y, Aperture;
		SGE::GetNextCommand(nextCmd, X);
		SGE::GetNextCommand(nextCmd, Y);
		SGE::GetNextCommand(nextCmd, Aperture);
		runCanny(GlobSrc1,X, Y, Aperture);
	}
	//CVG_IMGPROC__SAMPLE_INTER_GEO_GRP
	else if (CurCmd=="resize")
	{
		GPUCV_NOTICE("\nwidth and height?");
		int Width, Height;
		SGE::GetNextCommand(nextCmd, Width);
		SGE::GetNextCommand(nextCmd, Height);
		runResize(GlobSrc1, Width, Height);
	}
	//CVG_IMGPROC__MORPHO_GRP
	else if (CurCmd=="dilate")
	{
		int pos;
		int iter;
		GPUCV_NOTICE("\nPos [3|5]:");
		SGE::GetNextCommand(nextCmd, pos);
		GPUCV_NOTICE("\nIteration number?");
		SGE::GetNextCommand(nextCmd, iter);
		runDilate(GlobSrc1, pos, iter);
	}
	else if (CurCmd=="erode")
	{
		int pos;
		int iter;
		GPUCV_NOTICE("\nPos [3|5]:");
		SGE::GetNextCommand(nextCmd, pos);
		GPUCV_NOTICE("\nIteration number?");
		SGE::GetNextCommand(nextCmd, iter);
		runErode(GlobSrc1, pos, iter);
	}
	else if (CurCmd=="morpho")
	{
		int pos;
		int iter;
		GPUCV_NOTICE("\nPos [3|5]:");
		SGE::GetNextCommand(nextCmd, pos);
		GPUCV_NOTICE("\nIteration number?");
		SGE::GetNextCommand(nextCmd, iter);
		runMorpho(GlobSrc1, pos, iter);
	}
	//CVG_IMGPROC__FILTERS_CLR_CONV_GRP
	else if (CurCmd=="cvtcolor")
	{
		int type;
		GPUCV_NOTICE("\nTransformation type :");
		GPUCV_NOTICE("\n\t 1- CV_RGB2XYZ");
		GPUCV_NOTICE("\n\t 2- CV_BGR2XYZ");
		GPUCV_NOTICE("\n\t 3- CV_XYZ2RGB");
		GPUCV_NOTICE("\n\t 4- CV_XYZ2BGR");
		GPUCV_NOTICE("\n\t 5- CV_RGB2YCrCb");
		GPUCV_NOTICE("\n\t 6- CV_BGR2YCrCb");
		GPUCV_NOTICE("\n\t 7- CV_YCrCb2RGB");
		GPUCV_NOTICE("\n\t 8- CV_YCrCb2BGR");
		GPUCV_NOTICE("\n\t 9- CV_RGB2HSV");
		GPUCV_NOTICE("\n\t 10- CV_RGB2Lab");
		GPUCV_NOTICE("\n\t 11- CV_BGR2Lab");
		GPUCV_NOTICE("\n\t 12- CV_RGB2GRAY");
		GPUCV_NOTICE("\n\t 13- CV_BGR2GRAY");
		SGE::GetNextCommand(nextCmd, type);

		//if (threshtype!=0)
		runCvtColor(GlobSrc1,type);
	}
	else if (CurCmd=="dist")
	{runDist(GlobSrc1);}
	else if (CurCmd=="threshold")
	{
		int threshtype;
		double threshold, max_value;
		GPUCV_NOTICE("\nThreshold level :");
		SGE::GetNextCommand(nextCmd, threshold);
		//SGE::GetNextCommand(nextCmd, threshold;
		GPUCV_NOTICE("\nThreshold max value :");
		SGE::GetNextCommand(nextCmd, max_value);
		GPUCV_NOTICE("\nThreshold type :");
		GPUCV_NOTICE("\n\t 1- CV_THRESH_BINARY");
		GPUCV_NOTICE("\n\t 2- CV_THRESH_BINARY_INV");
		GPUCV_NOTICE("\n\t 3- CV_THRESH_TRUNC");
		GPUCV_NOTICE("\n\t 4- CV_THRESH_TOZERO");
		GPUCV_NOTICE("\n\t 5- CV_THRESH_TOZERO_INV\n?");
		SGE::GetNextCommand(nextCmd, threshtype);
		switch(threshtype)
		{
		case 1 : threshtype = CV_THRESH_BINARY;break;
		case 2 : threshtype = CV_THRESH_BINARY_INV;break;
		case 3 : threshtype = CV_THRESH_TRUNC;break;
		case 4 : threshtype = CV_THRESH_TOZERO;break;
		case 5 : threshtype = CV_THRESH_TOZERO_INV;break;
		default:printf("Unknown type %d",threshtype);threshtype =0;
		}
		//if (threshtype!=0)
		runThreshold(GlobSrc1,threshold, max_value, threshtype);
	}
	else if (CurCmd=="smooth")
	{
		int type;
		GPUCV_NOTICE("\nSmooth type:");
		GPUCV_NOTICE("\t 1- CV_BLUR_NO_SCALE");
		GPUCV_NOTICE("\t 2- CV_BLUR");
		GPUCV_NOTICE("\t 3- CV_GAUSSIAN");
		GPUCV_NOTICE("\t 4- CV_MEDIAN");
		GPUCV_NOTICE("\t 5- CV_BILATERAL");
		SGE::GetNextCommand(nextCmd, type);

		int Size[2]={3,0};
		double Params[2]={0,0};
		switch(type)
		{
		case CV_BLUR_NO_SCALE://0
		case CV_BLUR://1
		case CV_GAUSSIAN://2
			//params 1 and 2
			GPUCV_NOTICE("\nSize X and Y?");
			SGE::GetNextCommand(nextCmd, Size[0]);
			SGE::GetNextCommand(nextCmd, Size[1]);
			break;
		case CV_MEDIAN://3
			GPUCV_NOTICE("\nSize X?(square)");
			SGE::GetNextCommand(nextCmd, Size[0]);
			break;
		case CV_BILATERAL://4
			GPUCV_ERROR("NOT DONE");
			break;
		}
		runSmooth(GlobSrc1, type, Size[0],Size[1], Params[0], Params[1]);
	}
	else if (CurCmd=="integral")		runIntegral(GlobSrc1);

	//CVG_IMGPROC__PYRAMIDS_GRP
	//CVG_IMGPROC__IMGSEGM_CC_CR_GRP
	//CVG_IMGPROC__IMG_CONT_MOMENT_GRP
	//CVG_IMGPROC__SPE_IMG_TRANS_GRP
#if _GPUCV_DEVELOP_BETA
	else if (CurCmd=="dist")			runDist(GlobSrc1);
#endif
	//CVG_IMGPROC__HISTOGRAM_GRP
	else if (CurCmd=="histo64")			runHisto(GlobSrc1,64);
	else if (CurCmd=="histo256")		runHisto(GlobSrc1,256);
	//CVG_IMGPROC__MATCHING_GRP
	else if (CurCmd=="facedetect"|| CurCmd=="fd")
	{
		runFaceDetect(GlobSrc1);
		CmdOperator = false;
	}
#endif
	//CVG_IMGPROC__CUSTOM_FILTER_GRP
	else if (CurCmd=="deriche")
	{
		float param;
		GPUCV_NOTICE("\nParam1 ?");
		SGE::GetNextCommand(nextCmd, param);
		runDeriche(GlobSrc1,param);
	}
	else if (CurCmd=="localsum")
	{
		std::string Curmd2;
		//!\todo add help here...
		//!\todo rename variables....
		SGE::GetNextCommand(nextCmd, Curmd2);
		std::istringstream b(Curmd2);
		float scale=0;
		b >> scale;
		SGE::GetNextCommand(nextCmd, Curmd2);
		std::istringstream c(Curmd2);
		float shift=0;
		c >> shift;
		runLocalSum(GlobSrc1,scale,shift);
	}
	else return false;

	return true;
}
//
void cvg_runAll(IplImage **src1, IplImage ** src2, IplImage ** mask)
{
	if (cv_test_enable==false)
	{
		return;
	}
	CurLibraryName="cv";
	GPUCV_NOTICE("Benchmarking CV libs");
	runDeriche(GlobSrc1,1);
	//runLocalSum(GlobSrc1, 10,10);

#if 1
	//CVG_IMGPROC__GRAD_EDGE_CORNER_GRP
	runSobel(GlobSrc1, 1, 0, 3);
	/*		runLaplace(GlobSrc1, 3);
	runLaplace(GlobSrc1, 1);
	#if _GPUCV_DEVELOP_BETA

	runCanny(GlobSrc1, 1, 1, 3);
	#endif
	//CVG_IMGPROC__SAMPLE_INTER_GEO_GRP
	runResize(*src1, 64, 64);
	runResize(*src1, 128, 128);
	runResize(*src1, 256, 256);
	runResize(*src1, 512, 512);
	runResize(*src1, 1024, 1024);
	runResize(*src1, 2048, 2048);
	//CVG_IMGPROC__MORPHO_GRP
	*/		for (int Seqi=3; Seqi < 6; Seqi+=2)
	{//filter size
		for(int iter=1; iter < 2; iter+=2)
		{//iteration nb
			runDilate(*src1,Seqi, iter);
			runErode(*src1,Seqi, iter);
			runMorpho(*src1,Seqi, iter);
		}
	}
	//CVG_IMGPROC__FILTERS_CLR_CONV_GRP
	for (int i=1; i<=13; i++)
		runCvtColor(*src1,i);
	for (int i=1; i<=4; i++)
		runThreshold(*src1,128,255,i);
#if _GPUCV_DEVELOP_BETA
	//runSmooth(GlobSrc1, type, Size[0],Size[1], Params[0], Params[1]);
#endif
	//runIntegral(GlobSrc1);
	//CVG_IMGPROC__PYRAMIDS_GRP
	//CVG_IMGPROC__IMGSEGM_CC_CR_GRP
	//CVG_IMGPROC__IMG_CONT_MOMENT_GRP
	//CVG_IMGPROC__SPE_IMG_TRANS_GRP
	//CVG_IMGPROC__HISTOGRAM_GRP
#endif	
	runHisto(GlobSrc1,64);
	runHisto(GlobSrc1,256);



	//runFaceDetect(GlobSrc1);

}
//=======================================================
void runCvtColor(IplImage * src1, int type)
{
	if(src1->nChannels==1)
	{
		GPUCV_WARNING("runCvtColor()->single channel source image");
		return;
	}
	GPUCV_FUNCNAME("CvtColor");
	std::string localParams="";
	IplImage * src2=NULL;


	switch(type)
	{
	case 1 : type = CV_RGB2XYZ;localParams ="CV_RGB2XYZ"; break;
	case 2 : type = CV_BGR2XYZ;localParams ="CV_BGR2XYZ";break;
	case 3 : type = CV_XYZ2RGB;localParams ="CV_XYZ2RGB";break;
	case 4 : type = CV_XYZ2BGR;localParams ="CV_XYZ2BGR";break;
	case 5 : type = CV_RGB2YCrCb;localParams ="CV_RGB2YCrCb";break;
	case 6 : type = CV_BGR2YCrCb;localParams ="CV_BGR2YCrCb";break;
	case 7 : type = CV_YCrCb2RGB;localParams ="CV_YCrCb2RGB";break;
	case 8 : type = CV_YCrCb2BGR;localParams ="CV_YCrCb2BGR";break;
	case 9 : type = CV_RGB2HSV;localParams ="CV_RGB2HSV";break;
	case 10 : type = CV_RGB2Lab;localParams ="CV_RGB2Lab";break;
	case 11 : type = CV_BGR2Lab;localParams ="CV_BGR2Lab";break;
	case 12 : type = CV_RGB2GRAY;localParams ="CV_RGB2GRAY";break;
	case 13 : type = CV_BGR2GRAY;localParams ="CV_BGR2GRAY";break;

	default:GPUCV_WARNING("Unknown type " << type);type =0;
	}

	char Channels = src1->nChannels;

	if ((type==CV_RGB2GRAY)||(type==CV_BGR2GRAY))
	{//dst must be gray
		Channels = 1;
	}
	__CreateImages__(cvGetSize(src1) ,src1->depth, Channels, OperALL);
	__CreateWindows__();

	_SW_benchLoop(cvgswCvtColor(src1, destSW, type) ,localParams);
	_CV_benchLoop(cvCvtColor(src1, destCV, type) ,localParams);
	_GPU_benchLoop(cvgCvtColor(src1, destGLSL, type), destGLSL,localParams);
	_CUDA_benchLoop(cvgCudaCvtColor(src1, destCUDA, type), destCUDA,localParams);

	__ShowImages__();
	__ReleaseImages__();
}
//======================================================

void runConnectedComp(IplImage * src1)// int type)
{
	GPUCV_FUNCNAME("ConnectedComp");
#if 0//_GPUCV_VERSION
	IplImage * src2=NULL;//for compatibility issue....
	IplImage * destCV, * destGLSL ;
	IplImage * srcGray1;
	IplImage * srcGray2;


	//resizeImage(src1, 256,256);
	/*	if( (src1 = cvLoadImage("data/pictures/lena256.jpg",1)) == 0 )
	if( (src1 = cvLoadImage("../GPUCVExport/data/pictures/lena256.jpg",1)) == 0 )//for debugging under MS Visual C++
	exit(0);
	*/
	destCV  = cvgCreateImage(cvGetSize(src1),8,3);
	destGLSL = cvgCreateImage(cvGetSize(src1),8,3);
	srcGray1  = cvgCreateImage(cvGetSize(src1),8,1);
	srcGray2  = cvgCreateImage(cvGetSize(src1),8,1);
	cvCvtColor(src1, srcGray1, CV_BGR2GRAY);
	cvgThreshold(srcGray1, srcGray2, 128, 255,1);


	const int element_shape = CV_SHAPE_RECT;
	int pos = (3 -1)/2;
	IplConvKernel* element = cvCreateStructuringElementEx( pos*2+1, pos*2+1, pos, pos, element_shape, 0 );
	cvgDilate(srcGray2, srcGray1, element,1);
	cvgErode(srcGray1, srcGray2, element,1);

	__CreateWindows__();

	//_CV_benchLoop(cvgConnectedComp(srcGray1, destCV) ,"");
	_GPU_benchLoop(cvgConnectedComp2(srcGray2 , destGLSL), destGLSL,"");
	//   _GPU_benchLoop(cvgConnectedComp2(srcGray1 , destGLSL), destGLSL,"");




	IplImage* src = srcGray1;
	// the first command line parameter must be file name of binary (black-n-white) image
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	/*
	IplImage* dst = cvCreateImage( cvGetSize(src), 8, 3 );


	cvThreshold( src, src, 1, 255, CV_THRESH_BINARY );
	cvNamedWindow( "Source", 1 );
	cvShowImage( "Source", src );
	*/
	_CV_benchLoop(cvFindContours( srcGray1, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE ),"");
	cvZero( destCV );

	for( ; contour != 0; contour = contour->h_next )
	{
		CvScalar color = CV_RGB( rand()&255, rand()&255, rand()&255 );
		// replace CV_FILLED with 1 to see the outlines
		cvDrawContours( destCV, contour, color, color, -1, CV_FILLED, 8 );
	}

	/*  cvNamedWindow( "Components", 1 );
	cvShowImage( "Components", destCV );
	cvWaitKey(0);
	*/
	__ShowImages__();
	__ReleaseImages__();
	cvgReleaseImage(&srcGray2);
	cvgReleaseImage(&srcGray1);
#endif
}

//=======================================================
void runDilate(IplImage * src1,int pos, int iter/*=1*/ )
{
	GPUCV_FUNCNAME("Dilate");
	IplImage * src2=NULL;
	int forceChannel = src1->nChannels;//1

	__CreateImages__(cvGetSize(src1) ,src1->depth, forceChannel, OperALL);

	const int element_shape = CV_SHAPE_RECT;
	pos = (pos -1)/2;
	IplConvKernel* element = cvCreateStructuringElementEx( pos*2+1, pos*2+1, pos, pos, element_shape, 0 );

	__CreateWindows__();
	std::string localParams="Size_";
	localParams+= SGE::ToCharStr(pos*2+1);
	localParams+="/Iteration_";
	localParams+= SGE::ToCharStr(iter);

	IplImage * Src1Gray = NULL;
	IplImage * TempSrc = src1;
	if(forceChannel==1 && src1->nChannels!=forceChannel)
	{
		Src1Gray = cvgCreateImage(cvGetSize(src1), src1->depth, 1);
		cvCvtColor(src1, Src1Gray, CV_BGR2GRAY);
		TempSrc = Src1Gray;
	}
	__SetInputImage__(TempSrc, "DilateSrcGray");

	_SW_benchLoop(cvgswDilate(TempSrc,destSW,element,iter),localParams);
	_CV_benchLoop(cvDilate(TempSrc,destCV,element,iter),localParams);
	_GPU_benchLoop(cvgDilate(TempSrc,destGLSL,element,iter),destGLSL, localParams);
#ifdef _GPUCV_SUPPORT_NPP
	_CUDA_benchLoop(cvgNppDilate(TempSrc,destCUDA,element,iter),destCUDA, localParams);
#endif		
	if(pos <=5)
	{
		#if _GPUCV_DEVELOP_BETA
		_CUDA_benchLoop(cvgCudaDilate(TempSrc,destCUDA,element,iter),destCUDA, localParams);
		#endif// _GPUCV_DEVELOP_BETA
	}

	cvReleaseStructuringElement(&element);
	if(forceChannel==1)
		cvgReleaseImage(&Src1Gray);
	__ShowImages__();
	__ReleaseImages__();

}
//=======================================================
void runErode(IplImage * src1, int pos, int iter/*=1*/)
{
	GPUCV_FUNCNAME("Erode");
	IplImage * src2=NULL;
	int forceChannel = src1->nChannels;//1
	__CreateImages__(cvGetSize(src1) ,src1->depth, forceChannel, OperALL-OperCuda);

	pos = (pos -1)/2;
	const int element_shape = CV_SHAPE_RECT;
	IplConvKernel* element = cvCreateStructuringElementEx( pos*2+1, pos*2+1, pos, pos, element_shape, 0 );

		std::string localParams="Size_";
	localParams+= SGE::ToCharStr(pos*2+1);
	localParams+="/Iteration_";
	localParams+= SGE::ToCharStr(iter);

	__CreateWindows__();

	IplImage * Src1Gray = NULL;
	IplImage * TempSrc = src1;
	if(forceChannel==1&& src1->nChannels!=forceChannel)
	{
		Src1Gray = cvgCreateImage(cvGetSize(src1), src1->depth, 1);
		cvCvtColor(src1, Src1Gray, CV_BGR2GRAY);
		TempSrc = Src1Gray;
		__SetInputImage__(TempSrc, "DilateSrcGray");
	}

	_SW_benchLoop(cvgswErode(TempSrc,destSW,element,iter),localParams);
	_CV_benchLoop(cvErode(TempSrc,destCV,element,iter),localParams);
	_GPU_benchLoop(cvgErode(src1,destGLSL,element,iter), destGLSL,localParams);

	if(pos <=5)
	{
		#if _GPUCV_DEVELOP_BETA
			_CUDA_benchLoop(cvgCudaErode(TempSrc,destCUDA,element,iter),destCUDA, localParams);
		#endif
	}
	//_CUDA_benchLoop("Erode2", cvgCudaErodeNoShare(TempSrc,destCUDA,element,iter),destCUDA, localParams);

	cvReleaseStructuringElement(&element);
	if(forceChannel==1)
		cvgReleaseImage(&Src1Gray);
	__ShowImages__();
	__ReleaseImages__();

}
//=======================================================
void runMorpho(IplImage * src1, int pos, int iter/*=1*/)
{
	GPUCV_FUNCNAME("MorphologyEx");
	IplImage * src2=NULL;
	int forceChannel = src1->nChannels;//1

	__CreateImages__(cvGetSize(src1),  src1->depth, forceChannel, OperALL-OperCuda);
	IplImage * TempImage = cvCloneImage(src1);

	pos = (pos -1)/2;
	const int element_shape = CV_SHAPE_RECT;
	IplConvKernel* element = cvCreateStructuringElementEx( pos*2+1, pos*2+1, pos, pos, element_shape, 0 );

		std::string localParams="Size_";
	localParams+= SGE::ToCharStr(pos*2+1);
	localParams+="/Iteration_";
	localParams+= SGE::ToCharStr(iter);

	__CreateWindows__();

	IplImage * Src1Gray = NULL;
	IplImage * TempSrc = src1;
	if(forceChannel==1&& src1->nChannels!=forceChannel)
	{
		Src1Gray = cvgCreateImage(cvGetSize(src1), src1->depth, 1);
		cvCvtColor(src1, Src1Gray, CV_BGR2GRAY);
		TempSrc = Src1Gray;
	}
	__SetInputImage__(TempSrc, "DilateSrcGray");

	_SW_benchLoop(cvgswMorphologyEx	(TempSrc, destSW,	TempImage, element,	CV_MOP_OPEN, iter), localParams);
	_CV_benchLoop(cvMorphologyEx	(TempSrc, destCV,	TempImage, element,	CV_MOP_OPEN, iter), localParams);
	_GPU_benchLoop(cvgMorphologyEx	(TempSrc, destGLSL, TempImage, element,	CV_MOP_OPEN, iter), destGLSL,localParams);

	if(pos <=5)
	{
		//_CUDA_benchLoop("Erode", cvgCudaMorphologyExErode(TempSrc,destCUDA,element,iter),destCUDA, localParams);
	}
	//_CUDA_benchLoop("Erode2", cvgCudaErodeNoShare(TempSrc,destCUDA,element,iter),destCUDA, localParams);

	cvReleaseStructuringElement(&element);
	if(forceChannel==1)
		cvgReleaseImage(&Src1Gray);
	__ShowImages__();
	__ReleaseImages__();
	cvgReleaseImage(&TempImage);
}
//=======================================================
void runHisto(IplImage *src1, int _bins)
{
	GPUCV_FUNCNAME("CalcHist");
#if 1
	IplImage *src2=NULL;
	__CreateImages__(cvSize(255,255),8,3, OperALL);

	IplImage  *planes[1];
	CvHistogram *red_histCV, *red_histGPU, *red_histCUDA, *red_histSW;
	int nb_buckets = _bins;
	int sizes[1];
	//	float val_histo, max_val_histo;
	//	int val_graph;
	float red_range[] = {0,255};
	float* ranges[1];


	//------------------------------------------------------
	// Creating Specific display list for Histogram
	// DEBUG PURPOSE ONLY
	/*	GLuint liste = glGenLists(1);
	glNewList(liste, GL_COMPILE);
	glBegin(GL_POINTS);
	glPushMatrix();
	//int cpt = 0;
	for(float i=0;i<=2;i+=2/((float)(src1->height)))
	for(float j=0;j<=2;j+=2/((float)(src1->width)))
	{
	glTexCoord2f(j/2.0, i/2.0);
	glVertex3f(j-1., i-1., -0.5f);
	}
	glPopMatrix();
	glEnd();
	glEndList();
	*/
	//------------------------------------------------------
	// Creating and Computing Histogram
	if(src1->nChannels!=1)
	{
		planes[0] = cvgCreateImage(cvGetSize(src1), src1->depth, 1);
		cvCvtColor(src1, planes[0], CV_BGR2GRAY);
	}
	else
		planes[0] = cvCloneImage(src1);

	cvgSetLabel(planes[0], "CalcHisto_Src1Mono");

	sizes[0]  = nb_buckets;
	ranges[0] = red_range;
	red_histSW		= cvCreateHist(1,sizes,CV_HIST_ARRAY,ranges,1);
	red_histCV		= cvCreateHist(1,sizes,CV_HIST_ARRAY,ranges,1);
	red_histGPU		= cvCreateHist(1,sizes,CV_HIST_ARRAY,ranges,1);
	red_histCUDA	= cvCreateHist(1,sizes,CV_HIST_ARRAY,ranges,1);


	//=========creating FLOAT32 texture for CVGcalcHist2:
#if 0
	GLuint	FloatTex=0;
	glGenTextures(1,&FloatTex);
	glBindTexture(GL_TEXTURE_2D, FloatTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA_FLOAT32_ATI,
		planes[0]->width, planes[0]->height,
		0,
		GL_LUMINANCE,GL_UNSIGNED_BYTE,
		planes[0]->imageData);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
#endif
	//==========================================


	cvgSetOptions(planes[0], DataContainer::UBIQUITY, true);

	std::string Params = "Bin "+SGE::ToCharStr(_bins);

	if (AvoidCPUReturn)
		cvgUnsetCpuReturn(planes[0]);
	//	if (AvoidCPUReturn)cvgUnsetCpuReturn(red_hist2);

	_SW_benchLoop(cvgswCalcHist(planes,red_histSW), Params)
	_CV_benchLoop(cvCalcHist(planes,red_histCV), Params)
	_GPU_benchLoop(cvgCalcHist(planes,red_histGPU), NULL, Params)//,liste, FloatTex )
	_CUDA_benchLoop(cvgCudaCalcHist(planes,red_histCUDA), NULL, Params)
	
	cvgSetOptions(planes[0], DataContainer::UBIQUITY, false);
	cvgSetLocation<DataDsc_IplImage>(planes[0], true);


		// Creating Histogram Image
		if(ShowImage)
		{
			__CreateWindows__()
				cvNamedWindow("SourceImg",1);
			cvgShowImage("SourceImg",planes[0]);
			float max_val_histo;
			if(destCV)
			{
				cvZero(destCV);
				cvgCreateHistImage(destCV, red_histCV, CV_RGB(255,0,0));
				//cvgCreateHistImage(destCV, red_hist2, CV_RGB(0,255,0));
				cvGetMinMaxHistValue(red_histCV,0,&max_val_histo, 0, 0 );
				GPUCV_DEBUG("cvHisto, max val:" << max_val_histo);
			}
			if(destGLSL)
			{
				cvZero(destGLSL);
				cvgCreateHistImage(destGLSL, red_histGPU, CV_RGB(0,255,0));
				//cvgCreateHistImage(destGLSL, red_hist, CV_RGB(255,0,0));
				cvGetMinMaxHistValue(red_histGPU,0,&max_val_histo, 0, 0 );
				GPUCV_DEBUG("cvgHisto, max val:" << max_val_histo);
			}
			if(destCUDA)
			{
				cvZero(destCUDA);
				cvgCreateHistImage(destCUDA, red_histCUDA, CV_RGB(0,0,255));
				cvGetMinMaxHistValue(red_histCUDA,0,&max_val_histo, 0, 0 );
				GPUCV_DEBUG("cvgCudaHisto, max val:" << max_val_histo);
			}
			__ShowImages__()
				cvDestroyWindow("SourceImg");
			cvReleaseHist(&red_histCUDA);
			cvReleaseHist(&red_histGPU);
			cvReleaseHist(&red_histCV);
			cvReleaseHist(&red_histSW);
		}

		cvgReleaseImage(&planes[0]);
		__ReleaseImages__();


		return;
#endif//_GPUCV_VERSION
}
//======================================================
void runSobel(IplImage * src1, int xOrder/*=1*/, int yOrder/*=1*/, int aperture_size/*=3*/)
{
	GPUCV_FUNCNAME("Sobel");
	IplImage * src2=NULL;//for compatibility issue....
	//convert images

	IplImage *srcGray1 = NULL;
	__CreateImages__(cvGetSize(src1) , IPL_DEPTH_32F, 1, OperALL);

	if (src1->nChannels !=1)
	{
		srcGray1 = cvgCreateImage(cvGetSize(src1),src1->depth,1);
		GPUCV_DEBUG("\nConverting Src image to gray");
		cvCvtColor(src1, srcGray1, CV_BGR2GRAY);
	}
	else
	{
		GPUCV_DEBUG("\nCloning Src images");
		srcGray1 = cvCloneImage(src1);
	}
	//=====================================
	cvgSetLabel(srcGray1, "cvSobel_srcGray1");
	//cvgSetOptions(srcGray1, DataContainer::UBIQUITY, true);


	string localParams="";
	localParams+= "X-";
	localParams+= SGE::ToCharStr(xOrder);
	localParams+= "/Y-";
	localParams+= SGE::ToCharStr(yOrder);
	localParams+= "/Aper-";
	localParams+= SGE::ToCharStr(aperture_size);

	__CreateWindows__();

	//	int xOrder = 1;
	//	int yOrder = 0;
	//	int aperture_size = 3;
	__SetInputImage__(srcGray1, "SobelSrcGray");

	_SW_benchLoop(cvgswSobel(srcGray1, destSW, xOrder,yOrder,aperture_size) ,localParams);
	_CV_benchLoop(cvSobel(srcGray1, destCV, xOrder,yOrder,aperture_size) ,localParams);
	_GPU_benchLoop(cvgSobel(srcGray1, destGLSL, xOrder,yOrder,aperture_size), destGLSL,localParams);
	_CUDA_benchLoop(cvgCudaSobel(srcGray1, destCUDA, xOrder,yOrder,aperture_size),destCUDA ,localParams);

	cvgReleaseImage(&srcGray1);
	__ShowImages__();
	__ReleaseImages__();
}
//======================================================
#if 1//_GPUCV_DEVELOP_BETA
void runLaplace(IplImage * src1, int aperture_size/*=3*/)
{
	GPUCV_FUNCNAME("Laplace");
	IplImage * src2=NULL;	//for compatibility issue...
	//convert images
	IplImage *srcGray1 = NULL;
	__CreateImages__(cvGetSize(src1) ,IPL_DEPTH_16S, 1, OperALL - OperCuda);

	//GLSL does not support SHORT16
	cvgReleaseImage(&destGLSL);
	destGLSL = cvgCreateImage(cvGetSize(src1) ,IPL_DEPTH_8S, 1);\
		cvgSetLabel(destGLSL,GPUCV_GET_FCT_NAME()+"-destGLSL");\
		//=============================

		if (src1->nChannels !=1)
		{
			GPUCV_DEBUG("\nConverting Src image to gray");
			srcGray1 = cvgCreateImage(cvGetSize(src1),src1->depth,1);
			cvCvtColor(src1, srcGray1, CV_BGR2GRAY);
		}
		else
		{
			GPUCV_DEBUG("\nCloning Src images");
			srcGray1 = cvCloneImage(src1);
		}
		//=====================================
		string localParams=SGE::ToCharStr(aperture_size);

		__CreateWindows__();
		__SetInputImage__(srcGray1, "cvlaplace_srcGray1");

		_SW_benchLoop(cvgswLaplace(srcGray1, destSW, aperture_size) ,localParams);
		_CV_benchLoop(cvLaplace(srcGray1, destCV, aperture_size) ,localParams);
		if(__CurrentOperMask & OperOpenCV)
		{
			cvScale(destCV, destCV, 2);
		}
		_GPU_benchLoop(cvgLaplace(srcGray1, destGLSL, aperture_size), destGLSL,localParams);
#if _GPUCV_DEVELOP_BETA
		_CUDA_benchLoop(cvgCudaLaplace(srcGray1, destCUDA, aperture_size),destCUDA ,localParams);
#endif// _GPUCV_DEVELOP_BETA
		__ShowImages__();
		__ReleaseImages__();
		cvgReleaseImage(&srcGray1);
}
void runCanny(IplImage * src1, int thresh1, int thresh2, int aperture_size)
{
	GPUCV_FUNCNAME("Canny");
	IplImage * src2=NULL;	//for compatibility issue...
	//convert images
	IplImage *srcGray1 = NULL;
	__CreateImages__(cvGetSize(src1) ,src1->depth, 1, OperALL);

	if (src1->nChannels !=1)
	{
		GPUCV_DEBUG("\nConverting Src image to gray");
		srcGray1 = cvgCreateImage(cvGetSize(src1),src1->depth,1);
		cvCvtColor(src1, srcGray1, CV_BGR2GRAY);
	}
	else
	{
		GPUCV_DEBUG("\nCloning Src images");
		srcGray1 = cvCloneImage(src1);
	}
	//=====================================
	string localParams="";

	__CreateWindows__();
	__SetInputImage__(srcGray1, "cvCanny_srcGray1");

	_SW_benchLoop(cvgswCanny(srcGray1, destSW, thresh1, thresh2, aperture_size) ,localParams);
	_CV_benchLoop(cvCanny(srcGray1, destCV, thresh1, thresh2, aperture_size) ,localParams);
	//_GPU_benchLoop(cvgCanny(srcGray1, destGLSL, thresh1, thresh2, aperture_size), destGLSL,localParams);

#if 0//def _GPUCV_SUPPORT_NPP	
	//this is a try
	CvgArr * pImg = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(destCUDA));	
	pImg->GetDataDsc<DataDsc_CUDA_Buffer>()->SetAutoMapGLBuff(false);
	pImg = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(srcGray1));	
	pImg->GetDataDsc<DataDsc_CUDA_Buffer>()->SetAutoMapGLBuff(false);
	//
	_CUDA_benchLoop(cvgNppCanny(srcGray1, destCUDA, thresh1, thresh2, aperture_size),destCUDA ,localParams);
#endif
	__ShowImages__();
	__ReleaseImages__();
	cvgReleaseImage(&srcGray1);
}
#endif
//=======================================================
void runSmooth(IplImage * src1, int smoothtype, int param1, int param2, double param3, double param4)
{
	GPUCV_FUNCNAME("Smooth");
	IplImage * src2=NULL;//for compatibility issue....
	//convert images

	IplImage *srcGray1 = NULL;
	__CreateImages__(cvGetSize(src1) ,IPL_DEPTH_8U, 1, OperOpenCV-OperCuda);

	if (src1->nChannels !=1)
	{
		srcGray1 = cvgCreateImage(cvGetSize(src1),src1->depth,1);
		GPUCV_DEBUG("\nConverting Src image to gray");
		cvCvtColor(src1, srcGray1, CV_BGR2GRAY);
	}
	else
	{
		GPUCV_DEBUG("\nCloning Src images");
		srcGray1 = cvCloneImage(src1);
	}
	//=====================================
	string localParams="";
	__CreateWindows__();
	__SetInputImage__(srcGray1, "SmoothSrcGray");
	//add smooth type parameter
	switch(smoothtype)
	{
	case CV_BLUR_NO_SCALE:	localParams+="CV_BLUR_NO_SCALE";break;
	case CV_BLUR:			localParams+="CV_BLUR";break;
	case CV_GAUSSIAN:		localParams+="CV_GAUSSIAN";break;
	case CV_MEDIAN:			localParams+="CV_MEDIAN";break;
	case CV_BILATERAL:		localParams+="CV_BILATERAL";break;
	}
	localParams+="-";
	//add size
	switch(smoothtype)
	{
	case CV_BLUR_NO_SCALE:
	case CV_BLUR:
	case CV_GAUSSIAN:		localParams+="X "+SGE::ToCharStr(param1)+" -Y "+SGE::ToCharStr(param2);break;
	case CV_MEDIAN:			localParams+="X "+SGE::ToCharStr(param1)+" -Y "+SGE::ToCharStr(param1);break;
	case CV_BILATERAL:		localParams+="X 3 - Y 3";break;
	}

	_SW_benchLoop(cvgswSmooth(srcGray1, destSW, smoothtype, param1, param2, param3, param4) ,localParams);

	if(smoothtype!=CV_BLUR_NO_SCALE)
	{//don't know why it is not working with CV_BLUR_NO_SCALE
		_CV_benchLoop(cvSmooth(srcGray1, destCV, smoothtype, param1, param2, param3, param4) ,localParams);
	}
	//    _GPU_benchLoop("cvSmooth",cvgSmooth(srcGray1, destGLSL,smoothtype, param1, param2, param3, param4), destGLSL,localParams);
	if(smoothtype == CV_BLUR_NO_SCALE || smoothtype == CV_BLUR)
	{
#if _GPUCV_DEVELOP_BETA
		_CUDA_benchLoop(cvgCudaSmooth(srcGray1, destCUDA, smoothtype, param1, param2, param3, param4),destCUDA ,localParams);
#endif// _GPUCV_DEVELOP_BETA
	}

	cvgReleaseImage(&srcGray1);
	__ShowImages__();
	__ReleaseImages__();
}
//=======================================================

void runThreshold(IplImage * src1, double threshold, double max_value, int type)
{
	GPUCV_FUNCNAME("Threshold");
	IplImage *src2=NULL;
	IplImage *srcGray1 = NULL;
	__CreateImages__(cvGetSize(src1) ,src1->depth, 1, OperALL);

	if (src1->nChannels !=1)
	{
		srcGray1 = cvgCreateImage(cvGetSize(src1),src1->depth,1);
		cvCvtColor(src1, srcGray1, CV_BGR2GRAY);
	}
	else
	{
		srcGray1 = cvCloneImage(src1);
	}
	//=====================================
	string localParams;
	switch (type)
	{
	case 1: localParams ="CV_THRESH_BINARY" ; break;
	case 2: localParams ="CV_THRESH_BINARY_INV" ; break;
	case 3: localParams ="CV_THRESH_TRUNC" ; break;
	case 4: localParams ="CV_THRESH_TOZERO" ; break;
	case 5: localParams ="CV_THRESH_TOZERO_INV" ; break;
	}
	__CreateWindows__();
	__SetInputImage__(srcGray1, "ThresholdSrcGray");
	
	_SW_benchLoop(cvgswThreshold(srcGray1,destSW, threshold, max_value, type), localParams);
	_CV_benchLoop(cvThreshold(srcGray1,destCV, threshold, max_value, type), localParams);
	_GPU_benchLoop(cvgThreshold(srcGray1,destGLSL, threshold, max_value, type), destGLSL, localParams);
	_CUDA_benchLoop(cvgCudaThreshold(srcGray1, destCUDA, threshold, max_value, type),destCUDA ,localParams);

	__ShowImages__();
	cvgReleaseImage(&srcGray1);
	__ReleaseImages__();

}
//=======================================================
void runResize(IplImage * src1, int width, int height)
{
	GPUCV_FUNCNAME("Resize");
	IplImage * src2=NULL;
	CvSize Size;
	Size.width  = width;
	Size.height = height;

	int interpolation = CV_INTER_NN;//CV_INTER_NN;//
	std::string localParams="CV_INTER_NN";//"CV_INTER_NN";//
	localParams+="_";
	localParams+=SGE::ToCharStr(width);
	localParams+="x";
	localParams+=SGE::ToCharStr(height);

	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperOpenCV|OperGLSL);

	IplImage * destGLSL2 = cvgCreateImage(Size,src1->depth,src1->nChannels);
	IplImage * destCV2  = cvgCreateImage(Size,src1->depth,src1->nChannels);

	cvgSetLabel(destGLSL2, "destGPU2");
	cvgSetLabel(destCV2, "destCV2");

	__CreateWindows__();
	
	_SW_benchLoop(cvgswResize(src1, destSW, interpolation), localParams);
	_CV_benchLoop(cvResize(src1, destCV, interpolation), localParams);
	_GPU_benchLoop(cvgResize(src1,destGLSL,interpolation),destGLSL , localParams);

	Size.width  = src1->width;
	Size.height = src1->height;

	if(__CurrentOperMask&OperGLSL)
		cvgResize(destGLSL,destGLSL2,CV_INTER_NN);
	cvResize(destCV,destCV2,CV_INTER_NN);
	__ShowImages__();

	cvgReleaseImage(&destGLSL2);
	cvgReleaseImage(&destCV2);
	__ReleaseImages__();
}
//=======================================================
void runResizeFct(IplImage * src1, int width, int height, char * _FctName)
{
	GPUCV_FUNCNAME("ResizeFct");
	IplImage * src2=NULL;
	CvSize Size;
	Size.width  = width;
	Size.height = height;

	__CreateImages__(cvGetSize(src1) ,8, src1->nChannels, OperOpenCV|OperGLSL);

	__CreateWindows__();
	_GPU_benchLoop(cvgResizeGLSLFct(src1,destGLSL, _FctName, NULL),destGLSL , FctName);
	cvgCopy(destGLSL, destCV);
	resizeImage(&destCV, src1->width, src1->height,CV_INTER_NN);
	__ShowImages__();
	__ReleaseImages__();
}
//=======================================================


//=======================================================================
void runIntegral(IplImage * src1)
{
	GPUCV_FUNCNAME("Integral");
	if(src1->width > _GPUCV_TEXTURE_MAX_SIZE_X
		|| src1->width > _GPUCV_TEXTURE_MAX_SIZE_Y)
	{
		GPUCV_WARNING("Integral require image inferior to _GPUCV_TEXTURE_MAX_SIZE_X or _GPUCV_TEXTURE_MAX_SIZE_Y");
		return;
	}
	//int i=0;
	CvSize Size = cvGetSize(src1);
	Size.width  +=1;
	Size.height +=1;

	IplImage * src2=NULL;
	__CreateImages__(Size, IPL_DEPTH_32F, src1->nChannels, OperCuda+OperOpenCV);
	//__CreateWindows__();

	//open 3 channels image:
#if 0
	std::string NameImg2 = GetGpuCVSettings()->GetShaderPath() + "/data/pictures/plage512.jpg";
	SGE::ReformatFilePath(NameImg2);

	if( (src2 = cvLoadImage(NameImg2.data(),1)) == 0 )
		SG_Assert(src2, "Can't load image source 1");
#else
	src2 = cvCloneImage(src1);
	//CvScalar Scal ;
	//Scal.val[0]=Scal.val[1]=Scal.val[2]=Scal.val[3] = 2.;
	//cvSet(src2,Scal );
#endif
	__SetInputImage__(src2, "Src2");
	IplImage * Sum=NULL;
	IplImage * SQSum=NULL;
	IplImage * CSQSum=NULL;
	IplImage * TiltedSum=NULL;
	std::string localParams="";



	//!\todo Add _SW_benchLoop for runIntegral. Manage several destination images
	//prepare to test OpenCV function
	if(__CurrentOperMask & OperOpenCV)
	{
		int InternalType= IPL_DEPTH_64F;
		Sum= cvCreateImage(Size, InternalType, src1->nChannels);
		SQSum= cvCreateImage(Size, IPL_DEPTH_64F , src1->nChannels);

		TiltedSum= cvCreateImage(Size, InternalType, src1->nChannels);
		localParams="Sum+SQSum+TiltedSum";
		//_CV_benchLoop("cvIntegral", cvIntegral( src1, Sum, SQSum, TiltedSum),"");
		localParams="Sum";
		_CV_benchLoop(cvIntegral( src2, Sum, NULL, NULL),localParams);
		localParams="Sum+SQSum";
		_CV_benchLoop(cvIntegral( src2, Sum, SQSum, NULL),localParams);
		if(src2->nChannels==1){//3 channels not yet implemented.
			_CV_benchLoop(cvIntegral( src2, Sum, SQSum, TiltedSum),localParams);
		}

		//cvReleaseMemStorage( &storage );
	}

	if(__CurrentOperMask & OperCuda)
	{
		CSQSum= cvCreateImage(Size, IPL_DEPTH_32F , src1->nChannels);
		__SetOutputImage__(CSQSum, "CSQSum");
#ifdef _GPUCV_CUDA_SUPPORT_CUDPP
		ControlOperators = false;
		localParams="Sum+SQSum";
		_CUDA_benchLoop(cvgCudaIntegral(src2, destCUDA,CSQSum), destCUDA,localParams);
		//ControlResultsImages(
		localParams="Sum";
		_CUDA_benchLoop(cvgCudaIntegral(src2, destCUDA), destCUDA,localParams);
		//cvgSub(CSQSum,SQSum,CSQSum,NULL);
		//cvgSub(destCUDA,destCV,destCV,NULL);
		ControlOperators = true;
#endif
	}

	if (ShowImage==true)
	{
		if(__CurrentOperMask & OperOpenCV)
		{
			cvNamedWindow("cvIntegral-SUM",1);
			cvNamedWindow("cvIntegral-SQSUM",1);
			cvNamedWindow("cvIntegral-TILTEDSUM",1);
			cvShowImage("cvIntegral-SUM",Sum);
			cvShowImage("cvIntegral-SQSUM",SQSum);
			cvShowImage("cvIntegral-TILTEDSUM",TiltedSum);
			cvgShowImage3("opencv square sum",SQSum);
		}
		if(__CurrentOperMask & OperCuda)
		{
			cvNamedWindow("cvgcuIntegral-SUM",1);
			cvNamedWindow("cvgcuIntegral-SQSUM",1);
			//cvNamedWindow("cvgcuIntegral-TILTEDSUM",1);
			cvShowImage("cvgcuIntegral-SUM",destCUDA);
			cvShowImage("cvgcuIntegral-SQSUM",CSQSum);
			//cvShowImage("cvgcuIntegral-TILTEDSUM",TiltedSum);
			cvgShowImage3("cuda sum",destCUDA);
			cvgShowImage3("cuda square sum",CSQSum);
		}
	}
	cvWaitKey(0);
}

void runFaceDetect(IplImage *src1)
{
#if 0
	GPUCV_FUNCNAME("FaceDetect");

	CvHaarClassifierCascade * pCascade = 0;
	CvMemStorage * pStorage = 0;
	CvSeq * pFaceRectSeq;
	CvSeq * pFaceRectSeq2;
	int i;

	//intializations
	IplImage * pInpImg  = src1;
	IplImage * src2=NULL;
	//pInpImg = cvgCreateImage(cvGetSize(src1), src1->depth , src1->nChannels);
	__CreateImages__(cvGetSize(src1), src1->depth, src1->nChannels, OperALL);

	__CreateWindows__();
	destCV = cvCloneImage(src1);
	destCUDA = cvCloneImage(src1);

	pStorage = cvCreateMemStorage(0);
	pCascade = (CvHaarClassifierCascade *)cvLoad (("C:/Program Files/OpenCV/data/haarcascades/haarcascade_frontalface_alt_tree.xml "),0 , 0 , 0 ) ;

	//ConvertHaarClassifierIntoIplImage(pCascade);

	__SetInputImage__(src1, "src1");
	// detect faces in image
	// use XML default for smallest

	/*pFaceRectSeq2 = cvgHaarDetectObjects
	(destCUDA,pCascade,pStorage,
	1.1,
	3,
	CV_HAAR_DO_CANNY_PRUNING,
	cvSize(0,0));*/

#ifdef _GPUCV_SUPPORT_CUDA
	pFaceRectSeq = cvgHaarDetectObjects
		(destCUDA,pCascade,pStorage,
		1.1,
		3,
		0,
		cvSize(0,0));
#endif


	_CV_benchLoop("FaceDetect",cvHaarDetectObjects
		(destCV, pCascade, pStorage,
		1.1,
		3,
		CV_HAAR_DO_CANNY_PRUNING,
		cvSize(0,0)),localParams);

	_CUDA_benchLoop("FaceDetect",cvgHaarDetectObjects
		(destCUDA, pCascade, pStorage,
		1.1,
		3,
		CV_HAAR_DO_CANNY_PRUNING,
		cvSize(0,0)),destCV,localParams);


	//draw a rectangular outline around each detection
	for(i=0;i<(pFaceRectSeq? pFaceRectSeq->total:0) ; i++ )
	{
		CvRect * r = (CvRect*)cvGetSeqElem(pFaceRectSeq, i) ;
		CvPoint pt1 = { r->x , r->y } ;
		CvPoint pt2 = { r->x + r->width, r->y + r->height };
		cvRectangle(destCUDA, pt1, pt2, CV_RGB(0,255,0), 3, 4, 0) ;
	}
	/*
	//draw a rectangular outline around each detection
	for(i=0;i<(pFaceRectSeq2? pFaceRectSeq2->total:0) ; i++ )
	{
	CvRect * r = (CvRect*)cvGetSeqElem(pFaceRectSeq, i) ;
	CvPoint pt1 = { r->x , r->y } ;
	CvPoint pt2 = { r->x + r->width, r->y + r->height };
	cvRectangle(destCUDA, pt1, pt2, CV_RGB(0,255,0), 3, 4, 0) ;
	}
	*/
	//create a window to display output
	//cvNamedWindow ( "Haar_Window", 1);

	//cvgShowImage("Haar_Window", destCV);

	__ShowImages__();

	__ReleaseImages__();

	//cvWaitKey(0);
	//cvDestroyWindow("Haar_Window");

	// clean up and release resources
	//if (pInpImg) cvReleaseImage(&pInpImg) ;
	if (pCascade) cvReleaseHaarClassifierCascade (&pCascade);
	if (pStorage) cvReleaseMemStorage (&pStorage );
#endif
}
void runDeriche(IplImage * src1, float alpha)
{
	GPUCV_FUNCNAME("Deriche");
	if(GCV::GetGLDepth(src1) != GL_FLOAT || GCV::GetnChannels(src1) != 1)
	{
		GPUCV_WARNING("Deriche requieres 32f 1 channel input image");
		return;
	}
	IplImage * src2=NULL;
	__CreateImages__(cvGetSize(src1) , src1->depth, src1->nChannels, OperCuda+OperOpenCV);
	__CreateWindows__();
	std::string localParams="";
//	_SW_benchLoop(cvgswDeriche(src1, destSW, alpha), localParams);
	_CV_benchLoop(cvDeriche(src1, destCV, alpha), localParams);
	_CUDA_benchLoop(cvgCudaDeriche(src1, destCUDA, 1.695f/alpha),destCUDA , localParams);

	__ShowImages__();
	__ReleaseImages__();
}
//==========================================

