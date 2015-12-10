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
#if 0

#include "GpuCVCamDemo.h"
#if 0
#include <cvg/cvg.h>
#include <cxcoreg/cxcoreg.h>
#include <highguig/highguig.h>
#include <cv.h>
#include <GPUCVTexture/include.h>
#include <GPUCV/include.h>
#include <cvcam.h>

#endif


using namespace GCV;
//Background substraction function
//IplImage* im  --  Current captured image
//IplImage* bg --  First image of the background without  the human
//CvMat* output -- The result of the background
void RemoveBackground (IplImage* im, IplImage* bg, IplImage* output){
	int num_rows = im->height;
	int num_cols = im->width;

	IplImage* im_smoothed = cvCreateImage (cvGetSize(im), IPL_DEPTH_8U, 3);
	IplImage* im_diff		= cvCreateImage (cvGetSize(im), IPL_DEPTH_8U, 3);
	CvMat* total_diff	= cvCreateMat(num_rows, num_cols, CV_32FC1);//cvCreateImage (cvGetSize(im), IPL_DEPTH_8U, 1);

	//smooth the original
	cvSmooth (im,im_smoothed,CV_BLUR,15,15);

	//  cvCvtColor(im_smoothed,im_smoothed,CV_BGR2YCrCb);
	//cvCvtColor(bg,bg,CV_BGR2YCrCb);
	//subtract the bg from the image, to get the difference image
	cvAbsDiff (im_smoothed, bg, im_diff);
	//quick convert it back
	//cvCvtColor(bg,bg,CV_YCrCb2BGR);

	//convert the diff to a 32-bit image
	IplImage* diff32 = cvCreateImage(cvGetSize(im),IPL_DEPTH_32F,3);
	cvConvert (im_diff,diff32);

	//  CvMat* total_diff = cvCreateMat(num_rows, num_cols,CV_32FC1);  
	CvMat** diffs = (CvMat**) malloc (3 * sizeof (CvMat*));
	for (int i = 0; i < 3; i++){
		diffs[i] = cvCreateMat(num_rows, num_cols, CV_32FC1);
	}
	cvSplit (diff32,diffs[0],diffs[1],diffs[2],NULL);

	cvZero (total_diff);

	double max_val, min_val;
	CvPoint ignore;
	//obviously this assumes the guy is already in the frame
	for (int i = 0; i < 3; i++){
		cvMinMaxLoc (diffs[i], &min_val, &max_val, &ignore, &ignore);
		cvConvertScale (diffs[i], diffs[i], 256.0 / max_val);
		//    printf ("%i %f %f\n", i, min_val, max_val);
		cvMul (diffs[i], diffs[i], diffs[i]);
	}
	//  cvMinMaxLoc (diff_edge32, &min_val, &max_val, &ignore, &ignore);
	//cvConvertScale (diff_edge32, diff_edge32, 256.0 / max_val);
	//cvMul (diff_edge32, diff_edge32, diff_edge32);

	cvConvertScale (diffs[0], diffs[0], .5 / 2.5); //down-weigh Y component
	cvConvertScale (diffs[1], diffs[1], 1.0 / 2.5); 
	cvConvertScale (diffs[2], diffs[2], 1.0 / 2.5); 
	cvAdd (diffs[0], total_diff, total_diff);
	cvAdd (diffs[1], total_diff, total_diff);
	cvAdd (diffs[2], total_diff, total_diff);
	cvPow (total_diff, total_diff, .5);  

	/*  CvMat* total_diff_smoothed = cvCloneMat (total_diff);*/
	cvSmooth (total_diff, total_diff, CV_GAUSSIAN, 11, 11);

	cvConvert (total_diff, output);

	for (int i = 0; i < 3; i++){
		cvReleaseMat (&(diffs[i]));
	}
	free (diffs);
	cvReleaseImage(&im_smoothed);
	cvReleaseImage(&im_diff);
	cvReleaseImage(&diff32);
	cvReleaseMat(&total_diff);

	//  cvReleaseMat (&total_diff);

	//return total_diff;
}
#endif