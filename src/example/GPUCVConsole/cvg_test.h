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


#ifndef __GPUCV_CONSOLE_CV_TEST_H
#define __GPUCV_CONSOLE_CV_TEST_H

#include <cv.h>
#include <highgui.h>
#include <highguig/highguig.h>


/** @defgroup CV_TEST_GRP Test functions for cvg.h operators
*	@ingroup CV__GRP
*   @ingroup GPUCVCONSOLE_GRP
@{*/

#define TEST_GPUCV_CV			1

void cv_test_print();

#if TEST_GPUCV_CV
bool cvg_processCommand(std::string & CurCmd, std::string & nextCmd);
void cvg_runAll(IplImage **src1, IplImage ** src2, IplImage ** mask);
//CVG_IMGPROC__GRAD_EDGE_CORNER_GRP
void runSobel(IplImage * src1, int xOrder=1, int yOrder=1, int aperture_size=3);
void runLaplace(IplImage * src1, int aperture_size=3);
void runCanny(IplImage * src1, int thresh1, int thresh2, int aperture_size=3);
//CVG_IMGPROC__SAMPLE_INTER_GEO_GRP
void runResize(IplImage * src1, int width, int height);
void runResizeFct(IplImage * src1, int width, int height, char * FctName);
//CVG_IMGPROC__MORPHO_GRP
void runDilate(IplImage * src1,int pos, int iter=1 );
void runErode(IplImage * src1,int pos, int iter=1);
void runMorpho(IplImage * src1,int pos, int iter=1);
//CVG_IMGPROC__FILTERS_CLR_CONV_GRP
void runCvtColor(IplImage * src1, int type);
void runThreshold(IplImage * src1, double threshold, double max_value, int type);
void runSmooth(IplImage * src1, int smoothtype=CV_GAUSSIAN, int param1=3, int param2=0, double param3=0, double param4=0);
void runIntegral(IplImage * src1);
void runLocalSum(IplImage * src1,int height,int width);
void runFaceDetect(IplImage * src1);
//CVG_IMGPROC__PYRAMIDS_GRP
//CVG_IMGPROC__IMGSEGM_CC_CR_GRP
//CVG_IMGPROC__IMG_CONT_MOMENT_GRP
//CVG_IMGPROC__SPE_IMG_TRANS_GRP
void runDist(IplImage * src1);
//CVG_IMGPROC__HISTOGRAM_GRP
void runHisto(IplImage *src1, int _bins);
//CVG_IMGPROC__MATCHING_GRP

//CVG_IMGPROC__CUSTOM_FILTER_GRP
void runDeriche(IplImage * src1, float alpha);

#endif
/** @}*/


#endif//__GPUCV_CONSOLE_CV_TEST_H
