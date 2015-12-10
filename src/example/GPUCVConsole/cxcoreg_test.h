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
#ifndef __GPUCV_CONSOLE_CXCORE_TEST_H
#define __GPUCV_CONSOLE_CXCORE_TEST_H

#include <cxcore.h>
#include <highgui.h>
#include <highguig/highguig.h>
#include <cxcoreg/cxcoreg.h>

#ifdef _GPUCV_SUPPORT_NPP
#	include <gcvnpp/gcvnpp.h>
#endif

/** @defgroup CVGXCORE_TEST_GRP Test functions for cxcoreg.h operators
*	@ingroup CVGXCORE__GRP
*   @ingroup GPUCVCONSOLE_GRP
@{*/

#define TEST_GPUCV_CXCORE		1

void cxcore_test_print();

/** \brief Parse the current command string to look for all cxcoreg operators and call them.
*/
bool cxcoreg_processCommand(std::string & CurCmd, std::string & nextCmd);
/** \brief Execute all cxcoreg operators. It is called from the benchmarking loop.
*/
void cxcoreg_runAll(IplImage **src1, IplImage ** src2, IplImage ** mask);

//CVGXCORE_OPER_ARRAY_INIT_GRP
void runClone(IplImage * src1);
//CVGXCORE_OPER_ARRAY_ACCESS_ELEM_GRP
//CVGXCORE_OPER_COPY_FILL_GRP
void runCopy(IplImage * src1,IplImage * src2 = NULL );
void runSet(IplImage * src1, CvScalar* _scalar, IplImage * mask);
//CVGXCORE_OPER_TRANSFORM_PERMUT_GRP
void runSplit(IplImage * src1, GLuint channel);
void runMerge(IplImage * src1, GLuint channel);
//CVGXCORE_OPER_ARITHM_LOGIC_COMP_GRP
void runLut(IplImage * src1);//,IplImage * src2);
void runConvertScale(IplImage * src1, double scale, double shift);
//KEYTAGS: TUTO_TEST_OP_TAG__STP1_1__DECLARE_TEST_FUNCT
void runAdd(IplImage * src1,IplImage * src2, IplImage * mask=NULL, CvScalar * value2=NULL);
void runAddWeighted(IplImage * src1,double s1,IplImage * src2,double s2,double gamma);
void runSub(IplImage * src1,IplImage * src2, IplImage * mask=NULL, CvScalar * value2=NULL, CvScalar * _scale=NULL, bool reversed=false);
void runLogics(GPUCV_ARITHM_LOGIC_OPER _opertype, IplImage * src1,IplImage * src2, IplImage * mask=NULL, CvScalar * value2=NULL);
void runCmp(IplImage * src1,IplImage * src2,int op, double * value=NULL);
void runDFT(IplImage *src1,int flags,int zr);
void runAbsDiff(IplImage * src1 , IplImage * src2, CvScalar * _scalar = NULL);
void runAbs(IplImage * src1);
void runDiv(IplImage * src1,IplImage * src2, float factor);
void runMul(IplImage * src1,IplImage * src2, float factor);
void runMinMax(IplImage * src1,IplImage * src2, bool Min, double *value=NULL);
//CVGXCORE_OPER_STATS_GRP
void runSum(IplImage * src1);
void runAvg(IplImage * src1, IplImage * mask/*=NULL*/);
void runMinMaxLoc(IplImage * src1);
//CVGXCORE_OPER_LINEAR_ALGEBRA_GRP
void runScaleAdd(IplImage * src1,IplImage * src2, CvScalar * _scale);
void runGEMM(int size, double alpha=1., double beta=0.);
void runTranspose(IplImage * src1);
void runFlip(IplImage * src1, int Flip_mode);
//CVGXCORE_OPER_MATH_FCT_GRP
void runSqrt(IplImage * src1);
void runPow(IplImage * src, double factor);
//CVGXCORE_OPER_RANDOW_NUMBER_GRP
//CVGXCORE_OPER_DISCRETE_TRANS_GRP
//CVGXCORE_DRAWING_CURVES_SHAPES_GRP
//CVGXCORE_DRAWING_TEXT_GRP
//CVGXCORE_DRAWING_POINT_CONTOURS_GRP
void runLine(IplImage * src1);
void runRectangle(IplImage * src1);
void runCircle(IplImage * src1);
/**	@} */
/** Try a GLSL + CUDA sequence and compare result with full OpenCV sequence.
*/
void runCompat_GLSLxCUDA(IplImage * src1, IplImage * src2);

//Other operators that are not part of OpenCV.
void runCXCOREBench(IplImage * src1, IplImage * src2, IplImage * mask);
void runDeriche(IplImage * src1, float alpha);
void runLocalSum(IplImage *src1,int h,int w);
#endif
