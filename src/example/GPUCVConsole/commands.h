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


#ifndef __GPUCV_CONSOLE_COMMAND_H
#define __GPUCV_CONSOLE_COMMAND_H

#include "mainSampleTest.h"

/** @ingroup GPUCVCONSOLE_GRP_COMMANDS
@{*/

/** \note Called by 'enable %ARG' or 'disable %ARG'. */
bool EnableDisableSettings(std::string  _arg, bool _flag);
/** \note Called by 'imgformat %IMG_NAME %CHANNELS %DEPTH %OPT_SCALE'. */
bool changeImageFormat(std::string & Command);
bool changeImageFormat2(IplImage ** Img, int channels, int depth, float scale);
#if _GPUCV_DEPRECATED
	void localS(std::string & Command);
#endif
/** \note Called by 'benchmode'. */
void SwitchToBenchMode();
/** \note Called by 'gpucvstats'. */
void ShowGpuCVStats();
/** \note Called by 'showsrc'. */
void ShowSrcImgs(IplImage * Src1, IplImage * Src2, IplImage * mask);
/** \note Called by 'runbench %LOOP_NBR'. */
void runBench(IplImage **src1, IplImage ** src2, IplImage ** mask);
/** Function that parse the command strin and call the corresponding command.*/
int ProcessCommand(std::string & Cmd);
/** Resize all input images used for benchmarks
\note Called by 'resizeimg %width %height'.
*/
bool resizeImage(IplImage ** Img, int width, int height, int interpolation=CV_INTER_LINEAR);
/** \note Called by 'loadbench %file'. */
void LoadBench(const char * filename);
/** \note Called by 'savebench %file'. */
void SaveBench(const char * filename);

/** \note Called by 'benchreport %file'. */
void BenchReport(const char * _filename);
/** @}*/

bool ParseCommands(const int _argc, char ** argv);
float ControlResultsImages(CvArr * srcRef, CvArr * srcTest, const char * FctName, const char * Params);
float ControlResultsImages(CvArr * srcRef, CvArr * srcTest, const std::string & FctName, const std::string & Params);

#endif//__GPUCV_CONSOLE_COMMAND_H
