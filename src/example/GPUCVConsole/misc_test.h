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
#ifndef __GPUCV_CONSOLE_MISC_TEST_H
#define __GPUCV_CONSOLE_MISC_TEST_H

#include "mainSampleTest.h"

bool misc_processCommand(std::string & CurCmd, std::string & nextCmd);
//void misc_runAll(IplImage **src1, IplImage ** src2, IplImage ** mask);


#if _GPUCV_DEPRECATED
void runGpuToCpu(IplImage * _src1);
void runCpuToGpu(IplImage * _src1);
#endif
bool RunIplImageTransferTest(IplImage * _src1, bool datatransfer);
bool runCloneTest(IplImage * _src1);

#endif
