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



#ifndef __GPUCV_CORE_FPS_H
#define __GPUCV_CORE_FPS_H

#include <cstdio>
#include <stdlib.h>
#include <GPUCVHardware/ToolsGL.h>
#include <GPUCVCore/config.h>

namespace GCV{
/*
#ifdef _WINDOWS
#include<windows.h>
#endif
*/
/**
*	\brief simple class to show FPS and get time of execution
*	\author Olivier Nocent
*	\author Erwan Guehenneux
*	\author Jean-Philippe Farrugia
*
*/
#if _GPUCV_DEPRECATED
class FPS
{
public:
	FPS(double _x=0.0, double _y=0.0, double _r=1.0, double _g=1.0, double _b=1.0);
	void setPosition(double _x, double _y);
	void setColor(double _r, double _g, double _b);
	void update();
	void draw() const;

	void initTime();
	double getTime();

private:
	int    t0, t;
	int    frames;
	double x, y;
	double r, g, b;
	char   text[7];

#ifndef _WINDOWS
	timeval AbsoluteTime, NewTime, OldTime;
#else
	LARGE_INTEGER AbsoluteTime, NewTime, OldTime, TicksPerSecond;
#endif

};

FPS * GetFPS();
#endif
}//namespace GCV
#endif
