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
#include <GPUCVCore/fps.h>
#include <time.h>


#ifndef _WINDOWS
#include <sys/time.h>
#endif


namespace GCV{

#if _GPUCV_DEPRECATED

FPS::FPS(double _x, double _y, double _r, double _g, double _b)
:t0(0), t(0), frames(0), x(_x), y(_y), r(_r), g(_g), b(_b) {}

void FPS::setPosition(double _x, double _y)
{
	x = _x; y = _y;
}

void FPS::setColor(double _r, double _g, double _b)
{
	r = _r; g = _g; b = _b;
}

void FPS::update()
{
#if _GPUCV_GL_USE_GLUT
	frames++;
	t=glutGet(GLUT_ELAPSED_TIME);
	if ((t-t0)>1000)
	{
		sprintf(text, "%03d FPS",(int)( frames*1000.0f/(t-t0) ));
		t0=t;
		frames=0;
	}
#endif
}

void FPS::draw() const
{
#if _GPUCV_GL_USE_GLUT
	glLoadIdentity();
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_TEXTURE_2D);
	glColor4f(r, g, b, 1);
	glRasterPos2f(x, y);

	for(int i=0;i<8;i++)
		glutBitmapCharacter(GLUT_BITMAP_8_BY_13,*(text+i));

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_DEPTH_TEST);
#endif
}

void FPS::initTime()
{
#ifndef _WINDOWS
	gettimeofday(&AbsoluteTime,NULL);
	OldTime = AbsoluteTime;
#else
	QueryPerformanceFrequency(&TicksPerSecond);
	QueryPerformanceCounter(&AbsoluteTime);
	OldTime = AbsoluteTime;
#endif
}

double FPS::getTime()
{
#ifndef _WINDOWS
	gettimeofday(&NewTime,NULL);
	return ( NewTime.tv_usec - AbsoluteTime.tv_usec );// + NewTime.tv_sec - AbsoluteTime.tv_sec;
#else
	QueryPerformanceCounter(&NewTime);
	return ( (double) NewTime.QuadPart - (double) AbsoluteTime.QuadPart ) / (double) TicksPerSecond.QuadPart * 1000.;
#endif
}

FPS * GetFPS()
{
	static FPS FPSSingleton;
	return &FPSSingleton;
}
#endif

}//namespace GCV

