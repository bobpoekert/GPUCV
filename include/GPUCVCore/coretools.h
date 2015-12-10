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
/**
header file containg definitions of some tools to optimize coding

Yannick Allusse
*/
#ifndef __GPUCV_CORE_TOOLS_H
#define __GPUCV_CORE_TOOLS_H
#define MON_PI 3.141592653
#include "GPUCVCore/config.h"

namespace GCV{
/*!
*	\brief Initialize GpuCV framework, test extensions, and create or use a GL context.
*	\param InitGLContext -> define if library should create its own GL context or use an existing one
*	\param isMultiThread -> allow to use GPUCV library on multi-thread programs (in development)
*	\return int -> status(-1:failed, 0:already initialized, 1:init done)
*	\sa GpuCVTerminate(), cvgTerminate(), cvgInit()
*/
_GPUCV_CORE_EXPORT
int  GpuCVInit(bool InitGLContext=true, bool isMultiThread=false);

/*!
*	\brief Close GpuCV framework, release manager and save benchmarks.
*	\sa GpuCVInit(), cvgTerminate(), cvgInit()
*/
_GPUCV_CORE_EXPORT
void GpuCVTerminate();

/*==============================================================================

OPENGL TOOLS

==============================================================================*/

//		_GPUCV_CORE_EXPORT
//			void CreateViewerWindow(std::string _name, unsigned int _width = 512, unsigned int _height = 512);

_GPUCV_CORE_EXPORT
void ViewerWindowDisplay();
_GPUCV_CORE_EXPORT
void ViewerWindowIdle();
_GPUCV_CORE_EXPORT
void ViewerWindowKeyboard( unsigned char key, int x, int y);
_GPUCV_CORE_EXPORT
void ViewerWindowReshape(int w, int h);

/*==============================================================================*/

#if _GPUCV_SHADER_LOAD_FORCE
/*!
*	\brief parse default directories for pre-loading shader.
*/
_GPUCV_CORE_EXPORT
int load_default_shaders();
#endif
}//namespace GCV
#endif//TOOLS_H
