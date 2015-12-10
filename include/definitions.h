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
\file definitions.h
\brief header file containg defines and preprocessors definitions

\author Yannick Allusse, Jean-Philippe Farrugia, Erwan Guehenneux
*/
#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <main_dox_grp.h>//use to generate doxygen doc.

/** @ingroup GPUCV_MACRO_GRP
*	@{
*/

#if defined (_WIN32)|| defined (WIN32) || defined (_WIN64) || defined (WIN64)
	#ifndef _WINDOWS
		#define _WINDOWS
	#endif
#elif defined(APPLE) || defined(__APPLE__) || defined(MACOS)
	#ifndef _MACOS
		#define _MACOS
	#endif
#elif defined(UNIX) || defined(__linux__) || defined(__LINUX__)
	#ifndef _LINUX
		#define _LINUX
	#endif
#endif

/** Define the internal RELEASE/DEBUG mode
*	<br>Set _GPUCV_DEBUG_MODE to 0 to compile the GPUCV library in release mode (best performances), it disables all debugging functionnalities.
*	<br>Set _GPUCV_DEBUG_MODE to 1 to compile the GPUCV library in debug mode (lower performances but more debugging outputs).
*/
#if _DEBUG
#define _GPUCV_DEBUG_MODE				1
#else
#define _GPUCV_DEBUG_MODE				0
#endif

#define _GPUCV_DEVELOP				1	//(1 & _GPUCV_DEBUG_MODE)	//!< _GPUCV_DEVELOP should be 1 to compile operators still in development. Better leave it to 0!!
#define _GPUCV_DEVELOP_BETA			(0 & _GPUCV_DEVELOP)	//!< _GPUCV_DEVELOP_BETA should be 1 to compile beta operators still in development. Better leave it to 0!!
#define _GPUCV_SUPPORT_GPU_PLUGIN	(0 & _GPUCV_DEBUG_MODE)	//!< Still in development.
//==============================================
#define _GPUCV_DEBUG_FBO			(0 &_GPUCV_DEBUG_MODE)
#define _GPUCV_DEBUG_PBUFFER		(0 &_GPUCV_DEBUG_MODE)
#define _GPUCV_DEBUG_MOMORY_LEAK	(0 & _SG_TLS_MEMORY_MANAGER)
#define _GPUCV_DEPRECATED			0
//==================================
//! Enable using GLUT to init OpenGL context and draw images
#ifdef __LINUX
#	define _GPUCV_GL_USE_GLUT			0	
#else
#	define _GPUCV_GL_USE_GLUT			1
#endif


#ifdef _WINDOWS
//#ifdef _VCC2005
#pragma warning(once : 4996) //some C++ functions were declared deprecated, show only once by function.
//#endif
#endif
//==============================================



//! Force the library to use GL_TEXTURE_2D
#define _GPUCV_TEXTURE_FORCE_TEX_NPO2	0
//! Force the library to use GL_TEXTURE_RECTANGLE_ARB
#define _GPUCV_TEXTURE_FORCE_TEX_RECT	0
#define _GPUCV_SHADER_LOAD_FORCE		0
//! Enable use of OPENGL MIPMAPPING
#define _GPUCV_GL_USE_MIPMAPING 1
/** @}*/ //GPUCV_MACRO_GRP

/** @ingroup GPUCV_MACRO_SHADER_GRP
*  @{*/
#define _GPUCV_SHADER_MAX_PARAM_NB		256
#define _GPUCV_SHADER_MAX_NB			256
#define _GPUCV_TEXTURE_MAX_SIZE_X		8192
#define _GPUCV_TEXTURE_MAX_SIZE_Y		_GPUCV_TEXTURE_MAX_SIZE_X
#define _GPUCV_FRAMEBUFFER_DFLT_FORMAT	GL_RGBA16_EXT//still in test...
/** @}*/ //GPUCV_MACRO_SHADER_GRP
//---------------------------------

#ifdef _WINDOWS
#define _GPUCV_NOP NOP_FUNCTION
#else
#define _GPUCV_NOP
#endif



//	BENCHMARKS DEFINES =========================================================
/** @ingroup GPUCV_MACRO_BENCH_GRP
*  This groupe describes the benchmarking of the library and the applications using some included class.
*	\warning Enabling profiling settings inside the library with decrease operators performances. These functionalities will be disabled when library is in release mode.
*	\sa _GPUCV_DEBUG_MODE
*  @{*/

/** \var _GPUCV_PROFILE
*	enable/disable general profiling.
*/
/** \var _GPUCV_PROFILE_CLASS
*	enable/disable class profiling.
*/
/** \var _GPUCV_PROFILE_TOCONSOLE
*	enable/disable output profiling result to console.
*/
#if _DEBUG
#define _GPUCV_PROFILE				1//&_GPUCV_DEBUG_MODE
#define _GPUCV_PROFILE_CLASS		1//&_GPUCV_DEBUG_MODE
#define _GPUCV_PROFILE_TOCONSOLE	0
#else
#define _GPUCV_PROFILE				1//&_GPUCV_DEBUG_MODE
#define _GPUCV_PROFILE_CLASS		1//&_GPUCV_PROFILE
#define _GPUCV_PROFILE_TOCONSOLE	0
#endif

/**
*	enable/disable profiling of OpenGL calls.
* <br>Set it to 1 or 0.
* Benchmarked cvg functions using OpenGL are:
* <ul><li>cvgGLReadPixels()</li><li>TexCreate()</li><li>InitGLView()</li><li>drawQuad()</li></ul>
* Benchmarked OpenGL functions using the _BENCH_GL macro are:
* <ul>
<li>glCallList()</li>
<li>glFlush()</li>
<li>glFinish()</li>
<li>glReadPixels()</li>
<li>glBeginQueryARB()</li>
<li>...</li>
</ul>
*/
#define _GPUCV_PROFILE_GL 0//(1*_GPUCV_PROFILE)

#if _GPUCV_DEPRECATED
/**
*	enable/disable profiling of all the image transfer between CPU and GPU.
*	<br>Set it to 1 or 0.
*/
#define _GPUCV_PROFILE_IMG_LOADING 0//1 & _GPUCV_PROFILE

//! _GPUCV_PROFILE_FILTER [MORE_HERE]
#define _GPUCV_PROFILE_FILTER	0//1 & _GPUCV_PROFILE

//! _GPUCV_PROFILE_FBO [MORE_HERE]
#define _GPUCV_PROFILE_FBO	0//1 & _GPUCV_PROFILE
#endif

#if _GPUCV_PROFILE || _GPUCV_DEBUG_MODE
#define GPUCV_FUNCNAME(FCT_NAME) std::string FctName=FCT_NAME;
#define GPUCV_FUNCNAME_STATIC(FCT_NAME)static const std::string FctName=FCT_NAME;
#define GPUCV_FUNCNAME_TEMPLATE_STR(TPL, NAME)\
	std::string FctName=NAME;\
	FctName+= "<";\
	FctName+= TPL;\
	FctName+= ">";
#define GPUCV_FUNCNAME_TEMPLATE(TPL, NAME)GPUCV_FUNCNAME_TEMPLATE_STR(typeid(TPL).name(),NAME)
#define GPUCV_GET_FCT_NAME()FctName
#else
#define GPUCV_FUNCNAME(FCT_NAME)
#define GPUCV_FUNCNAME_STATIC(FCT_NAME)
#define GPUCV_FUNCNAME_TEMPLATE_STR(TPL, NAME)
#define GPUCV_FUNCNAME_TEMPLATE(TPL, NAME)
#define GPUCV_GET_FCT_NAME()std::string("?")
#endif
/** @} */ //GPUCV_MACRO_BENCH_GRP


//Benchmarking results using SVG
/** @defgroup SVG_GRP Generating SVG and PNG benchmarks files.
@ingroup GPUCV_MACRO_BENCH_GRP
*  This groupe describes the generation of SVG and PNG files.
*  @{*/
//! _GPUCV_SVG_CREATE_FILES enable/disable the generation of SVG files.
//! <br>Set it to 1 or 0.
#define _GPUCV_SVG_CREATE_FILES 1

//! _GPUCV_SVG_CONVERT_TO_PNG enable/disable the generation of PNG files from SVG files
//! <br>Set it to 1 or 0.
//! <br>Default is set to _GPUCV_SVG_CREATE_FILES.
#define _GPUCV_SVG_CONVERT_TO_PNG _GPUCV_SVG_CREATE_FILES

//! _GPUCV_SVG_BATIK_RASTERIZER_PATH defines the location of BATIK rasterizer file. Set it to your own BATIK setup directory.
#define _GPUCV_SVG_BATIK_RASTERIZER_PATH  "../resources/bin/batik-1.6/batik-rasterizer.jar"

//! _GPUCV_SVG_EXPORT_PATH defines the export directory for SVG and PNG files.
#define _GPUCV_SVG_EXPORT_PATH  "svg/"

//! _GPUCV_SVG_BATIK_RASTERIZER_SIZE defines the output size of PNG files.
//! <br>Format is "-w width -h heiht" with with and height in pixels.
#define _GPUCV_SVG_BATIK_RASTERIZER_SIZE "-w 450 -h 320 "
/** @} */ // end of SVG_GRP
//==============================================================================
/** @ingroup GPUCV_MACRO_LOGGING_GRP.
@{
\name Global debugging
*/
/** \brief Output debugging informations to the main target(file/console) when option GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG is true.
*	Output format is:"[DBG] %INDENT_STRING %MSG\n"
*	\sa GPUCV_CLASS_DEBUG
*/
#if _GPUCV_DEBUG_MODE //SG_LOG_MESSAGE("Warning", msg)
#define GPUCV_DEBUG(msg)\
	{if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG))\
		{SG_NOTICE_PREFIX_LOG("[DBG]", msg)}\
	}
#else
#define GPUCV_DEBUG(msg)
#endif

#define GPUCV_LOG(FLAG, TYPE,msg)\
	{if(GET_GPUCV_OPTION(FLAG))\
		{			SG_NOTICE_PREFIX_LOG("[" << TYPE << "DBG]", msg)}\
	}

/** \brief Output debugging informations to the main target(file/console) when option GpuCVSettings::GPUCV_SETTINGS_GLOBAL_WARNING is true.
*	Output format is:"[WRNG] %INDENT_STRING %MSG\n"
*	\sa GPUCV_CLASS_WARNING
*/
#define GPUCV_WARNING(msg)\
				{if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_WARNING))\
					{SG_NOTICE_PREFIX_LOG("[WRNG]", yellow << msg  << white)}\
				}
/** \brief Output debugging informations to the main target(file/console) when option GpuCVSettings::GPUCV_SETTINGS_GLOBAL_ERROR is true.
*	Output format is:"[ERR] %INDENT_STRING %MSG\n"
*	\sa GPUCV_CLASS_ERROR
*/
#define GPUCV_ERROR(msg)\
			{if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_ERROR))\
				{SG_NOTICE_PREFIX_LOG("[ERR]", red << msg << white)}\
			}
/** \brief Output debugging informations to the main target(file/console) when option GpuCVSettings::GPUCV_SETTINGS_GLOBAL_WARNING is true.
*	Output format is:"[PRF-WRNG] %INDENT_STRING %MSG\n"
*	\sa GPUCV_CLASS_WARNING
*/
#define GPUCV_PERF_WARNING(msg)\
			{if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_NOTICE))\
				{SG_NOTICE_PREFIX_LOG("[PRF-WRNG]", msg)}\
			}
/** \brief Output debugging informations to the main target(file/console) when option GpuCVSettings::GPUCV_SETTINGS_GLOBAL_NOTICE is true.
*	Output format is:"%INDENT_STRING %MSG\n"
*	\sa GPUCV_CLASS_NOTICE
*/
#define GPUCV_NOTICE(msg)\
			{if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_NOTICE))\
				{SG_NOTICE_LOG(msg)}\
			}



/** @}*/ //GPUCV_MACRO_LOGGING_GRP
#endif
