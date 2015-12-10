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
#ifndef __GPUCV_SWITCH_CONFIG_H
#define __GPUCV_SWITCH_CONFIG_H

#include <GPUCVHardware/include.h>



#ifdef _WINDOWS
#	ifdef _GPUCV_SWITCH_DLL
#		define _GPUCV_SWITCH_EXPORT		    __declspec(dllexport)
#		define _GPUCV_SWITCH_EXPORT_C		extern "C" _GPUCV_SWITCH_EXPORT
#	else
#		ifdef __cplusplus
#			define _GPUCV_SWITCH_EXPORT			__declspec(dllimport)
#			define _GPUCV_SWITCH_EXPORT_C		extern "C" _GPUCV_SWITCH_EXPORT
#		else
#			define _GPUCV_SWITCH_EXPORT			__declspec(dllimport)
#			define _GPUCV_SWITCH_EXPORT_C		 _GPUCV_SWITCH_EXPORT
#		endif
#	endif
#else
#	define _GPUCV_SWITCH_EXPORT
#	ifdef __cplusplus
#		define _GPUCV_SWITCH_EXPORT_C extern "C"
#	else
#		define _GPUCV_SWITCH_EXPORT_C
#	endif
#endif

#if _GPUCV_DEPRECATED
	#if __GPUTEXTURE_PROFILE//????
		#define __GPUSWITCH_PROFILE_ADD_FCT(FCTY_NAME)\
			static CL_FUNCT_TRC<CL_TimerVal>	*ThisFct=\
				CL_Profiler::GetTimeTracer().AddFunct(FCTY_NAME);

		#define __GPUSWITCH_PROFILE_START_TRACER()\
				CL_TRACE_BASE_PARAMS * TempParams = new CL_TRACE_BASE_PARAMS();\
				CL_TEMP_TRACER<CL_TimerVal> Tracer(TrcFct, TempParams);
	#else
		#define __GPUSWITCH_PROFILE_ADD_FCT(FCTY_NAME)
		#define __GPUSWITCH_PROFILE_START_TRACER()
	#endif


	#ifdef _MSC_VER
		#define _GPUCV_SWITCH_INLINE inline
	#else
		#define _GPUCV_SWITCH_INLINE
	#endif
#endif


#endif //__SWITCH_CONFIG_H__
