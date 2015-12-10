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
#ifndef __GPUCV_CXCOREGCU_CONFIG_H
#define __GPUCV_CXCOREGCU_CONFIG_H

#include	<GPUCVCuda/config.h>

#ifdef _WINDOWS
	#ifdef	_GPUCV_CXCOREGCU_DLL
		#define		_GPUCV_CXCOREGCU_EXPORT		    __declspec(dllexport)
		#define		_GPUCV_CXCOREGCU_EXPORT_C		extern "C" _GPUCV_CXCOREGCU_EXPORT//
	#else
		#ifdef __cplusplus
			#define		_GPUCV_CXCOREGCU_EXPORT		__declspec(dllimport)
			#define		_GPUCV_CXCOREGCU_EXPORT_C	extern "C" _GPUCV_CXCOREGCU_EXPORT
		#else
			#define		_GPUCV_CXCOREGCU_EXPORT		__declspec(dllimport)
			#define		_GPUCV_CXCOREGCU_EXPORT_C	_GPUCV_CXCOREGCU_EXPORT
		#endif
	#endif
	#define		_GPUCV_CXCOREGCU_EXPORT_CU	 extern "C" __declspec(dllexport)//_GPUCV_CXCOREGCU_EXPORT_C
#else
	#define		_GPUCV_CXCOREGCU_EXPORT
	#define		_GPUCV_CXCOREGCU_EXPORT_C
	#define		_GPUCV_CXCOREGCU_EXPORT_CU	 extern "C"
#endif


/** \bug On LINUX 64 bit platform, the following error message might happen:"/usr/lib/gcc/x86_64-redhat-linux/4.1.2/include/mmintrin.h(49): error:identifier "__builtin_ia32_emms" is undefined". I might be solved by disactivating SSE or MMX instruction for some part of the code. GpuCVCuda plugins do not use MMX or SSE manually, so it might not impact on performances.
    \sa http://forums.nvidia.com/lofiversion/index.php?t73134.html
*/
#if defined _AMD64 ||defined _WIN64 || __X86_64_
#undef __SSE2__
#undef _MM_SHUFFLE2
#undef CV_ICC
#endif



#endif //__GPUCV_CXCOREGCU_CONFIG_H
