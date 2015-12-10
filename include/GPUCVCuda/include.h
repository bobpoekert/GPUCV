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



/** \brief Includes all header files required by any application that use GpuCV-CUDA plugin.
*	\author Yannick Allusse
*/
#ifndef __GPUCV_CUDA_INCLUDE_H
#define __GPUCV_CUDA_INCLUDE_H

#include <GPUCVCuda/config.h>
#include <GPUCVCuda/cuda_misc.h>


#if 0
//define some internal class to avoid using the header files
namespace GCV{
class _GPUCV_CUDA_EXPORT DataDsc_CUDA_Buffer;
class _GPUCV_CUDA_EXPORT DataDsc_CUDA_Array;
class _GPUCV_CUDA_EXPORT DataDsc_CUDA_Base;
}//namespace GCV
#else
#include <GPUCVCuda/DataDsc_CUDA_Buffer.h>
#endif

#endif//__GPUCV_CUDA_INCLUDE_H
