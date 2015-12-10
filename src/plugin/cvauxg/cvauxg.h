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
#ifndef __GPUCV_CVAUXG_H 
#define __GPUCV_CVAUXG_H
#include <GPUCV/misc.h>

//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @ingroup CVAUXG_EIGEN_OBJ_GRP
*  @{
*/
#if _GPUCV_DEVELOP_BETA
/**
*	\brief The function calculates object projection to the eigen sub-space.
*	Purpose: The function calculates object projection to the eigen sub-space (restores
*             an object) using previously calculated eigen objects basis, mean (averaged)
*             object and decomposition coefficients of the restored object
*	\param nEigObjs -> number of eigen objects
*	\param eigInput -> pointer either to array of pointers to eigen objects
*                               or to read callback function (depending on ioFlags)
*	\param ioFlags -> input/output flags
*	\param userData -> pointer to the structure which contains all necessary
*                               data for the callback function
*	\param coeffs -> array of decomposition coefficients
*	\param avg -> averaged object
*	\param proj -> object projection (output data)
*   \note   See notes for cvCalcEigenObjects function
*/
_GPUCV_CVAUXG_EXPORT_C
/*CV_IMPL*/ void
cvgEigenProjection( void*     eigInput,
				   int       nEigObjs,
				   int       ioFlags,
				   void*     userData,
				   float*    coeffs, 
				   IplImage* avg,
				   IplImage* proj );
#endif// _GPUCV_DEVELOP_BETA
/** @}*///CVAUXG_EIGEN_OBJ_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________

#endif//_GPUCV_CVAUXG_H
