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
/** \brief Contains enum descriptions of arithmetics and logics operators.
*	\author Yannick Allusse
*/
#ifndef __GPUCV_OPERATOR_ENUM_H
#define __GPUCV_OPERATOR_ENUM_H


/** \brief Contains enum descriptions of arithmetics and logics operators.
\note KEYTAGS: TUTO_CUDA_ARITHM__STP3__ADD_ENUM	
*/
enum GPUCV_ARITHM_LOGIC_OPER{
	OPER_ADD
	,OPER_SUB
	,OPER_SUBR
	,OPER_DIV
	,OPER_MUL
	,OPER_MIN
	,OPER_MAX	
	,OPER_AND
	,OPER_OR
	,OPER_XOR
	,OPER_NOT
	,OPER_ABS
	,OPER_ABSDIFF
	,OPER_SQUARE
	,OPER_RTSQUARE
	/* ... insert your new operators here */

	,GPUCV_ARITHM_LOGIC_OPER__LAST_VAL // must always be the last value.
};

#endif
