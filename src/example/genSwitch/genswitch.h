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

/** @defgroup GPUCV_GENSWITCH_GRP genSwitch
	@ingroup GPUCV_EXAMPLE_LIST_GRP
	@{
This application is used to create switch wrapper functions for all GpuCV/OpenCV operators.
	\author Yannick Allusse
	\version GpuCV 1.0 rev 560
*/
#include <GPUCVSwitch/switch.h>	
#include "SugoiTools/MultiPlat.h"

//switch functions
void cleanHeader(std::string Infilename, std::string IN_FILE_PATH, std::string Outfilename, std::string OUT_FILE_PATH);
void cleanHeaders();
void ParseHeaderFiles();
void InitPaths();
//==================

/** @}*/ //GPUCV_GENSWITCH_GRP






