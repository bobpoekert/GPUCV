//CVG_LicenseBegin==============================================================
//
//	Copyright@ Institut TELECOM 2005
//		http://www.institut-telecom.fr/en_accueil.html
//	
//	This software is a GPU accelerated library for computer-vision. It 
//	supports an OPENCV-like extensible interface for easily porting OPENCV 
//	applications.
//
//
//	Contacts :
//				patrick.horain@it-sudparis.eu		
//				gpucv-developers@picoforge.int-evry.fr
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
#ifndef __GPUCV_SWITCH_H
#define __GPUCV_SWITCH_H


#include <GPUCVSwitch/Cl_FctSw.h>
#include <GPUCV/misc.h>
#include <GPUCVSwitch/Cl_FctSw_Mngr.h>
/** Initialize the switch mechanism, load the XML file containing DLL informations and plugins to load and initialize each plugins.
 \note Default loaded XML file is "./data/gcv_dlls.xml".
 \sa cvgInit(), cvgCudaInit().
 */
_GPUCV_SWITCH_EXPORT_C void cvgswInit(bool InitGLContext=true, bool isMultiThread=false);

/** Add manually a new plug-in libray to the switch.
 \param _lib -> Library name to load.
 \param _prefix -> plugin function prefix name if any.
 */
_GPUCV_SWITCH_EXPORT_C void cvgswAddLib(const std::string _lib, const std::string _prefix); 


/** Return the implementation ID of the implementation used in the last called of function _fctName.
\param _fctName -> fonction name to look for.
 */
_GPUCV_SWITCH_EXPORT_C int  cvgswGetLastCalledImpl(const std::string & _fctName);

/** Print to the console some informations about the functions called by the switch. Give the number of calls, min/avg/max time, etc...
\sa cvgswPrintFctStats().
 */
_GPUCV_SWITCH_EXPORT_C void cvgswPrintAllFctStats(void);

/** Print to the console some informations about the function _fctName. Give the number of calls, min/avg/max time, etc...
 \sa cvgswPrintAllFctStats().
 */
_GPUCV_SWITCH_EXPORT_C void cvgswPrintFctStats(const std::string _fctName);


/** Force global implementation to the given ID. If no function implementation is available for a function, the switch we look for other implementations available.
 */
_GPUCV_SWITCH_EXPORT_C void cvgswSetGlobalImplementation(GCV::BaseImplementation _impl);


/** Return global implementation ID.
 */
_GPUCV_SWITCH_EXPORT_C int  cvgswGetGlobalImplementation();
/*====================================*/

/** Set the last execution state of an operator, giving the execution parameters. It is used when an implementation give false results, so we can rise a flag and disable it for the given set of parameters.
\return True if fonction name has been found and the given parameter set exists, else false.
 */
_GPUCV_SWITCH_EXPORT_C bool cvgswSetFctLastExecutionState(const char* _pFctName, const char * _pParameterSet, SG_TRC::ExecState _State);

/** Register the profiling singleton into current DLL, cause singleton are not shared by default among DLLs.
 */
_GPUCV_SWITCH_EXPORT_C void cvg_switch_RegisterTracerSingletons(SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> * _pAppliTracer, SG_TRC::CL_TRACING_EVENT_LIST *_pEventList);

#endif
