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
#ifndef __GPUCV_SWITCH__MACRO_H
#define __GPUCV_SWITCH__MACRO_H
/* 
 * \brief Run time switch upon all implementation available, possibility to force implementation locally or globally. 
 * \author Allusse Yannick , C S Priyadarshan

 Dynamically switch enclosed inside each operator calls faster implementation among OpenCV, GpuCV-GLSL and GpuCV-CUDA to run the operator.
*/
#include <GPUCVSwitch/Cl_FctSw.h>

using namespace std;
using namespace GCV;

#define _GPUCV_PROFILE_DARSHAN 1


//! Log switch information to the console
#define GPUCV_SWITCH_LOG(msg)\
	{if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_SWITCH_LOG))\
		{SG_NOTICE_PREFIX_LOG("[SWCH]", msg)}\
	}


#define SWITCH_START_OPR(_Img)\
	static CL_FctSw * LocalFctSw = NULL;\
	static bool LocalInitDone = false;\
	try\
	{\
		if(!DllManager::GetSingleton())cvgswInit();\
		GPUCV_SWITCH_LOG("");\
		GPUCV_SWITCH_LOG("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");\
		GPUCV_SWITCH_LOG("Start operator:" << GPUCV_GET_FCT_NAME());\
		GPUCV_SWITCH_LOG("================================================================");\
		SetThread();\
		LogIndentIncrease();\
		if(LocalInitDone==false)\
		{\
			LocalFctSw = DllManager::GetSingleton()->GetFunctionObj(GPUCV_GET_FCT_NAME());\
			LocalInitDone=true;\
			SG_Assert(LocalFctSw, "No switching function found!");\
		}\
		if(SrcARR)LocalFctSw->SetInputArray(SrcARR, sizeof(SrcARR)/sizeof(CvArr*));\
		if(DstARR)LocalFctSw->SetOutputArray(DstARR, sizeof(DstARR)/sizeof(CvArr*));\
		SG_TRC::CL_TRACE_BASE_PARAMS * paramsobj = LocalFctSw->GetParamsObj(_Img);\
		LogIndentIncrease();

//SetSwitchOperators(SwitchImplementations, sizeof(SwitchImplementations)/sizeof(switchFctStruct));\


					
/*!
	*\brief Checks the profile mode flag and acts accordingly to run or profile the operator.
	*\param ARGS => all the arguments in the present operator.
	*\param CVGType => typedef.
	*\note decides if still the operator is to be profiled or not and runs the operator accordingly.
	*\note Execution status is set to Failed by default, if no exception is rised, it will be set to SUCCESS again in SWITCH_STOP_OPR macro.
	*\author : CS Priyadarshan 
	*/
//FctName=GetStrImplemtation(impltocall->m_Implementation)+"::"+FctName;\

#define RUNOP(ARGS, CVGType, RETURN_OPT)\
	GPUCV_SWITCH_LOG("RUNOP");\
				switchFctStruct * impltocall = NULL;\
				if(LocalFctSw==NULL)GPUCV_ERROR("LocalFctSw is NULL");\
				if(DstARR)		impltocall=LocalFctSw->GetImplIDToCall(DstARR[0],	paramsobj);\
				else			impltocall=LocalFctSw->GetImplIDToCall(NULL,		paramsobj);\
				LogIndentDecrease();\
				SG_Assert(impltocall, "No ipmltocall object of switchFctStruct");\
				SG_Assert(impltocall->m_ImplPtr, "no ImplPtr of switchFctStruct");\
				CVGType MyFct = (CVGType)impltocall->m_ImplPtr;\
				SG_Assert(MyFct, "No implementation found to be called");\
				if (LocalFctSw->GetProfileMode())\
				{\
						FctName=FctName;\
						{\
							if(paramsobj)paramsobj->AddParamAtStart("type", GetStrImplemtation(impltocall->m_Implementation));\
							DARSHAN_GPUCV_PROFILE_IMPL(GPUCV_GET_FCT_NAME(), impltocall, paramsobj);\
							GPUCV_SWITCH_LOG("\t>Profiling "<< GetStrImplemtation(impltocall->m_Implementation) << " implementation.");\
							if(DstARR && DstARR[0])cvgFlush(DstARR[0]);\
							RETURN_OPT MyFct ARGS;\
						}\
					impltocall->m_FctTracer->Process();\
				}\
				else\
				{\
					GPUCV_SWITCH_LOG("\t>Processing "<< GetStrImplemtation(impltocall->m_Implementation)<< " implementation.");\
					RETURN_OPT MyFct ARGS;\
				}



/*!
	*\brief Catches any exceptions and resumes the operation in the opencv mode (default).
	*\author : CS Priyadarshan
	*\note Execution status Success is set here.
	*/

#define SWITCH_STOP_OPR()\
				LogIndentDecrease();\
				UnsetThread();\
				GPUCV_SWITCH_LOG("================================================================");\
				GPUCV_SWITCH_LOG("Stop operator:" << GPUCV_GET_FCT_NAME());\
				GPUCV_SWITCH_LOG("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");\
				GPUCV_SWITCH_LOG("");\
				if(LocalFctSw->GetCurTracerRecord())LocalFctSw->GetCurTracerRecord()->SetExecutionState(SG_TRC::EXEC_SUCCESS);\
			}\
			catch(GCV::Exception &e)\
			{\
			if(LocalFctSw->GetCurTracerRecord())LocalFctSw->GetCurTracerRecord()->SetExecutionState(SG_TRC::EXEC_FAILED_COMPAT_ISSUE);\
				if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_PLUG_THROW_EXCEPTIONS))\
				{\
					LogIndentDecrease();\
					throw;\
				}\
				else\
				{\
					GPUCV_ERROR("Compatibility error catched");\
					GPUCV_ERROR("Operator '" << GPUCV_GET_FCT_NAME() << "' is not compatible with the current parameters.");\
					LogIndentIncrease();\
					GPUCV_ERROR("========Catched in:");\
						GPUCV_ERROR("Line : "<< __LINE__);\
						GPUCV_ERROR("File : "<< __FILE__);\
						LogIndentDecrease();\
					GPUCV_ERROR("========Rised in:");\
						LogIndentIncrease();\
						GPUCV_ERROR("Line : "<< e.GetLine());\
						GPUCV_ERROR("File : "<< e.GetFile());\
						GPUCV_ERROR("Msg: " << e.GetOriginalMsg());\
						LogIndentDecrease();\
					GPUCV_ERROR("=================");\
					LogIndentDecrease();\
				}\
			}\
			catch(SGE::CAssertException &e)\
			{\
			if(LocalFctSw->GetCurTracerRecord())LocalFctSw->GetCurTracerRecord()->SetExecutionState(SG_TRC::EXEC_FAILED_CRITICAL_ERROR);\
				UnsetThread();\
				GPUCV_ERROR("=================== Exception catched Start =================")\
				GPUCV_ERROR("");\
				GPUCV_ERROR("catched in function : "<< GPUCV_GET_FCT_NAME());\
				GPUCV_ERROR("line : "<< __LINE__);\
				GPUCV_ERROR("file : "<< __FILE__);\
				GPUCV_ERROR("_____________________________________________________________");\
				GPUCV_ERROR("Exception description:");\
				GPUCV_ERROR("_____________________________________________________________");\
				GPUCV_ERROR(e.what());\
				if(GetGpuCVSettings()->GetLastExceptionObj())\
				{\
				GPUCV_ERROR("Object description:");\
				GPUCV_ERROR(*GetGpuCVSettings()->GetLastExceptionObj());\
				GPUCV_ERROR(GetGpuCVSettings()->GetLastExceptionObj()->LogException());\
				}\
				GPUCV_ERROR("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");\
				GPUCV_ERROR("Printing Texture manager data=============================");	\
				GetTextureManager()->PrintAllObjects();										\
				GPUCV_ERROR("==================================================");			\
				GPUCV_ERROR("_____________________________________________________________");\
				GPUCV_ERROR("Resolving exception:");\
				if(_GPUCV_DEBUG_MODE)Beep(3000, 2);\
				GPUCV_ERROR("A GpuCV operator failed!");\
				GPUCV_ERROR("_____________________________________________________________");\
				GPUCV_ERROR("=================== Exception catched End =================");\
				GPUCV_ERROR("<SWITCH_STOP_OPR: exception caught '" << GPUCV_GET_FCT_NAME() <<"'");\
			}
			


/*!
	* \brief Profiling and tracing the function. 
	*\param NAME => Function name.
	*\param CUR_FCT_OBJ => current implementation class object.
	*\param PARAMS_OBJ => params object.
	*\param RSLT_IMG => result image.
	*\param IMPL_FLAG => Implementation flag(enum).
	*\note  Traces the operator run in different implementations and profile it.
	*\author : Cindula Saipriyadarshan 
	*/
#if _GPUCV_PROFILE_DARSHAN
	#define DARSHAN_GPUCV_PROFILE_IMPL(NAME, CUR_FCT_OBJ, PARAMS_OBJ)\
			SG_Assert(CUR_FCT_OBJ, "No function ptr");\
			SG_TRC::CL_TEMP_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> Tracer (NAME,PARAMS_OBJ);\
			Tracer.SetOpenGL(CUR_FCT_OBJ->m_UseGpu);
//			PARAMS_OBJ->AddChar("type", GetStrImplemtation((GpuCVSettings::Implementation)impltocall->m_ImplID).data());

#if _GPUCV_DEPRECATED

	#define DARSHAN_PROFILE_CURRENT_FCT(NAME, RSLT_IMG, IMPL_FLAG)\
			SG_TRC::CL_FUNCT_TRC<SG_TRC::SG_TRC_Default_Trc_Type> *CurFct##__LINE__ = NULL;\
			SG_TRC::CL_TRACE_BASE_PARAMS *_PROFILE_PARAMS = NULL;\
			if(1)\
			{\
				CurFct##__LINE__=\
				MainTagTracer->AddFunct(NAME);\
				if(RSLT_IMG)\
					_PROFILE_PARAMS = CL_FctSw::GetParamsObj(RSLT_IMG);\
				else\
					_PROFILE_PARAMS = new SG_TRC::CL_TRACE_BASE_PARAMS();\
				_PROFILE_PARAMS->AddChar("type", GetStrImplemtation((GpuCVSettings::Implementation)IMPL_FLAG).data());\
				if(true)cvgFlush(RSLT_IMG);\
			}\
			SG_TRC::CL_TEMP_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> Tracer (NAME, _PROFILE_PARAMS);\
			Tracer.SetOpenGL(true);
#endif
#else 
	#define DARSHAN_GPUCV_PROFILE_IMPL(NAME, CUR_FCT_OBJ, RSLT_IMG)
#endif

#endif
