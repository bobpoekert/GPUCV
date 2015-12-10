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
#ifndef __GPUCV_CONFIG_H
#define __GPUCV_CONFIG_H

#ifdef __cplusplus
#	include <GPUCVCore/include.h>
#	include <GPUCVHardware/exception.h>
#endif

#define _GPUCV_SUPPORT_CVMAT 1


#ifdef _WINDOWS
#	ifdef _GPUCV_DLL
#		define _GPUCV_EXPORT			__declspec(dllexport)
#		define _GPUCV_EXPORT_C			extern "C"  _GPUCV_EXPORT
#	else
#		ifdef __cplusplus
#			define _GPUCV_EXPORT			__declspec(dllimport)
#			define _GPUCV_EXPORT_C			extern "C" _GPUCV_EXPORT
#		else
#			define _GPUCV_EXPORT			__declspec(dllimport)
#			define _GPUCV_EXPORT_C			_GPUCV_EXPORT
#		endif
#	endif
#else
#	define _GPUCV_EXPORT
#	ifdef __cplusplus
#		define _GPUCV_EXPORT_C extern "C"
#	else
#		define _GPUCV_EXPORT_C
#	endif
#endif

#define GCV_OPER_ASSERT(ASSERT_TEST, MSG) 			if (!(ASSERT_TEST)) throw GCV::Exception			(__FILE__, __LINE__, #ASSERT_TEST, #MSG); //SGE::SG_Assert(ASSERT_TEST, MSG)
#define GCV_OPER_COMPAT_ASSERT(ASSERT_TEST, MSG)	if (!(ASSERT_TEST)) throw GCV::ExceptionCompat		(__FILE__, __LINE__, #ASSERT_TEST, #MSG);
#define GCV_OPER_TODO_ASSERT(ASSERT_TEST, MSG)		if (!(ASSERT_TEST)) throw GCV::ExceptionTodo		(__FILE__, __LINE__, #ASSERT_TEST, #MSG);

#define GPUCV_START_OP(CV_FCT, FCT_NAME, DEST_IMG, HRD_PRFL)\
	GPUCV_FUNCNAME(FCT_NAME);\
	{\
		try\
		{\
			GPUCV_DEBUG("");\
			GPUCV_DEBUG("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");\
			GPUCV_DEBUG("Start operator:" << GPUCV_GET_FCT_NAME());\
			GPUCV_DEBUG("================================================================");\
			LogIndentIncrease();\
			SetThread();\
			GCV_OPER_COMPAT_ASSERT(GetHardProfile()->IsCompatible(HRD_PRFL), "Hardware profile("<< HRD_PRFL << ") not compatible with current operator");

//GPUCV_PROFILE_CURRENT_FCT(FCT_NAME, DEST_IMG, true, GpuCVSettings::GPUCV_SETTINGS_PROFILING_OPER);


#define GPUCV_STOP_OP(CV_FCT, IMG1, IMG2, IMG3, IMG4)\
			UnsetThread();\
			LogIndentDecrease();\
			GPUCV_DEBUG("================================================================");\
			GPUCV_DEBUG("Stop operator:" << GPUCV_GET_FCT_NAME());\
			GPUCV_DEBUG("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");\
			GPUCV_DEBUG("");\
		}\
		catch(GCV::Exception &e)\
		{\
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
			GPUCV_ERROR("=============================================================")\
			GPUCV_ERROR("=================== Exception catched Start =================")\
				LogIndentIncrease();\
				GPUCV_ERROR("========Catched in:");\
					LogIndentIncrease();\
					GPUCV_ERROR("Function : "<< GPUCV_GET_FCT_NAME());\
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
				if(GetGpuCVSettings()->GetLastExceptionObj())\
				{\
					GPUCV_NOTICE(GetGpuCVSettings()->GetExceptionObjectTree());\
				}\
				GPUCV_ERROR("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");\
				GPUCV_ERROR("Printing Texture manager data  =============================");\
				GetTextureManager()->PrintAllObjects();										\
				GPUCV_ERROR("============================================================");\
				if(_GPUCV_DEBUG_MODE){Beep(3000, 2);}\
				LogIndentDecrease();\
			GPUCV_ERROR("=================== Exception catched End ===================");\
			GPUCV_ERROR("=============================================================");\
		}\
	}

//Deprecated
/*	if( GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_SYNCHRONIZE_ON_ERROR))\
	{\
	GPUCV_NOTICE("Moving back all CvArr* to cpu to continue process on CPU using OpenCv.")\
	cvError( 0, std::string(GPUCV_GET_FCT_NAME()).data(), e.what(), __FILE__, __LINE__ );\
	if (IMG1!=NULL)cvgSynchronize(IMG1);\
	if (IMG2!=NULL)cvgSynchronize(IMG2);\
	if (IMG3!=NULL)cvgSynchronize(IMG3);\
	if (IMG4!=NULL)cvgSynchronize(IMG4);\
	CV_FCT;\
	}\
#define IS_OPENCV_FORCED() GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_USE_OPENCV)
*/
#define IS_OPENCV_FORCED() 0
#endif//CVGPU_CONFIG_H
