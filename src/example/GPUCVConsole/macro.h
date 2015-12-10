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
#ifndef __GPUCV_MAINSAMPLETEST_MACRO_H__
#define __GPUCV_MAINSAMPLETEST_MACRO_H__
#include "StdAfx.h"


#define _PROFILE_

#ifdef _PROFILE_
#include <SugoiTracer/appli.h>

#define _PROFILE_TO_CONSOLE false

#define _PROFILE_BLOCK_INITPARAM(STRPARAMS)\
	SG_TRC::CL_TRACE_BASE_PARAMS *_PROFILE_PARAMS = new SG_TRC::CL_TRACE_BASE_PARAMS();\
	_PROFILE_PARAMS->AddParamAtStart("type", STRPARAMS);

#define _PROFILE_BLOCK_SETPARAM_SIZE(IMG)\
	if(IMG!=NULL)\
	{\
		_PROFILE_PARAMS->AddParam("width", GetWidth(IMG));\
		_PROFILE_PARAMS->AddParam("height", GetHeight(IMG));\
		_PROFILE_PARAMS->AddParam("channels", GetnChannels(IMG));\
		_PROFILE_PARAMS->AddParam("format", GetStrCVPixelType(GetCVDepth(IMG)));\
	}

#define _PROFILE_BLOCK(NAME, PARAMS)\
		  SG_TRC::CL_TEMP_TRACER<SG_TRC::CL_TimerVal> Tracer (NAME, PARAMS, _PROFILE_TO_CONSOLE);

#define _PROFILE_BLOCK_GPU(NAME, PARAMS)\
		  SG_TRC::CL_TEMP_TRACER<SG_TRC::CL_TimerVal> Tracer (NAME, PARAMS, _PROFILE_TO_CONSOLE);\
		  Tracer.SetOpenGL(true);
#else
	#define _PROFILE_BLOCK_INITPARAM(STRPARAMS)
	#define _PROFILE_BLOCK_SETPARAM_SIZE(IMG)
	#define _PROFILE_BLOCK(NAME, PARAMS)
	#define _PROFILE_BLOCK_GPU(NAME, PARAMS)
#endif

#if _GPUCV_DEPRECATED
/*
#define _CV_TimerStart(FctName){\
	FctTrcCV->StartTrc();\
	}

#define _CV_TimerStop(FctName){\
	FctTrcCV->StopTrc();\
	}

#define _GPU_TimerStart(FctName)\
	{   FctTrcGPU->StartTrc();\
	}

#define _GPU_TimerStop(FctName){\
	FctTrcGPU->StopTrc();\
	}
	*/
#endif


/**	\brief Enum to specify all the version of an operator to test.
*/
enum OperType{
	OperOpenCV	= 1,
	OperIPP		= 2,
	OperGLSL	= 4,
#if !_GCV_CUDA_EXTERNAL
#ifdef _GPUCV_SUPPORT_CUDA
	OperCuda	= 8,
#else
	OperCuda	= 0,//this disable all CUDA image creation
#endif
#else
	OperCuda	= 0,//this disable all CUDA image creation
#endif
	OperSW		= 16,
	OperALL		= OperOpenCV|OperGLSL|OperCuda|OperIPP|OperSW
};



#include <iostream>

//==================================


/** \brief Benchmark loop for OpenCV functions
*	\param FctName => Function name as string
*	\param FUNCTION => Function to call with parameters such as: cvAdd(src1, src2, destCV);
*	\param PARAMS => Optional parameter string used to customized benchmarks
*	This macro is used to call and benchmark given function a certain amount of time(NB_ITER_BENCH).
It can be enabled/disabled by calling "enable/disable opencv" in the console.
*	\note Images src1 and scr2 are synchronized (cvgSynchronize) before calling the loop to ensure data are on CPU.
*	\sa OperType.
*/
//FctTrcCV = AppliTracer()->AddRecord(FctName, (SelectIpp)?_BENCH_IPP_TYPE:_BENCH_OPENCV_TYPE, PARAMS,src1->width, src1->height);
#define _CV_benchLoop(FUNCTION, PARAMS)\
if(__CurrentOperMask & OperOpenCV)\
{\
	std::string strLocalFctName = (CurLibraryName!="" && 0)?CurLibraryName + "::" + FctName:FctName;\
	GPUCV_NOTICE("Testing operator CV: '" << strLocalFctName  << "("<< PARAMS << ")'");\
	GPUCV_DEBUG("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");\
	GPUCV_DEBUG("Start " << ((SelectIpp)?"IPP":"CV") << " test function:" << strLocalFctName );\
	GPUCV_DEBUG("================================================================");\
	{\
		SGE::LoggerAutoIndent LocalIndent;\
		if(src1)cvgSynchronize(src1);\
		if(src2)cvgSynchronize(src2);\
		if (GlobMask)cvgSynchronize(GlobMask);\
		for(int i=0;i<NB_ITER_BENCH;i++)\
		{\
			GPUCV_DEBUG("~~~~~~~~~~~"<< strLocalFctName << "->Iteration " << i << "start ~~~~~~~~~~~~");\
			{\
				_PROFILE_BLOCK_INITPARAM((SelectIpp)?GPUCV_IMPL_IPP_STR:GPUCV_IMPL_OPENCV_STR);\
				_PROFILE_BLOCK_SETPARAM_SIZE(src1);\
				if(PARAMS!="")_PROFILE_PARAMS->AddParam("option", PARAMS.data());\
				_PROFILE_BLOCK(strLocalFctName, _PROFILE_PARAMS);\
				FUNCTION;\
			}\
			GPUCV_DEBUG("~~~~~~~~~~~"<< strLocalFctName  << "->Iteration " << i << "stop ~~~~~~~~~~~~");\
		}\
	}\
	GPUCV_DEBUG("================================================================");\
	GPUCV_DEBUG("Stop " << ((SelectIpp)?"IPP":"CV") << " test function:" << strLocalFctName );\
	GPUCV_DEBUG("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");\
}

//"cv"+
//strLocalFctName = (CurLibraryName!="")?CurLibraryName + "::" + FctName:FctName;\
//if(pCurrentFunction)pCurrentFunction->SetForcedImpl(NULL);\
	

//	if(!destSW)continue;\
//		pCurrentFunction->SetProfileMode(true);\


#ifdef _GPUCV_SUPPORT_SWITCH
#define _SW_benchLoop(FUNCTION,PARAMS)\
if(__CurrentOperMask & OperSW)\
{\
	GPUCV_NOTICE("");\
	std::string strLocalFctName = (CurLibraryName!="" && 0)?CurLibraryName + "::" + FctName:FctName;\
	GPUCV_DEBUG("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");\
	GPUCV_DEBUG("Start GPUCVSwitch test function:" << strLocalFctName);\
	GPUCV_DEBUG("================================================================");\
	FctName = "cv"+FctName;\
	SG_TRC::CL_TRACE_BASE_PARAMS LocalProfileParams;\
	CL_FctSw * pCurrentFunction=CL_FctSw_Mngr::Instance().Find(FctName);\
	if(!pCurrentFunction)\
	{\
		PUSH_GPUCV_OPTION();\
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_PROFILING||GpuCVSettings::GPUCV_SETTINGS_PLUG_THROW_EXCEPTIONS, false);\
		FUNCTION;\
		POP_GPUCV_OPTION();\
		pCurrentFunction=CL_FctSw_Mngr::Instance().Find(FctName);\
	}\
	if(pCurrentFunction==NULL)\
	{ GPUCV_ERROR(FctName << "-> Given function name could not be found in the switch function manager!");}\
	else\
	{\
		switchFctStruct * pCurrentImpl = NULL;\
		std::string strImpl = "";\
		CvArr* pCurDest = NULL;\
		size_t uiImplNbr = pCurrentFunction->GetImplSwitchNbr();\
		if(uiImplNbr==0)GPUCV_ERROR("No implementation found for '" << FctName << "'");\
		CvArr ** destArray = new CvArr * [uiImplNbr];\
		destArray[0] = destSW;\
		for(size_t i=1; i< uiImplNbr; i++) destArray[i] = cvCloneImage((IplImage*)destSW);\
		unsigned int uiWinPos = 10;\
		std::string strWinName;\
		CvArr * pOpenCVReference=NULL;\
		for (size_t i = 0; i < uiImplNbr; i++)\
		{\
			pCurrentImpl = pCurrentFunction->GetImplSwitch(i);\
			SG_Assert(pCurrentImpl, "Given implementation could not be found!");\
			SG_Assert(pCurrentImpl->m_Implementation, "Given implementation have no descriptor!");\
			GPUCV_SWITCH_LOG("Forcing function '"<< FctName << "' to implementation:" << pCurrentImpl->m_Implementation->m_strImplName);\
			pCurrentFunction->SetForcedImpl(pCurrentImpl->m_Implementation);\
			destSW = (IplImage*)destArray[i];\
			if(!destSW)continue;\
			strImpl = GetStrImplemtation(pCurrentImpl->m_Implementation);\
			strLocalFctName = FctName;\
			cvgSetLabel(destSW, strLocalFctName);\
			if (ShowImage==true)\
			{\
				__GetWindowsName__(strWinName,"");\
				strWinName +="("+ strImpl+ ")";\
				cvNamedWindow(strWinName.data(),1);\
				cvMoveWindow(strWinName.data(), uiWinPos, 10);\
				uiWinPos+=80;\
			}\
			try{\
				for(int l=0;l<NB_ITER_BENCH;l++)\
				{\
					GPUCV_DEBUG("~~~~~~~~~~~"<< strLocalFctName << "->Iteration " << l << "start ~~~~~~~~~~~~");\
					{\
						_PROFILE_BLOCK_INITPARAM(strImpl);\
						_PROFILE_BLOCK_SETPARAM_SIZE(destSW);\
						if(PARAMS!="")_PROFILE_PARAMS->AddParam("option", PARAMS.data());\
						_PROFILE_PARAMS->AddParam("switch", 1);\
						LocalProfileParams=*_PROFILE_PARAMS;\
						_PROFILE_BLOCK(strLocalFctName, _PROFILE_PARAMS);\
						FUNCTION;\
						cvgFlush(destSW);\
					}\
				}\
				if (ShowImage==true)cvgShowImage(strWinName.data(),destSW);\
				if(pCurrentImpl->m_Implementation->m_baseImplID == GPUCV_IMPL_OPENCV)\
				{\
					GPUCV_SWITCH_LOG("Setting reference image");\
					pOpenCVReference = destSW;\
				}\
				else if(ControlOperators && pOpenCVReference!=NULL && pOpenCVReference!=destSW && !CV_IS_MAT(destSW))\
				{\
					GPUCV_LOG(GpuCVSettings::GPUCV_SETTINGS_DEBUG_MEMORY, "MEM", "\nMemory allocated after operator: " << DataDsc_Base::ms_totalMemoryAllocated);\
					float fError = ControlResultsImages(pOpenCVReference, destSW, strLocalFctName, PARAMS);\
					if(fError < fEpsilonOperTest)\
					{   GPUCV_NOTICE("Testing operator CV " << strImpl << " '" << strLocalFctName << "("<< PARAMS << ")'" << green <<" passed!" << white);}\
					else\
					{\
						GPUCV_NOTICE("Testing operator CV " << strImpl << " '" << strLocalFctName << "("<< PARAMS << ")'" << red <<" failed"  << white << " with rate " << fError << "!!!!!!");\
						cvgswSetFctLastExecutionState(FctName.data(), LocalProfileParams.GetParamsValue().data(), SG_TRC::EXEC_FAILED_FALSE_RESULT);\
					}\
				}\
			}\
			catch(GCV::ExceptionCompat &e)\
			{\
				if(ControlOperators){GPUCV_NOTICE("Testing operator CV >> " << strImpl << " '" << strLocalFctName << "(" << PARAMS << ")'" << blue <<" failed"  << white << ", not compatible(" << e.GetOriginalMsg() << ").");}\
				else\
				{\
					GPUCV_ERROR("Operator '" << GPUCV_GET_FCT_NAME() << "' is not compatible with the current parameters.");\
						LogIndentIncrease();\
						GPUCV_NOTICE("========Catched in:");\
							GPUCV_NOTICE("Line : "<< __LINE__);\
							GPUCV_NOTICE("File : "<< __FILE__);\
							LogIndentDecrease();\
						GPUCV_NOTICE("========Rised in:");\
							LogIndentIncrease();\
							GPUCV_NOTICE("Line : "<< e.GetLine());\
							GPUCV_NOTICE("File : "<< e.GetFile());\
							GPUCV_NOTICE("Msg: " << e.GetOriginalMsg());\
							LogIndentDecrease();\
						GPUCV_NOTICE("=================");\
						LogIndentDecrease();\
				}\
			}\
			catch(GCV::ExceptionTodo &e)\
			{\
				if(ControlOperators)\
				{\
					GPUCV_NOTICE("Testing operator CV >> " << strImpl << " '" << strLocalFctName << "(" << PARAMS << ")'" << blue <<" failed"  << white << ", not done yet(" << e.GetOriginalMsg() << ").");\
					cvgswSetFctLastExecutionState(FctName.data(), LocalProfileParams.GetParamsValue().data(), SG_TRC::EXEC_FAILED_WORK_REQUIRED);\
				}\
			}\
			catch(GCV::Exception &e)\
			{\
				if(ControlOperators){GPUCV_NOTICE("Testing operator CV >> " << strImpl << " '" << strLocalFctName << "(" << PARAMS << ")'" << red <<" failed"  << white << ", (" << e.GetOriginalMsg() << ").");}\
				else\
				{\
					GPUCV_ERROR("Operator '" << strLocalFctName << "' rised critical exception:");\
						LogIndentIncrease();\
						GPUCV_NOTICE("========Catched in:");\
							GPUCV_NOTICE("Line : "<< __LINE__);\
							GPUCV_NOTICE("File : "<< __FILE__);\
							LogIndentDecrease();\
						GPUCV_NOTICE("========Rised in:");\
							LogIndentIncrease();\
							GPUCV_NOTICE("Line : "<< e.GetLine());\
							GPUCV_NOTICE("File : "<< e.GetFile());\
							GPUCV_NOTICE("Msg: " << e.GetOriginalMsg());\
							LogIndentDecrease();\
						GPUCV_NOTICE("=================");\
						LogIndentDecrease();\
				}\
			}\
		}\
		for (size_t i = 0; i < uiImplNbr; i++)\
		{\
			cvgReleaseImage((IplImage**)&(destArray[i]));\
		}\
		destSW=NULL;\
		delete []destArray;\
		if (ShowImage==true)\
		{\
			cvWaitKey(0);\
			cvDestroyAllWindows();\
		}\
	}\
	GPUCV_DEBUG("================================================================");\
	GPUCV_DEBUG("Stop GPUCVSwitch test function:" << strLocalFctName);\
	GPUCV_DEBUG("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");\
}
#else
#define _SW_benchLoop(FUNCTION,PARAMS)
#endif//_GPUCV_SUPPORT_SWITCH

/** \brief Benchmark loop for GpuCV-GLSL functions
*	\param FctName => Function name as string
*	\param FUNCTION => Function to call with parameters such as: cvAdd(src1, src2, destCV);
*	\param DEST_IMG => Specify destination image so we can allocate it on GPU before benchmarking
*	\param PARAMS => Optional parameter string used to customized benchmarks
*	This macro is used to call and benchmark given function a certain amount of time(NB_ITER_BENCH).
It can be enabled/disabled by calling "enable/disable glsl" in the console.
*	\note Images src1 and scr2 are transfered to GPU before calling the loop to ensure that we do not record data transfer in the benchmarks.
*	\sa OperType.
*/
//	FctTrcGPU = AppliTracer()->AddRecord(FctName, GPUCV_IMPL_GLSL_STR, PARAMS,src1->width, src1->height);
#define _GPU_benchLoop(FUNCTION, DESTINATION_IMG, PARAMS)if(__CurrentOperMask&OperGLSL)\
{\
	std::string strLocalFctName = (CurLibraryName!="" && 0)?CurLibraryName + "::" + FctName:FctName;\
	GPUCV_DEBUG("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");\
	GPUCV_DEBUG("Start GPUCV test function:" << strLocalFctName);\
	GPUCV_DEBUG("================================================================");\
	try{\
		SGE::LoggerAutoIndent LocalIndent;\
		if(!IS_OPENCV_FORCED())\
		{\
			std::string temp_label;\
			if((void *)DESTINATION_IMG != NULL)\
			{\
				cvgPushSetOptions(DESTINATION_IMG, DataContainer::DEST_IMG, true);\
				if (AvoidCPUReturn)cvgUnsetCpuReturn(DESTINATION_IMG);\
				if(DataPreloading)\
				cvgSetLocation<DataDsc_GLTex>(DESTINATION_IMG, false);\
			}\
			if(src1 != NULL)\
			{\
				if (AvoidCPUReturn)cvgUnsetCpuReturn(src1);\
				cvgSetOptions(src1, DataContainer::UBIQUITY, true);\
				temp_label = FctName;\
				temp_label += "_Src1";\
				cvgSetLabel(src1, temp_label);\
				if(DataPreloading)\
				cvgSetLocation<DataDsc_GLTex>(src1, true);\
			}\
			if (src2 != NULL)\
			{\
				if (AvoidCPUReturn)cvgUnsetCpuReturn(src2);\
				cvgSetOptions(src2, DataContainer::UBIQUITY, true);\
				temp_label = FctName;\
				temp_label += "_Src2";\
				cvgSetLabel(src2, temp_label);\
				if(DataPreloading)\
				cvgSetLocation<DataDsc_GLTex>(src2, true);\
			}\
			if (GlobMask != NULL)\
			{\
				if (AvoidCPUReturn)cvgUnsetCpuReturn(GlobMask);\
				cvgSetOptions(GlobMask, DataContainer::UBIQUITY, true);\
				temp_label = FctName;\
				temp_label += "_Mask";\
				cvgSetLabel(GlobMask, temp_label);\
				if(DataPreloading)\
				cvgSetLocation<DataDsc_GLTex>(GlobMask, true);\
			}\
		}\
		SynchronizeOper(OperGLSL, src1);\
		for(int i=0;i<NB_ITER_BENCH;i++) \
		{\
			GPUCV_DEBUG("~~~~~~~~~~~"<< strLocalFctName << "->Iteration " << i << "start ~~~~~~~~~~~~");\
			SynchronizeOper(OperGLSL, DESTINATION_IMG);\
			{\
				_PROFILE_BLOCK_INITPARAM(GPUCV_IMPL_GLSL_STR);\
				_PROFILE_BLOCK_SETPARAM_SIZE(src1);\
				if(PARAMS!="")_PROFILE_PARAMS->AddParam("option", PARAMS.data());\
				_PROFILE_BLOCK_GPU(strLocalFctName, _PROFILE_PARAMS);\
				FUNCTION;\
				SynchronizeOper(OperGLSL, DESTINATION_IMG);\
			}\
			GPUCV_DEBUG("~~~~~~~~~~~"<< strLocalFctName << "->Iteration " << i << "stop ~~~~~~~~~~~~");\
		}\
		if(ControlOperators && destCV!=NULL && destCV!=NULL && !CV_IS_MAT(destCV))\
		{\
			GPUCV_LOG(GpuCVSettings::GPUCV_SETTINGS_DEBUG_MEMORY, "MEM", "\nMemory allocated after operator: " << DataDsc_Base::ms_totalMemoryAllocated);\
			float fError = ControlResultsImages(destCV, destGLSL, strLocalFctName, PARAMS);\
			if(fError < fEpsilonOperTest)\
			{   GPUCV_NOTICE("Testing operator CV >> GLSL '" << strLocalFctName << "("<< PARAMS << ")'" << green <<" passed!" << white);}\
			else\
			{	GPUCV_NOTICE("Testing operator CV >> GLSL '" << strLocalFctName << "("<< PARAMS << ")'" << red <<" failed"  << white << " with rate " << fError << "!!!!!!");}\
		}\
		if(!IS_OPENCV_FORCED())\
		{\
			if(DESTINATION_IMG != NULL)\
			{\
				if (AvoidCPUReturn)cvgSetCpuReturn(DESTINATION_IMG);\
				cvgPopOptions(DESTINATION_IMG);\
			}\
		}\
	}\
	catch(GCV::ExceptionCompat &e)\
	{\
		if(ControlOperators){GPUCV_NOTICE("Testing operator CV >> GLSL '" << strLocalFctName << "("<< PARAMS << ")'" << blue <<" failed"  << white << ", not compatible(" << e.GetOriginalMsg() << ").");}\
		else\
		{\
			GPUCV_ERROR("Operator '" << GPUCV_GET_FCT_NAME() << "' is not compatible with the current parameters.");\
				LogIndentIncrease();\
				GPUCV_NOTICE("========Catched in:");\
					GPUCV_NOTICE("Line : "<< __LINE__);\
					GPUCV_NOTICE("File : "<< __FILE__);\
					LogIndentDecrease();\
				GPUCV_NOTICE("========Rised in:");\
					LogIndentIncrease();\
					GPUCV_NOTICE("Line : "<< e.GetLine());\
					GPUCV_NOTICE("File : "<< e.GetFile());\
					GPUCV_NOTICE("Msg: " << e.GetOriginalMsg());\
					LogIndentDecrease();\
				GPUCV_NOTICE("=================");\
				LogIndentDecrease();\
		}\
	}\
	catch(GCV::ExceptionTodo &e)\
	{\
		if(ControlOperators){GPUCV_NOTICE("Testing operator CV >> GLSL '" << strLocalFctName << "("<< PARAMS << ")'" << blue <<" failed"  << white << ", not done yet(" << e.GetOriginalMsg() << ").");}\
	}\
	catch(GCV::Exception &e)\
	{\
		if(ControlOperators){GPUCV_NOTICE("Testing operator CV >> GLSL '" << strLocalFctName << "("<< PARAMS << ")'" << red <<" failed"  << white << ", (" << e.GetOriginalMsg() << ").");}\
		else\
		{\
			GPUCV_ERROR("Operator '" << strLocalFctName << "' is not compatible with the current parameters.");\
				LogIndentIncrease();\
				GPUCV_NOTICE("========Catched in:");\
					GPUCV_NOTICE("Line : "<< __LINE__);\
					GPUCV_NOTICE("File : "<< __FILE__);\
					LogIndentDecrease();\
				GPUCV_NOTICE("========Rised in:");\
					LogIndentIncrease();\
					GPUCV_NOTICE("Line : "<< e.GetLine());\
					GPUCV_NOTICE("File : "<< e.GetFile());\
					GPUCV_NOTICE("Msg: " << e.GetOriginalMsg());\
					LogIndentDecrease();\
				GPUCV_NOTICE("=================");\
				LogIndentDecrease();\
		}\
	}\
	GPUCV_DEBUG("================================================================");\
	GPUCV_DEBUG("Stop GPUCV test function:" << strLocalFctName);\
	GPUCV_DEBUG("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");\
}


#define _GCU_BENCH_PRELOAD_INTO_ARRAY	0
#define _GCU_BENCH_PRELOAD_INTO_BUFFER	1

/** \brief Benchmark loop for GpuCV-CUDA functions
*	\param FUNCTION => Function to call with parameters such as: cvAdd(src1, src2, destCV);
*	\param DESTINATION_IMG => Specify destination image so we can allocate it on GPU before benchmarking
*	\param PARAMS => Optional parameter string used to customized benchmarks
*	This macro is used to call and benchmark given function a certain amount of time(NB_ITER_BENCH).
It can be enabled/disabled by calling "enable/disable cuda" in the console.
*	\note Images src1 and scr2 are transfered to GPU before calling the loop to ensure that we do not record data transfer in the benchmarks.
*	\sa OperType.
*/

#if defined _GPUCV_SUPPORT_CUDA && !_GCV_CUDA_EXTERNAL

//	FctTrcGPU = AppliTracer()->AddRecord(FctName, _BENCH_CUDA_TYPE, PARAMS,src1->width, src1->height);
#define _CUDA_benchLoop(FUNCTION, DESTINATION_IMG, PARAMS)if(__CurrentOperMask&OperCuda)\
{\
	std::string strLocalFctName = (CurLibraryName!="" && 0)?CurLibraryName + "::" + FctName:FctName;\
	GPUCV_DEBUG("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");\
	GPUCV_DEBUG("Start CUDA test function:" << strLocalFctName);\
	GPUCV_DEBUG("================================================================");\
	try{\
	SGE::LoggerAutoIndent LocalIndent;\
	if(!IS_OPENCV_FORCED())\
	{\
		if((void *)DESTINATION_IMG != NULL)\
		{\
			cvgPushSetOptions(DESTINATION_IMG, DataContainer::DEST_IMG, true);\
			if (AvoidCPUReturn)cvgUnsetCpuReturn(DESTINATION_IMG);\
			if(DataPreloading)\
			{\
				cvgSetLocation<DataDsc_CUDA_Buffer>(DESTINATION_IMG,false);\
			}\
		}\
		if(src1 != NULL)\
		{\
			if (AvoidCPUReturn)cvgUnsetCpuReturn(src1);\
			cvgSetOptions(src1, DataContainer::UBIQUITY, true);\
			cvgSetLabel(src1, std::string(FctName) +  std::string("_Src1"));\
			if(DataPreloading)\
			{\
				if(_GCU_BENCH_PRELOAD_INTO_ARRAY)cvgSetLocation<DataDsc_CUDA_Array>(src1,true);\
				if(_GCU_BENCH_PRELOAD_INTO_BUFFER)cvgSetLocation<DataDsc_CUDA_Buffer>(src1,true);\
			}\
		}\
	if (src2 != NULL)\
{\
	if (AvoidCPUReturn)cvgUnsetCpuReturn(src2);\
	cvgSetOptions(src2, DataContainer::UBIQUITY, true);\
	cvgSetLabel(src2, std::string(FctName) +  std::string("_Src2"));\
	if(DataPreloading)\
{\
	if(_GCU_BENCH_PRELOAD_INTO_ARRAY)cvgSetLocation<DataDsc_CUDA_Array>(src2,true);\
	if(_GCU_BENCH_PRELOAD_INTO_BUFFER)cvgSetLocation<DataDsc_CUDA_Buffer>(src2,true);\
	}\
	}\
	if (GlobMask != NULL)\
{\
	if (AvoidCPUReturn)cvgUnsetCpuReturn(GlobMask);\
	cvgSetOptions(GlobMask, DataContainer::UBIQUITY, true);\
	cvgSetLabel(GlobMask, std::string(FctName) +  std::string("_Mask"));\
	if(DataPreloading)\
{\
	if(_GCU_BENCH_PRELOAD_INTO_ARRAY)cvgSetLocation<DataDsc_CUDA_Array>(GlobMask,true);\
	if(_GCU_BENCH_PRELOAD_INTO_BUFFER)cvgSetLocation<DataDsc_CUDA_Buffer>(GlobMask,true);\
	}\
	}\
	}\
	for(int i=0;i<NB_ITER_BENCH;i++) \
	{\
		GPUCV_DEBUG("~~~~~~~~~~~"<< strLocalFctName << "->Iteration " << i << "start ~~~~~~~~~~~~");\
		SynchronizeOper(OperCuda, DESTINATION_IMG);\
		{\
			_PROFILE_BLOCK_INITPARAM(GPUCV_IMPL_CUDA_STR);\
			_PROFILE_BLOCK_SETPARAM_SIZE(src1);\
			_PROFILE_BLOCK_GPU(strLocalFctName, _PROFILE_PARAMS);\
			if(PARAMS!="")_PROFILE_PARAMS->AddParam("option", PARAMS.data());\
			FUNCTION;\
			SynchronizeOper(OperCuda, DESTINATION_IMG);\
			CUT_CHECK_ERROR("Kernel execution failed");\
		}\
		GPUCV_DEBUG("~~~~~~~~~~~"<< FctName << "->Iteration " << i << "stop ~~~~~~~~~~~~");\
	}\
	if(ControlOperators && destCUDA!=NULL && destCV!=NULL && !CV_IS_MAT(destCV))\
	{\
		GPUCV_LOG(GpuCVSettings::GPUCV_SETTINGS_DEBUG_MEMORY, "MEM", "\nMemory allocated after operator: " << DataDsc_Base::ms_totalMemoryAllocated);\
		float fError = ControlResultsImages(destCV, destCUDA, strLocalFctName, PARAMS);\
		if(fError < fEpsilonOperTest)\
		{   GPUCV_NOTICE("Testing operator CV >> CUDA '" << strLocalFctName << "("<< PARAMS << ")'" << green <<" passed!" << white);}\
		else\
		{	GPUCV_NOTICE("Testing operator CV >> CUDA '" << strLocalFctName << "("<< PARAMS << ")'" << red <<" failed "  << white << "with rate " << fError << "!!!!!!");}\
	}\
	if(!IS_OPENCV_FORCED())\
	{\
		if(DESTINATION_IMG != NULL)\
		{\
			if (AvoidCPUReturn)cvgSetCpuReturn(DESTINATION_IMG);\
			cvgPopOptions(DESTINATION_IMG);\
		}\
	}\
	}\
	catch(std::exception &e)\
	{\
		if(ControlOperators && dynamic_cast<GCV::ExceptionTodo*> (&e))\
		{\
			GCV::ExceptionTodo* pExcep = dynamic_cast<GCV::ExceptionTodo*> (&e);\
			GPUCV_NOTICE("Testing operator CV >> CUDA '" << strLocalFctName << "("<< PARAMS << ")'" << blue <<" failed"  << white << ", not done yet(" << pExcep->GetOriginalMsg() << ").");\
		}\
		if (ControlOperators && dynamic_cast<GCV::ExceptionCompat*>(&e))\
		{\
			GCV::ExceptionCompat* pExcep = dynamic_cast<GCV::ExceptionCompat*> (&e);\
			GPUCV_NOTICE("Testing operator CV >> CUDA '" << strLocalFctName << "("<< PARAMS << ")'" << blue <<" failed"  << white << ", not compatible(" << pExcep->GetOriginalMsg() << ").");\
		}\
		else if(ControlOperators && dynamic_cast<GCV::Exception*>(&e))\
		{\
			GCV::Exception* pExcep = dynamic_cast<GCV::Exception*> (&e);\
			GPUCV_NOTICE("Testing operator CV >> CUDA '" << strLocalFctName << "("<< PARAMS << ")'" << red <<" failed"  << white << ", (" << pExcep->GetOriginalMsg() << ").");\
		}\
		else\
		{\
			GPUCV_ERROR("Operator '" << strLocalFctName << "' is not compatible with the current parameters.");\
				LogIndentIncrease();\
				GPUCV_NOTICE("========Catched in:");\
					GPUCV_NOTICE("Line : "<< __LINE__);\
					GPUCV_NOTICE("File : "<< __FILE__);\
					LogIndentDecrease();\
				GPUCV_NOTICE("========Rised in:");\
					LogIndentIncrease();\
					GPUCV_NOTICE(e.what());\
					LogIndentDecrease();\
				GPUCV_NOTICE("=================");\
				LogIndentDecrease();\
		}\
	}\
	GPUCV_DEBUG("================================================================");\
	GPUCV_DEBUG("Stop CUDA test function:" << strLocalFctName);\
	GPUCV_DEBUG("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");\
	}
#else
#define _CUDA_benchLoop(FUNCTION, DESTINATION_IMG, PARAMS)
#endif

/** \brief Create several destination images with specify format and size depending on implementation selected to run (GpuCVSelectionMask & MASK).
*	\param  SIZE => CvSize of destination images, usually same as source.
*	\param  DEPTH => depth of destination images, usually same as source.
*	\param  CHANNELS => nChannels of destination images, usually same as source.
*	\param  MASK => mask of operator type to run based on OperType.
*	This macro defines several destination images:
<ul>
<li>destCV: for OpenCv and IPP destination image</li>
<li>destGLSL: for GpuCV-GLSL destination image</li>
<li>destCUDA: for GpuCV-CUDA destination image</li>
</ul>
<br>They are all set to NULL by default and will be created if the condition is respected:
\code
unsigned char __CurrentOperMask = (MASK) & GpuCVSelectionMask;
...
if(__CurrentOperMask & OperOpenCV)
...
\endcode
A label is given by default such as:
\code
...
cvgSetLabel(destCV,GPUCV_GET_FCT_NAME()+"-destCV");\
...
\endcode
*	\sa  __ReleaseImages__, __CreateMatrixes__, __ReleaseMatrixes__.
*/
#define __CreateImages__( SIZE, DEPTH, CHANNELS, MASK)\
	unsigned char __CurrentOperMask = (MASK) & GpuCVSelectionMask;\
	IplImage * destCV  = NULL;\
	IplImage * destGLSL  = NULL;\
	IplImage * destCUDA  = NULL;\
	IplImage * destSW = cvgCreateImage(SIZE,DEPTH,CHANNELS);\
	cvgSetLabel(destSW,GPUCV_GET_FCT_NAME()+"-destSW");\
	if(__CurrentOperMask & OperOpenCV)\
	{\
		destCV = cvgCreateImage(SIZE,DEPTH,CHANNELS);\
		cvgSetLabel(destCV,GPUCV_GET_FCT_NAME()+"-destCV");\
	}\
	if(__CurrentOperMask & OperGLSL)\
	{\
		destGLSL = cvgCreateImage(SIZE,DEPTH,CHANNELS);\
		cvgSetLabel(destGLSL,GPUCV_GET_FCT_NAME()+"-destGLSL");\
	}\
	if(__CurrentOperMask & OperCuda)\
	{\
		destCUDA = cvgCreateImage(SIZE,DEPTH,CHANNELS);\
		cvgSetLabel(destCUDA,GPUCV_GET_FCT_NAME()+"-destCUDA");\
	}

/** \brief Release destination images created by __CreateImages__.
*	\sa  __CreateImages__, __CreateMatrixes__, __ReleaseMatrixes__.
*/
#define __ReleaseImages__()\
	if(__CurrentOperMask & OperOpenCV)\
	cvgReleaseImage(&destCV);\
	if(__CurrentOperMask & OperGLSL)\
	cvgReleaseImage(&destGLSL);\
	if(__CurrentOperMask & OperCuda)\
	cvgReleaseImage(&destCUDA);\
	if(destSW)cvgReleaseImage(&destSW);
	

/** \brief Create several destination matrices, see __CreateImages__ for details.
*	\param  SIZE_X => width of the destination matrices.
*	\param  SIZE_Y => height of the destination matrices.
*	\param  DEPTH => depth of the destination matrices.
*	\param  MASK => mask of operator type to run based on OperType.
*	This macro defines several destination matricess:
<ul>
<li>destCV: for OpenCv and IPP destination image</li>
<li>destGLSL: for GpuCV-GLSL destination image</li>
<li>destCUDA: for GpuCV-CUDA destination image</li>
</ul>
*	\sa __ReleaseMatrixes__, __CreateImages__, __ReleaseImages__.
*/
#define __CreateMatrixes__( SIZE_X, SIZE_Y,DEPTH, MASK)\
	unsigned char __CurrentOperMask = (MASK) & GpuCVSelectionMask;\
	CvMat * destCV  = NULL;\
	CvMat * destGLSL  = NULL;\
	CvMat * destCUDA  = NULL;\
	CvMat * destSW = cvgCreateMat(SIZE_X, SIZE_Y,DEPTH);\
	cvgSetLabel(destSW,"destCV-Mat");\
	CvMat * dstArray[OperALL]={NULL};\
	if(__CurrentOperMask & OperOpenCV)\
	{\
		destCV = cvgCreateMat(SIZE_X, SIZE_Y,DEPTH);\
		cvgSetLabel(destCV,"destCV-Mat");\
		destCV->hdr_refcount=0;\
	}\
	if(__CurrentOperMask & OperGLSL)\
	{\
		destGLSL = cvgCreateMat(SIZE_X, SIZE_Y,DEPTH);\
		cvgSetLabel(destGLSL,"destGLSL-Mat");\
		destGLSL->hdr_refcount=0;\
	}\
	if(__CurrentOperMask & OperCuda)\
	{\
		destCUDA = cvgCreateMat(SIZE_X, SIZE_Y,DEPTH);\
		cvgSetLabel(destCUDA,"destCUDA-Mat");\
		destCUDA->hdr_refcount=0;\
	}



/** \brief Release destination images created by __CreateImages__.
*	\sa __CreateMatrixes__, __CreateImages__, __ReleaseImages__.
*/
#define __ReleaseMatrixes__()\
{\
	if(__CurrentOperMask & OperOpenCV)\
	cvgReleaseMat(&destCV);\
	if(__CurrentOperMask & OperGLSL)\
	cvgReleaseMat(&destGLSL);\
	if(__CurrentOperMask & OperCuda)\
	cvgReleaseMat(&destCUDA);\
	if(destSW)\
	cvgReleaseMat(&destSW);\
	}

//destSW image is released in the switch macro directly.\
	


#if defined _GPUCV_SUPPORT_CUDA && !_GCV_CUDA_EXTERNAL
#define __SetInputImage_CUDA__(iImage, LABEL)\
	if(__CurrentOperMask & OperCuda)\
{\
	if(_GCU_BENCH_PRELOAD_INTO_ARRAY)cvgSetLocation<DataDsc_CUDA_Array>(iImage, true);\
	if(_GCU_BENCH_PRELOAD_INTO_BUFFER)cvgSetLocation<DataDsc_CUDA_Buffer>(iImage, true);\
	}
#else
#define __SetInputImage_CUDA__(iImage, LABEL)
#endif

/** \brief Declare given image as an input image, it affects UBICUITY/CPU_RETURN/DESTINATION_IMG falgs and set a new label.
*	\param  iImage => image to process.
*	\param  LABEL => new image label.
*	This macro affect some flags relative to data manipulations:
<ul>
<li>DataContainer::UBICUITY => true. Allow multiple instances of the data(CPU/GPU)</li>
<li>DataContainer::CPU_RETURN => false. Avoid returning data to CPU after applying the filter.</li>
<li>DataContainer::DESTINATION_IMG => false. Define image as input.</li>
</ul>
Then it transfers the image data to corresponding locations defined by local implementation mask:
\code
...
if(__CurrentOperMask & OperGLSL)\
cvgSetLocation<DataDsc_GLTex>(iImage, true);\
...
\endcode
Image locations managed are DataDsc_IplImage/DataDsc_GLTex and locations defined in __SetInputImage_CUDA__.
*	\sa __SetInputImage_CUDA_,__SetOutputImage__.
*/
#define __SetInputImage__(iImage, LABEL)\
{\
	cvgSetOptions(iImage, DataContainer::UBIQUITY, true);\
	cvgSetOptions(iImage, DataContainer::CPU_RETURN, false);\
	cvgSetOptions(iImage, DataContainer::DEST_IMG, false);\
	cvgSetLabel(iImage, LABEL);\
	if(__CurrentOperMask & OperOpenCV)\
	cvgSetLocation<DataDsc_IplImage>(iImage, true);\
	if(__CurrentOperMask & OperGLSL)\
	cvgSetLocation<DataDsc_GLTex>(iImage, true);\
	__SetInputImage_CUDA__(iImage, true);\
	}


/** \brief Declare given image as an output image, it affects UBICUITY/CPU_RETURN/DEST_IMG falgs.
*	\param  iImage => image to process.
*	\param  LABEL => new image label.
*	This macro affect some flags relative to data manipulations:
<ul>
<li>DataContainer::UBICUITY => false. Do not allow multiple instances of the data(CPU/GPU)</li>
<li>DataContainer::CPU_RETURN => false. Avoid returning data to CPU after applying the filter.</li>
<li>DataContainer::DEST_IMG => true. Define image as output.</li>
</ul>
*	\sa __SetInputImage__.
*/
#define __SetOutputImage__(iImage, LABEL)\
{\
	cvgSetOptions(iImage, DataContainer::UBIQUITY, false);\
	cvgSetOptions(iImage, DataContainer::CPU_RETURN, false);\
	cvgSetOptions(iImage, DataContainer::DEST_IMG, true);\
	cvgSetLabel(iImage, LABEL);\
	}



#if defined _GPUCV_SUPPORT_CUDA && !_GCV_CUDA_EXTERNAL
#define __SetInputMatrix_CUDA__(iMatrix)\
	if(__CurrentOperMask & OperCuda)\
{\
	if(_GCU_BENCH_PRELOAD_INTO_ARRAY)cvgSetLocation<DataDsc_CUDA_Array>(iMatrix, true);\
	if(_GCU_BENCH_PRELOAD_INTO_BUFFER)cvgSetLocation<DataDsc_CUDA_Buffer>(iMatrix, true);\
	}
#else
#define __SetInputMatrix_CUDA__(iImage)
#endif

/** \brief Declare given matrice as an input matrice, it affects UBICUITY/CPU_RETURN/DEST_IMG falgs and set a new label.
*	\param  iMatrix => matrice to process.
*	\param  LABEL => new image label.
See __SetInputImage__ for more details.
*	\sa __SetInputImage_CUDA_,__SetOutputImage__.
*/
#define __SetInputMatrix__(iMatrix, LABEL)\
{\
	cvgSetOptions(iMatrix, DataContainer::UBIQUITY, true);\
	cvgSetOptions(iMatrix, DataContainer::CPU_RETURN, false);\
	cvgSetOptions(iMatrix, DataContainer::DEST_IMG, false);\
	cvgSetLabel(iMatrix, LABEL);\
	if(__CurrentOperMask & OperOpenCV)\
	cvgSetLocation<DataDsc_CvMat>(iMatrix, true);\
	if(__CurrentOperMask & OperGLSL)\
	cvgSetLocation<DataDsc_GLTex>(iMatrix, true);\
	__SetInputMatrix_CUDA__(iMatrix);\
	}



#define __GetWindowsName__(VAR_STR,TYPE)\
	VAR_STR = TYPE;\
	VAR_STR += GPUCV_GET_FCT_NAME();\
	VAR_STR += "()";
	
/** \brief Create several windows depending on the implementation and ShowImage flags.
*	\sa __ShowImages__.
*/
#define __CreateWindows__()\
	if (ShowImage==true){\
	std::string strWinName;\
	if(__CurrentOperMask & OperOpenCV)\
	{\
		__GetWindowsName__(strWinName,"cv");\
		cvNamedWindow(strWinName.data(),1);\
		cvMoveWindow(strWinName.data(), 10, 10);\
	}\
	if(__CurrentOperMask & OperGLSL)\
	{\
		__GetWindowsName__(strWinName,"cvg");\
		cvNamedWindow(strWinName.data(),1);\
		cvMoveWindow(strWinName.data(), 400, 10);\
	}\
	if(__CurrentOperMask & OperCuda)\
	{\
		__GetWindowsName__(strWinName,"cvgcu");\
		cvNamedWindow(strWinName.data(),1);\
		cvMoveWindow(strWinName.data(), 800, 10);\
	}\
}

/** \brief Show destination images depending on the implementation and ShowImage flags.
*	\sa __CreateWindows__.
*/
#define __ShowImages__()\
	if (ShowImage==true)\
{\
	std::string strWinName;\
	if((__CurrentOperMask & OperOpenCV) && destCV)\
	{\
		__GetWindowsName__(strWinName,"cv");\
		cvgShowImage(strWinName.data(),destCV);\
	}\
	if((__CurrentOperMask & OperGLSL) && destGLSL)\
	{\
		__GetWindowsName__(strWinName,"cvg");\
		cvgShowImage(strWinName.data(),destGLSL);\
	}\
	if((__CurrentOperMask & OperCuda) && destCUDA)\
	{\
		__GetWindowsName__(strWinName,"cvgcu");\
		cvgShowImage(strWinName.data(),destCUDA);\
	}\
	cvWaitKey(0);\
	cvDestroyAllWindows();\
	}
#endif //__GPUCV_MAINSAMPLETEST_MACRO_H__
