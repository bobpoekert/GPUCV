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
#ifndef __GPUCV_HARDWARE_PROFILING_H
#define __GPUCV_HARDWARE_PROFILING_H

#include <GPUCVHardware/config.h>
#include <GPUCVHardware/CL_Options.h>

#if _GPUCV_PROFILE
	#include <SugoiTracer/appli.h>
	#include <SugoiTracer/timer.h>
	#include <SugoiTracer/function.h>
#endif

namespace GCV{

/**
*	\brief Contains main class and macros for profiling/debugging object and functions.
*	Internal profiling of GpuCV is used to record some class member execution time. Some parameters can be included like the template type
*	or the given argument. Profiling should be used only in DEBUG mode cause it impacts on global performances of the application.
*	It also includes some MACRO to trace function call and send debugging information to the current output, see CLASS_DEBUG(), CLASS_WARNING(), CLASS_NOTICE(), CLASS_ERROR().
*   <br>CL_Profiler also inherits from CL_Option, it can be used to control local settings for each class objects. In the constructor:
\code
SetOption(GetGpuCVSettings()->GetDefaultOption(m_className), true);
\endcode
*	Is called to retrieve default options for this class type.
*	\note _GPUCV_PROFILE_CLASS and _GPUCV_PROFILE must be set to 1 to enable all profiling features.
*	\author Yannick Allusse
*/
class _GPUCV_HARDWARE_EXPORT_CLASS CL_Profiler
	:public CL_Options
{
public:
#if _GPUCV_PROFILE
	typedef SG_TRC::CL_TimerVal TracerType;		//!< Typedef of the class used to profile time.
#endif
	/**	\brief Default constructor */
	CL_Profiler(const std::string _className);

	/**	\brief Destructor */
	~CL_Profiler();

	virtual std::string LogException(void)const;
	virtual std::ostringstream & operator << (std::ostringstream & _stream)const;



public:
	/**	\brief Return Class name.
	*/
	const std::string & GetClassName(void)const{return m_className;}
protected:
	/**	\brief Define class name, useful for debugging and profiling
	*/
	void SetClassName(const std::string _name){m_className=_name;}



//#if _GPUCV_PROFILE_CLASS
#if _GPUCV_PROFILE

	/**	\brief Return a pointer to the corresponding function in profiling base.
	*	\param _fctName => function name.
	*	\return SG_TRC::CL_FUNCT_TRC<TracerType> * Correponding profiling function object.
	*	\note _GPUCV_PROFILE_CLASS must be set to  to enable this function.
	*/
	SG_TRC::CL_FUNCT_TRC<TracerType> * AddGetFunct(std::string _fctName);

	/**	\brief Return a pointer to the corresponding class in profiling base.
	*	\param _Name => class name.
	*	\return SG_TRC::CL_CLASS_TRACER<TracerType> * Correponding profiling class object.
	*	\note _GPUCV_PROFILE_CLASS must be set to 1 to enable this function.
	*/
	SG_TRC::CL_CLASS_TRACER<TracerType>	* GetClassTracer(std::string _Name="");
private:

	SG_TRC::CL_CLASS_TRACER<TracerType>	* m_classTracer;	//!< Pointer to the real profiling class object in database.
	//#endif


public:
	/**	\brief Save all the profiling records into the given file as XML format.
	*	\param _filename => XML target filename.
	*	\return True if success, else false.
	*	\note _GPUCV_PROFILE must be set to 1 to enable this function.
	*/
	static
		bool Savebench(std::string _filename);

	/**	\brief Load some previously saved profiling records fromo the given XML file.
	*	\param _filename => XML source filename.
	*	\return True if success, else false.
	*	\note _GPUCV_PROFILE must be set to 1 to enable this function.
	*/
	static
		bool Loadbench(std::string _filename);

	/**	\brief Get the actual SG_TRC::TTCL_APPLI_TRACER (profiling manager).
	*	\return SG_TRC::TTCL_APPLI_TRACER<SG_TRC::CL_TimerVal> & to the main profiling manager.
	*	\note _GPUCV_PROFILE must be set to 1 to enable this function.
	*/
	static
		SG_TRC::TTCL_APPLI_TRACER<SG_TRC::CL_TimerVal> & GetTimeTracer();

private:
	static SG_TRC::TTCL_APPLI_TRACER<SG_TRC::CL_TimerVal> & m_timeTracer;	//!< Pointer to the main profiling manager.
#endif

protected:
	std::string					m_className;		//!< String to describe the class type.
};

_GPUCV_HARDWARE_EXPORT std::ostringstream & operator << (std::ostringstream & _stream, const CL_Profiler & TexDsc);

/** @ingroup GPUCV_MACRO_LOGGING_GRP.
@{
\name Class debugging
*/
/**	\brief Create a string containing the function name.
*	\sa CLASS_FCT_SET_NAME_TPL(), CLASS_FCT_SET_NAME_TPL_STR(), GPUCV_GET_FCT_NAME(), GPUCV_FUNCNAME().
*/
#define CLASS_FCT_SET_NAME(FCTNAME) GPUCV_FUNCNAME_STATIC(FCTNAME)

/**	\brief Create a string containing the function name. The name will include a template argument "<Tpl>".
*	\param	TPL => object type.
*	\param	FCTNAME => function name.
*	\sa CLASS_FCT_SET_NAME(), CLASS_FCT_SET_NAME_TPL_STR(), GPUCV_GET_FCT_NAME(), GPUCV_FUNCNAME().
*/
#define CLASS_FCT_SET_NAME_TPL(TPL,FCTNAME)GPUCV_FUNCNAME_TEMPLATE(TPL,FCTNAME)

/**	\brief Create a string containing the function name. The name will include a template argument "<Tpl>" given as a string.
*	\param	TPLSTR => object type as a string.
*	\param	FCTNAME => function name.
*	\sa CLASS_FCT_SET_NAME(), CLASS_FCT_SET_NAME_TPL(), GPUCV_GET_FCT_NAME(), GPUCV_FUNCNAME().
*/
#define CLASS_FCT_SET_NAME_TPL_STR(TPLSTR,FCTNAME)GPUCV_FUNCNAME_TEMPLATE_STR(TPLSTR,FCTNAME)

/** \brief Output debugging message from class function as :"ClassID : ClassType : ClassFunction => Message".*/
#define CLASS_LOG(FLAG, TYPE_STR, MSG)	GPUCV_LOG(FLAG, TYPE_STR, GetValStr() << "=" << m_className << "::"<< GPUCV_GET_FCT_NAME() << "=>" << MSG)

/** \brief Output debugging message from class function as :"ClassID : ClassType : ClassFunction => Message".*/
#define CLASS_MEMORY_LOG(MSG)	GPUCV_LOG(GpuCVSettings::GPUCV_SETTINGS_DEBUG_MEMORY, "MEM", GetValStr() << "=" << m_className << "::"<< GPUCV_GET_FCT_NAME() << "=>" << MSG)

/** \brief Output debugging message from class function as :"ClassID : ClassType : ClassFunction => Message".*/
#define CLASS_DEBUG(MSG)	GPUCV_DEBUG(GetValStr() << "=" << m_className << "::"<< GPUCV_GET_FCT_NAME() << "=>" << MSG)

/** \brief Output warning message from class function as :"ClassID : ClassType : ClassFunction => Message".*/
#define CLASS_WARNING(MSG)	GPUCV_WARNING(GetValStr() << "=" <<m_className << "::"<< GPUCV_GET_FCT_NAME() << "=>" << MSG)

/** \brief Output notice message from class function as :"ClassID : ClassType : ClassFunction => Message".*/
#define CLASS_NOTICE(MSG)	GPUCV_NOTICE(GetValStr() << "=" <<m_className << "::"<< GPUCV_GET_FCT_NAME() << "=>" << MSG)

/** \brief Output error message from class function as :"ClassID : ClassType : ClassFunction => Message".*/
#define CLASS_ERROR(MSG)	GPUCV_ERROR(GetValStr() << "=" <<m_className << "::"<< GPUCV_GET_FCT_NAME() << "=>" << MSG)

/** @}*/ //GPUCV_MACRO_LOGGING_GRP.

#if _GPUCV_PROFILE_CLASS

//! \todo For Class profiling, concatenating functions and class name may take some time, better to find another solutions...
#define CLASS_FCT_PROF_CREATE_START()\
	SG_TRC::CL_FUNCT_TRC<CL_Profiler::TracerType> *CurFct = NULL;\
	const char * cLocalName=NULL;\
	if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_PROFILING_CLASS))\
	{\
		std::string Name = GetClassName();\
		Name +="::";\
		Name +=GPUCV_GET_FCT_NAME();\
		cLocalName= Name.data();\
	}\
	SG_TRC::CL_TEMP_TRACER<CL_Profiler::TracerType> Tracer1(cLocalName, NULL);

/*
#define _PROFILE_FCT_CREATE(NAME) \
	SG_TRC::CL_FUNCT_TRC<CL_Profiler::TracerType> *CurFct=NULL;\
	if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_PROFILING_CLASS))\
	{\
		CurFct=CL_Profiler::GetTimeTracer().AddFunct(NAME);\
	}

#define _PROFILE_FCT_START()\
	SG_TRC::CL_TEMP_TRACER<CL_Profiler::TracerType> Tracer1(CurFct, NULL);\


#define _PROFILE_FCT_CREATE_START(NAME)\
	_PROFILE_FCT_CREATE(NAME);\
	_PROFILE_FCT_START();

#define _PROFILE_FCT_BENCH(NAME, FCT){\
	_PROFILE_FCT_CREATE(NAME);\
	_PROFILE_FCT_START();\
	FCT;

*/

#else
//#define CLASS_FCT_PROF_CREATE()
#define CLASS_FCT_PROF_CREATE_START()
//#define _PROFILE_FCT_CREATE(NAME)
//#define _PROFILE_FCT_START()
//#define _PROFILE_FCT_CREATE_START(NAME)
//#define _PROFILE_FCT_BENCH(NAME, FCT){FCT;}
#endif

/** \brief Macro use to benchmark any function.
\param NAME -> name of function benchmarked.
\param FCT -> source code to call and benchmark.
\param RSLT_IMG -> pointer to output image, use to register size, can be NULL.
\param GPU_FLAG -> if true, function glFinish() will be called before and after profiling, on parameter 'type' "GPUCV" will be added, else 'type' is "OPENCV"
\param PROF_FLAG -> if true, enable runtime profiling of GPUCV operators.
\note _GPUCV_PROFILE_CLASS must be set to 1, so the benchmarking functions are compiled.
\note GpuCVSettings::GPUCV_SETTINGS_PROFILING must be set to 1, so the benchmarking functions are called.
*/

#if _GPUCV_PROFILE
#if 1//new code

#define GPUCV_PROFILE_CURRENT_FCT(NAME, RSLT_IMG, GPU_FLAG, PROF_FLAG)\
	SG_TRC::CL_TRACE_BASE_PARAMS *_PROFILE_PARAMS = NULL;\
	if(GetGpuCVSettings()->GetOption(PROF_FLAG))\
	{\
		_PROFILE_PARAMS = new SG_TRC::CL_TRACE_BASE_PARAMS();\
		_PROFILE_PARAMS->AddChar("type", (GPU_FLAG)?"GPUCV":"OPENCV");\
		if(RSLT_IMG)\
	{\
		std::string Size="=";\
		Size+= SGE::ToCharStr(GetWidth(RSLT_IMG)) + "*";\
		Size+= SGE::ToCharStr(GetHeight(RSLT_IMG));\
		_PROFILE_PARAMS->AddChar("size", Size.data());\
		Size= SGE::ToCharStr(GetCVDepth(RSLT_IMG));\
		_PROFILE_PARAMS->AddChar("depth", Size.data());\
		Size= SGE::ToCharStr(GetnChannels(RSLT_IMG));\
		_PROFILE_PARAMS->AddChar("nChannels", Size.data());\
	}\
	if(GPU_FLAG)glFinish();\
}\
	SG_TRC::CL_TEMP_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> Tracer (NAME, _PROFILE_PARAMS);\
	Tracer.SetOpenGL(GPU_FLAG);

		//_PROFILE_PARAMS->AddChar("line", SGE::ToCharStr( __LINE__).data());\
		//_PROFILE_PARAMS->AddChar("file",__FILE__);\

#define GPUCV_PROFILE_FCT(NAME, FCT, RSLT_IMG, GPU_FLAG,PROF_FLAG){\
	GPUCV_PROFILE_CURRENT_FCT (NAME, RSLT_IMG, GPU_FLAG,PROF_FLAG);\
	FCT;\
}
#else//old code
#define GPUCV_PROFILE_CURRENT_FCT(NAME, RSLT_IMG, GPU_FLAG, PROF_FLAG)\
	SG_TRC::CL_FUNCT_TRC<SG_TRC::SG_TRC_Default_Trc_Type> *CurFct##__LINE__ = NULL;\
	SG_TRC::CL_TRACE_BASE_PARAMS *_PROFILE_PARAMS = NULL;\
	if(GetGpuCVSettings()->GetOption(PROF_FLAG))\
{\
	CurFct##__LINE__=\
	CL_Profiler::GetTimeTracer().AddFunct(NAME);\
	_PROFILE_PARAMS = new SG_TRC::CL_TRACE_BASE_PARAMS();\
	_PROFILE_PARAMS->AddChar("line", SGE::ToCharStr( __LINE__).data());\
	_PROFILE_PARAMS->AddChar("file",__FILE__);\
	_PROFILE_PARAMS->AddChar("type", (GPU_FLAG)?"GPUCV":"OPENCV");\
	if(RSLT_IMG)\
{\
	std::string Size="=";\
	Size+= SGE::ToCharStr(GetWidth(RSLT_IMG)) + "*";\
	Size+= SGE::ToCharStr(GetHeight(RSLT_IMG));\
	_PROFILE_PARAMS->AddChar("size", Size.data());\
	Size= SGE::ToCharStr(GetCVDepth(RSLT_IMG));\
	_PROFILE_PARAMS->AddChar("depth", Size.data());\
	Size= SGE::ToCharStr(GetnChannels(RSLT_IMG));\
	_PROFILE_PARAMS->AddChar("nChannels", Size.data());\
}\
	if(GPU_FLAG)glFinish();\
}\
	SG_TRC::CL_TEMP_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> Tracer (CurFct##__LINE__, _PROFILE_PARAMS);\
	Tracer.SetOpenGL(GPU_FLAG);


#define GPUCV_PROFILE_FCT(NAME, FCT, RSLT_IMG, GPU_FLAG,PROF_FLAG){\
	GPUCV_PROFILE_CURRENT_FCT (NAME, RSLT_IMG, GPU_FLAG,PROF_FLAG);\
	FCT;\
}

#endif
#else
#define GPUCV_PROFILE_FCT(NAME, FCT, RSLT_IMG, GPU_FLAG,PROF_FLAG){FCT;}
#define GPUCV_PROFILE_CURRENT_FCT(NAME, RSLT_IMG, GPU_FLAG,PROF_FLAG)
#endif


}//namespace GCV
#endif
