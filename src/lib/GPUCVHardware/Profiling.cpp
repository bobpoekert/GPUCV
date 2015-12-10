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

#include "StdAfx.h"
#include "GPUCVHardware/Profiling.h"
#include "GPUCVHardware/hardware.h"


namespace GCV{
#if _GPUCV_PROFILE_CLASS
//	SG_TRC::CL_CLASS_TRACER<CL_Profiler::TracerType>	* CL_Profiler::m_classTracer = NULL;
#endif

//=======================================================
CL_Profiler::CL_Profiler(const std::string _className)
: CL_Options()
#if _GPUCV_PROFILE_CLASS
,m_classTracer(NULL)
#endif
{
	m_className = _className;

#if _GPUCV_PROFILE_CLASS
	GetClassTracer(_className);
#endif
	//ask for default options for this object and affect them to current obj
	CL_Options::OPTION_TYPE LocalOption=0;
	if(GetGpuCVSettings()->GetDefaultOption(m_className, LocalOption))
		ForceAllOptions(LocalOption);
}
//=======================================================
CL_Profiler::~CL_Profiler()
{
#if _GPUCV_PROFILE
	//can not be destroyed here
	/*
	*/
#endif
}
//=======================================================
#if _GPUCV_PROFILE
/*static*/
bool CL_Profiler::Savebench(std::string _filename)
{
	std::string InternProfileName = GetGpuCVSettings()->GetShaderPath();
	InternProfileName += "./";
	InternProfileName += _filename;
	if(GetTimeTracer().XMLSaveToFile(InternProfileName))
		return true;
	else
		return false;
}
//=======================================================
/*static*/
bool CL_Profiler::Loadbench(std::string _filename)
{
	std::string InternProfileName = GetGpuCVSettings()->GetShaderPath();
	InternProfileName += "./";
	InternProfileName += _filename;

	GetTimeTracer().XMLLoadFromFile(InternProfileName);
	//	m_timeTracer->SetConsoleOutput(false);
	return true;
}
//=======================================================
SG_TRC::TTCL_APPLI_TRACER<SG_TRC::CL_TimerVal> & CL_Profiler::GetTimeTracer()
{
	return SG_TRC::TTCL_APPLI_TRACER<TracerType>::Instance();
}
#endif

//=======================================================
#if _GPUCV_PROFILE_CLASS
/*static*/
SG_TRC::CL_CLASS_TRACER<CL_Profiler::TracerType>	*
CL_Profiler::GetClassTracer(std::string _Name/*=""*/)
{
#if _GPUCV_PROFILE
	if(m_classTracer==NULL)
	{
	//	SGPT::CL_MutexPtr<SG_TRC::TTCL_APPLI_TRACER<TracerType>::TplClassMngr* > MutexClassMngr (SG_TRC::CreateMainTracer<SG_TRC::CL_TimerVal>()->GetClassManagerMutex(), "MutexClassMngr", __FILE__, __LINE__, true);

		MUTEXPTR_CREATE(SG_TRC::TTCL_APPLI_TRACER<TracerType>::TplClassMngr* , MutexClassMngr, GetTimeTracer().GetClassManagerMutex());
		m_classTracer = MutexClassMngr->Add(_Name);
		m_classTracer->SetParentAppliTracer(&GetTimeTracer());
	}
#endif
	return m_classTracer;
}
//=======================================================
/*static*/
SG_TRC::CL_FUNCT_TRC<CL_Profiler::TracerType> *
CL_Profiler::AddGetFunct(std::string _fctName)
{
	return m_classTracer->AddFunct(_fctName);
}
//=======================================================
#endif
//=======================================================
std::string CL_Profiler::LogException(void)const
{
	std::string Msg;
	Msg = "Exception was raised in object: ";
	Msg+= m_className;
	Msg+= "\n";
	return Msg;
}
//=======================================================
std::ostringstream & CL_Profiler::operator << (std::ostringstream & _stream)const
{
	_stream << std::endl;
	_stream << std::endl;
	_stream << LogIndent() << "======================================" << std::endl;
	_stream << LogIndent() << "CL_Profiler==============" << std::endl;
	_stream << LogIndent() << "..." << std::endl;
	_stream << LogIndent() << "CL_Profiler==============" << std::endl;
	return _stream;
}
//==================================================
std::ostringstream & operator << (std::ostringstream & _stream, const CL_Profiler & _Prof)
{
	return _Prof.operator << (_stream);
}

}//namespace GCV
