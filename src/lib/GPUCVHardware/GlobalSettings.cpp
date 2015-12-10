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
#include <GPUCVHardware/GlobalSettings.h>
#include <GPUCVHardware/revision.h>
#include <GPUCVHardware/ToolsGL.h>
namespace GCV{

//=================================================
GpuCVSettings::GpuCVSettings()
:CL_Options()
,m_initDone(false)
#if _GPUCV_GL_USE_GLUT
,m_glutDebug(false)
#endif
,m_defaultOptionManager(NULL)
{
	//start logger
	SetLoggingOutput(LoggingToConsole);

	//Default winfows size
	m_windowSize[0] = 512;
	m_windowSize[1] = 512;

	//set default parameters
#if _GPUCV_DEBUG_MODE
	CL_Options::SetOption(GPUCV_SETTINGS_GL_ERROR_CHECK
		| GPUCV_SETTINGS_GLOBAL_DEBUG
		| GPUCV_SETTINGS_GLOBAL_ERROR
		| GPUCV_SETTINGS_GLOBAL_WARNING
		| GPUCV_SETTINGS_GLOBAL_NOTICE
		| GPUCV_SETTINGS_CHECK_SHADER_UPDATE
		, true);
#elif defined(_DEBUG)
	CL_Options::SetOption(GPUCV_SETTINGS_GL_ERROR_CHECK
		| GPUCV_SETTINGS_GLOBAL_DEBUG
		| GPUCV_SETTINGS_GLOBAL_ERROR
		| GPUCV_SETTINGS_GLOBAL_WARNING
		| GPUCV_SETTINGS_GLOBAL_NOTICE
		| GPUCV_SETTINGS_CHECK_SHADER_UPDATE
		|GPUCV_SETTINGS_DEBUG_DATA_TRANSFER
		, true);
#else
	CL_Options::SetOption(
		GPUCV_SETTINGS_GLOBAL_NOTICE
		|GPUCV_SETTINGS_CHECK_IMAGE_ATTRIBS
		, true);
#endif





	//see http://tortoisesvn.net/docs/release/TortoiseSVN_fr/tsvn-subwcrev-example.html for details
	//fields filled by tortoiseSVN and SubWCRev program


	m_Version  = _GPUCV_VERSION_MAJOR;
	m_Revision = SGE::ToCharStr(_GPUCV_REVISION_VAL);
	m_RevisionDate     = _GPUCV_REVISION_DATE;
	m_URLSvn	= _GPUCV_REVISION_URL;
	m_URLHome   = _GPUCV_WEB_URL;
	//generate errors if the
#if _GPUCV_REVISION_MODIFIED
#error Source code has been modified, please commit before updating revision!
#endif
	//char *Modifiee = "$WCMODS?Modifi�e:Pas modifi�e$";
	//char *PlageRev = "$WCRANGE$";
	//char *Melangee  = "$WCMIXED?CdT avec R�visions m�lang�es:Pas m�lang�e$";

	/*if(m_revision.find("WCREV"))
	{
	GPUCV_NOTICE("Current version of GpuCV need to be recompiled with SubWCRev before the release");
	}
	*/
}
//=================================================
GpuCVSettings::~GpuCVSettings()
{
#if _GPUCV_GL_USE_GLUT
	//destroy all OpenGL windows
	for(unsigned int i = 0; i < m_GLutWindowsList.size(); i++)
	{
		glutDestroyWindow(m_GLutWindowsList[i]);
	}
#endif
}
//=================================================
void GpuCVSettings::SetShaderPath(const std::string _path)
{
	m_ShaderPath = _path;

	//check that last character is '/' or '\'
#if _WINDOWS
	if(_path.find_last_of("\\") != _path.size() -1)
		m_ShaderPath = _path + "\\";
#else //for LINUX based OS
	if(_path.find_last_of("/") != _path.size() -1)
		m_ShaderPath =  _path + "/";
#endif
	SGE::ReformatFilePath(GetShaderPath());
}
//=================================================
#define GPUCV_SETTINGS_OUTPUT_OPTIONS(OPTION)\
	TempString+= lineStart;\
	TempString+= (GetOption(OPTION))?"1":"0";\
	TempString+= lineMiddle;\
	TempString+= #OPTION;\
	TempString+= "(";\
	TempString+= SGE::ToCharStr(OPTION);\
	TempString+= ")";\
	TempString+= lineEnd;

std::string  GpuCVSettings::GetOptionsDescription(std::string lineStart/*=""*/, std::string lineMiddle/*=""*/,std::string lineEnd/*=""*/)
{
	std::string TempString = lineStart;
	TempString+= "GpuCVSettings values:";
	TempString+= lineEnd;

	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_GL_ERROR_CHECK);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_GL_ERROR_RISE_EXCEPTION);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_LOAD_ALL_SHADERS);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_USE_OPENCV);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_GLOBAL_DEBUG);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_GLOBAL_ERROR);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_GLOBAL_WARNING);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_GLOBAL_NOTICE);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_CHECK_SHADER_UPDATE);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_CHECK_IMAGE_ATTRIBS);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_FILTER_DEBUG);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_FILTER_SIMULATE);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_DEBUG_DATA_TRANSFER);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_SYNCHRONIZE_ON_ERROR);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_PROFILING);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_PROFILING_TRANSFER);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_PROFILING_OPER);
	GPUCV_SETTINGS_OUTPUT_OPTIONS(GPUCV_SETTINGS_PROFILING_CLASS);
	return TempString;
}
//=================================================
void GpuCVSettings::SetOption(OPTION_TYPE _opt, bool val)
{
	CL_Options::SetOption(_opt, val);
	//GPUCV_NOTICE(GetGpuCVSettings()->GetOptionsDescription());
}
//=================================================
std::string GpuCVSettings::GetVersionDescription()
{
	std::string TempDesc = "Major version: \t";
	TempDesc+= GetVersion();
	TempDesc+= "\nMinor revision: \t";
	TempDesc+= GetRevision();
	TempDesc+= "\nRelease date: \t";
	TempDesc+=  GetRevisionDate();
	TempDesc+= "\nWeb URL: \t";
	TempDesc+= GetURLHome();

	return TempDesc;
}
//=================================================
std::string GpuCVSettings::ExportToHTMLTable()
{
	std::string TempStr = "<H2>GpuCV description:</H2>";
	TempStr += HTML_OPEN_TABLE;
	TempStr += HTML_OPEN_ROW + std::string("<td width='20%'>Version</td>") + HTML_CELL(GetVersion())+ HTML_CLOSE_ROW;
	TempStr += HTML_OPEN_ROW + HTML_CELL("Revision") + HTML_CELL(GetRevision())+ HTML_CLOSE_ROW;
	TempStr += HTML_OPEN_ROW + HTML_CELL("Date") + HTML_CELL( GetRevisionDate())+ HTML_CLOSE_ROW;
	TempStr += HTML_OPEN_ROW + HTML_CELL("Build mode");
#if _GPUCV_DEBUG_MODE
	TempStr += HTML_CELL("<b>DEBUG, this might affect benchmarks results.</b>")+ HTML_CLOSE_ROW;
#else
	TempStr += HTML_CELL("RELEASE")+ HTML_CLOSE_ROW;
#endif
	std::string BDate = __DATE__;
	BDate+= " - ";
	BDate+= __TIME__;
	TempStr += HTML_OPEN_ROW + HTML_CELL("Build date") + HTML_CELL(BDate)+ HTML_CLOSE_ROW;
	//	TempStr += HTML_OPEN_ROW + HTML_CELL("Revision") + HTML_CELL(GetRevision())+ HTML_CLOSE_ROW;
	TempStr += HTML_OPEN_ROW + HTML_CELL("Options") + HTML_CELL(GetOptionsDescription("", " - ", "<br>"))+ HTML_CLOSE_ROW;
	TempStr += HTML_OPEN_ROW + HTML_CELL("Operating system");
#if defined(_WINDOWS)
	TempStr += HTML_CELL("Ms Windows XP");
#elif defined(_LINUX)
	TempStr += HTML_CELL("LINUX based OS");
#elif defined(_MACOS)
	TempStr += HTML_CELL("Mac OS ?");
#endif
	TempStr += HTML_CLOSE_ROW;
	TempStr += HTML_CLOSE_TABLE;
	return TempStr;
}
//=================================================
const unsigned int * GpuCVSettings::GetWindowSize()const
{
	return m_windowSize;
}
//=================================================
void GpuCVSettings::SetWindowSize(unsigned int _width, unsigned int _height)
{
	m_windowSize[0] = _width;
	m_windowSize[1] = _height;
}
//=================================================
const CL_Profiler	* GpuCVSettings::GetLastExceptionObj()const
{
	if(m_vExceptionParentObjects.size()>0)
		return m_vExceptionParentObjects[m_vExceptionParentObjects.size()-1];
	else
		return NULL;
}
//=================================================
const CL_Profiler	* GpuCVSettings::PushExceptionObj(const CL_Profiler	* _obj)
{
	m_vExceptionParentObjects.push_back(_obj);
	return _obj;
	//m_exceptionParentObject = _obj;
}
//=================================================
const CL_Profiler	* GpuCVSettings::PopExceptionObj()
{
	const CL_Profiler	* lastObj = GetLastExceptionObj();
	if(m_vExceptionParentObjects.size())
		m_vExceptionParentObjects.pop_back();
	return lastObj;
	//m_exceptionParentObject = _obj;
}
//=================================================
#ifdef _WINDOWS
std::ostringstream::_Mystr GpuCVSettings::GetExceptionObjectTree(/*, int nbr = -1*/)const
{
	std::ostringstream _stream;
	for (int i = (int)m_vExceptionParentObjects.size()-1; i >=0; i--)
	{
		GPUCV_NOTICE("========Object[" << i << "]" );
		LogIndentIncrease();
		LogIndentIncrease();
		GPUCV_NOTICE("Object debugging disabled..., GpuCVSettings::GetExceptionObjectTree @ GlobalSettings.cpp:l.254" );
		//m_vExceptionParentObjects[i]->operator << (_stream);
		LogIndentDecrease();
		LogIndentDecrease();
	}
	return _stream.str();
}
#else
std::string GpuCVSettings::GetExceptionObjectTree(/*, int nbr = -1*/)const
{
	return "Warning: GpuCVSettings::GetExceptionObjectTree() is not working yet for Linux!";
}
#endif
#if _GPUCV_GL_USE_GLUT

//=================================================
void GpuCVSettings::EnableGlutDebug()
{
	m_glutDebug = true;
}
//=================================================
bool GpuCVSettings::GetGlutDebug()const
{
	return 	m_glutDebug;
}
//=================================================
void GpuCVSettings :: AddGlutWindowsID(int _ID)
{
	m_GLutWindowsList.push_back(_ID);
}
#endif
//=================================================
bool GpuCVSettings::GetDefaultOption(std::string _objName, CL_Options::OPTION_TYPE & _opt)
{
	CL_OptionStorage * CurObj = m_defaultOptionManager.Find(_objName);
	if(CurObj)
	{
		_opt = CurObj->GetAllOptions();
		return true;
	}
	else
	{
		_opt = 0;
		return false;
	}
}
//=================================================
void GpuCVSettings::SetDefaultOption(std::string _objName, CL_Options::OPTION_TYPE &_opt,  bool _val)
{
	CL_OptionStorage * CurObj = m_defaultOptionManager.Add(_objName);
	if(CurObj)
	{
		CurObj->SetOption(_opt, _val);
	}
}
//=================================================
void GpuCVSettings::ForceAllOptions(std::string _objName, CL_Options::OPTION_TYPE &_opt)
{
	CL_OptionStorage * CurObj = m_defaultOptionManager.Add(_objName);
	if(CurObj)
	{
		CurObj->ForceAllOptions(_opt);
	}
}
//=================================================
const ImplementationDescriptor* GpuCVSettings::RegisterNewImplementation(LibraryDescriptor* _pDllInfos)
{
	SG_Assert(_pDllInfos, "No input DLL informations");

	std::vector<ImplementationDescriptor*>::iterator iterImpl;
	ImplementationDescriptor* pNewImpl = NULL;
	//try to find if current DLL has already been registered
	for(iterImpl = m_vImplementations.begin(); iterImpl != m_vImplementations.end(); iterImpl++)
	{
		if((*iterImpl)->m_baseImplID == _pDllInfos->GetBaseImplementationID()) 
			if((*iterImpl)->m_strImplName == _pDllInfos->GetImplementationName())
			{
				//already done
				break;
			}
	}
	if(pNewImpl==NULL)
	{
		//no done yet...
		pNewImpl = new ImplementationDescriptor();
		pNewImpl->m_baseImplID	= _pDllInfos->GetBaseImplementationID();//
		pNewImpl->m_strImplName = _pDllInfos->GetImplementationName();//
		pNewImpl->m_dynImplID	=  m_vImplementations.size();
		m_vImplementations.push_back(pNewImpl);
	}

	//affect implementation to library
	_pDllInfos->SetImplementationDescriptor(pNewImpl);

	return pNewImpl;
}
//=================================================
const ImplementationDescriptor* GpuCVSettings::GetImplementation(const std::string _name)
{
	SG_Assert(_name!="", "Empty implementation name");

	std::vector<ImplementationDescriptor*>::iterator iterImpl;
	ImplementationDescriptor* pNewImpl = NULL;
	//try to find if current DLL has already been registered
	for(iterImpl = m_vImplementations.begin(); iterImpl != m_vImplementations.end(); iterImpl++)
	{
		if((*iterImpl)->m_strImplName == _name) 
			//already done
			return (*iterImpl);
	}
	return NULL;//not found
}
//=================================================	
GpuCVSettings * GetGpuCVSettings()
{
	static GpuCVSettings * settings = new GpuCVSettings();
	return settings;
}
//=================================================
}//namespace GCV
