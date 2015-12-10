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
#include <GPUCVSwitch/Cl_Dll.h>
#include <GPUCVSwitch/config.h>
#include <GPUCVSwitch/Cl_FctSw_Mngr.h>


namespace GCV{
/*===================================================
=====================================================
 CLASS DllDescriptor
=====================================================
===================================================*/
DllDescriptor::DllDescriptor(const std::string _Name)
: SGE::CL_XML_BASE_OBJ<std::string>(_Name, "dllfile")
{}
//===================================================
DllDescriptor::DllDescriptor(void)
: SGE::CL_XML_BASE_OBJ<std::string>("", "dllfile")
{}
//===================================================
DllDescriptor::~DllDescriptor()
{}
//===================================================
TiXmlElement*	
DllDescriptor::XMLLoad(TiXmlElement* _XML_Root, const std::string & _subTagName)
{
	TiXmlElement* pLocalNode = SGE::XMLGetElement(_XML_Root, _subTagName);
	if(pLocalNode)
	{
		SGE::XMLReadVal(pLocalNode, "release", m_Release);
		SGE::XMLReadVal(pLocalNode, "debug", m_Debug);
		SGE::XMLReadVal(pLocalNode, "arch", m_Arch);
		SGE::XMLReadVal(pLocalNode, "version", m_Version);
		SGE::XMLReadVal(pLocalNode, "os", m_Os);
		SGE::XMLReadVal(pLocalNode, "depends", m_Dependency);
	}
	return pLocalNode;
}
//===================================================
TiXmlElement*	
DllDescriptor::XMLSave(TiXmlElement* _XML_Root, const std::string & _subTagName)
{
	TiXmlElement* pLocalNode = _XML_Root;//SGE::XMLAddNewElement(_XML_Root, _subTagName);
	if(pLocalNode)
	{
		SGE::XMLWriteVal(pLocalNode, "release", m_Release);
		SGE::XMLWriteVal(pLocalNode, "debug", m_Debug);
		SGE::XMLWriteVal(pLocalNode, "arch", m_Arch);
		SGE::XMLWriteVal(pLocalNode, "version", m_Version);
		SGE::XMLWriteVal(pLocalNode, "os", m_Os);
		SGE::XMLWriteVal(pLocalNode, "depends", m_Dependency);
	}
	//_XML_Root->InsertEndChild(*pLocalNode);
	return _XML_Root;
}
/*===================================================
 =====================================================
 CLASS DllMod
 =====================================================
 ===================================================*/
DllMod::DllMod(const std::string _fileName, std::string _modPrefix)
:SGE::CL_XML_BASE_OBJ<std::string>(_fileName, "dll")
,m_pCurrentDllFile(NULL)
,m_HandleDLL(NULL)
,m_Prefix(_modPrefix)
,m_ModInfos(NULL)
,m_Enabled(true)
,m_LoadStatus(SG_TRC::EXEC_UNKNOWN)
{
	GPUCV_NOTICE//DEBUG
	("Create new module: "<< _fileName << "(pref:" <<  _modPrefix <<")");
}
//===================================================
DllMod::~DllMod()
{
	if (GetHandleDLL())
		SGE::DLLClose(GetHandleDLL());
}
//===================================================
/*virtual*/
SG_TRC::ExecState DllMod::Load()
{
	if(GetLoadStatus()== SG_TRC::EXEC_SUCCESS)
		return SG_TRC::EXEC_SUCCESS;

	GPUCV_NOTICE//DEBUG
		("Start loading module: "<< GetIDStr());// << (m_Prefix=="")?"":"pref:" <<  m_Prefix <<")");
	SGE::LoggerAutoIndent LocalIndent;
	SGE::LoggerAutoIndent LocalIndent2;//twice



	std::string strOsName	=GetOSName();
	std::string strArch		=GetArchName();

	//OpenCV version
	std::string strVersion=SGE::ToCharStr(CV_MAJOR_VERSION);
	strVersion += SGE::ToCharStr(CV_MINOR_VERSION);
	strVersion += SGE::ToCharStr(CV_SUBMINOR_VERSION);

	//parse all DLL file descriptors to find the best one
	std::vector<DllDescriptor*>::iterator IterDllFile;
	DllDescriptor* pCurFileDsc=NULL;
	for(IterDllFile = m_vecDLLFiles.begin(); IterDllFile != m_vecDLLFiles.end(); IterDllFile++)
	{
		pCurFileDsc = (*IterDllFile);
		if(pCurFileDsc==NULL) continue;

		//debug/release
		std::string strFilename;
#ifdef _DEBUG
		strFilename = (pCurFileDsc->GetDebug()!="")?pCurFileDsc->GetDebug():pCurFileDsc->GetRelease();
#else
		strFilename = (pCurFileDsc->GetRelease()!="")?pCurFileDsc->GetRelease():pCurFileDsc->GetDebug();
#endif
		if(strFilename=="")
		{
			GPUCV_ERROR("No filename found. Skipping dll file.");
			continue;
		}

		//check system OS:
		if((pCurFileDsc->GetOs()!="") && (pCurFileDsc->GetOs() != strOsName))
		{
			GPUCV_DEBUG(strFilename << "-> Wrong OS("<< pCurFileDsc->GetOs() << "). Skipping dll file.");
			continue;
		}
		//check system arch:
		if((pCurFileDsc->GetArch()!="") && (pCurFileDsc->GetArch() != strArch))
		{
			GPUCV_DEBUG(strFilename << "-> Wrong Arch("<< pCurFileDsc->GetArch() << "). Skipping dll file.");
			continue;
		}
		//check version?
		if((pCurFileDsc->GetVersion()!="") && (pCurFileDsc->GetVersion() != strVersion))
		{
			GPUCV_DEBUG(strFilename << "-> Wrong version("<< pCurFileDsc->GetVersion() << "). Skipping dll file.");
			continue;
		}

		
		

		//if a dependency is required, we try to load it first
		if(pCurFileDsc->GetDependency()!="")
		{
			std::string tmpSrc,tmpSrc2, currentDep;
			tmpSrc = pCurFileDsc->GetDependency();
			bool bDependenciesMatched = true;
			bool bSeparatorFound = false;
			do
			{
				bSeparatorFound = SGE::ParseSplitString(tmpSrc, currentDep, tmpSrc2, ",");
				tmpSrc= tmpSrc2;
				if(currentDep!="") 
				{
					LibHandleType pDependency = SGE::DLLLoad((char*)currentDep.data());//LoadLibrary(LibName);
					if(pDependency==NULL)
					{//not found, we can not load this plugins
						GPUCV_DEBUG(strFilename << "-> Required dependency '"<< currentDep << "' not found, skipping plugin.");
						bDependenciesMatched = false;
						break;
					}
					else
					{
						GPUCV_DEBUG(strFilename << "-> Required dependency '"<< currentDep << "' found, loading plugin.");
						SGE::DLLClose(pDependency);//close it..
					}
				}
			}
			while(bSeparatorFound);

			if(bDependenciesMatched==false)//try another library
				continue;
		}

		//try to load DLL
				//PSTR LibName = (PSTR)((char*)TmpName.data());
			SetHandleDLL(SGE::DLLLoad((char*)strFilename.data()));//LoadLibrary(LibName);
			if(GetHandleDLL())
			{
				//CL_BASE_OBJ_FILE::Load();
				GPUCV_NOTICE//DEBUG
					(strFilename << "-> Load module succeed.");
				ReadLibInformations();
				m_pCurrentDllFile = pCurFileDsc;
				pCurFileDsc->SetID(strFilename);
				SetLoadStatus(SG_TRC::EXEC_SUCCESS);
				return SG_TRC::EXEC_SUCCESS;
			}
			else
			{
				GPUCV_ERROR(strFilename << "-> Load module error.");
				continue;
			}
	}
	SetLoadStatus(SG_TRC::EXEC_FAILED_CRITICAL_ERROR);
	return SG_TRC::EXEC_FAILED_CRITICAL_ERROR;
}
//===================================================
switchFctStruct* DllMod::GetProcImpl(const std::string _functionName, bool usePrefix)
{
	
	if (!GetEnabled())
		return NULL;//module disabled

	if (!GetLoadStatus()<SG_TRC::EXEC_UNKNOWN)
		return NULL;//module could not load properly in the past

	
	if(!GetHandleDLL())//control and load dll if required
	{
		if(Load()< SG_TRC::EXEC_UNKNOWN)
		{
			if(m_pCurrentDllFile)
			{	GPUCV_ERROR("Could not load given dll:" << m_pCurrentDllFile->GetID());}
			else
			{	GPUCV_ERROR("Could not load given dll:" << GetIDStr());}
			
			return NULL;
		}
	}

	//calculate function name to find
	std::string TmpName;
	if(usePrefix)
	{
		//test if prefix is not here already...
		if(_functionName.find(m_Prefix)==0)
		{//prefix already here
			TmpName+=_functionName;
		}
		else if (_functionName.find("cv")==0)
		{
			TmpName=m_Prefix;
			TmpName+=_functionName.substr(2, _functionName.size());
		}
		else
		{
			TmpName=m_Prefix;
			TmpName+=_functionName;
		}
	}
	else
		TmpName=_functionName;


	//look for function into library
	TpFunctAdd _FctPtr = SGE::DLLGetProcAddress<TpFunctAdd>(GetHandleDLL(), TmpName.c_str());
	if(_FctPtr)
	{
		GPUCV_DEBUG
			(GetID() << "=> Fct '"<< TmpName << "' found");

		//\todo implementation specification should be more generic, register implementation?
		switchFctStruct * newFctIplm = new switchFctStruct();
#if 0//DEPRECATED
		if(m_modInfos->m_technoUsed=="GLSL")
			newFctIplm->m_ImplID = GPUCV_IMPL_GLSL;
		else if(m_modInfos->m_technoUsed=="CUDA")
			newFctIplm->m_ImplID = GPUCV_IMPL_CUDA;
		else if(m_modInfos->m_technoUsed=="OPENCV")
			newFctIplm->m_ImplID = GPUCV_IMPL_OPENCV;
		else
			newFctIplm->m_ImplID = GPUCV_IMPL_OPENCV;//GPUCV_IMPL_OTHER;
#endif
		newFctIplm->m_Implementation	= GetModInfos()->GetImplementationDescriptor();
		newFctIplm->m_Library			= GetModInfos();
		newFctIplm->m_ImplPtr			= _FctPtr;
		newFctIplm->m_AmntOfTimeSaved	= 0;
		newFctIplm->m_counter			= 0;
		newFctIplm->m_FctTracer			= NULL;
		newFctIplm->m_HrdPrf			= GCV::GenericGPU::HRD_PRF_0;//must be change...??using m_modInfos?
		newFctIplm->m_UseGpu			= GetModInfos()->GetUseGpu();

		//link with tracer obj
		//<!get ptr to class Impl
		{//using mutex lock for the manager
#if 0	
			GET_CLASSMNGR_MUTEX(MutexClassMngr);
			SG_TRC::CL_CLASS_TRACER<SG_TRC::CL_TimerVal>* ImplClass = MutexClassMngr->Add(newFctIplm->m_Implementation->m_strImplName);//m_modInfos->m_technoUsed);
			
			//<!get ptr to fct tracer
			if(newFctIplm->m_FctTracer==NULL && ImplClass)
				newFctIplm->m_FctTracer = ImplClass->AddFunct(_functionName);

#else//we now use the function manager
	
			GET_FCTMNGR_MUTEX(MutexFctMngr);
			newFctIplm->m_FctTracer = MutexFctMngr->Add(_functionName);
#endif
		}
		//=================
		return newFctIplm;
	}
	else 
	{
		GPUCV_DEBUG("Could not found function '"<< TmpName <<"' from '"<< m_pCurrentDllFile->GetID() << "'(" << GetHandleDLL() <<").");
		return NULL;
	}
}
//===================================================
const LibraryDescriptor	 * DllMod::ReadLibInformations()
{
	typedef LibraryDescriptor* (*TypeDLL_modGetLibraryDescriptor)(void); 
	
	//retrieve Library informations
	TypeDLL_modGetLibraryDescriptor _FctPtr =  SGE::DLLGetProcAddress<TypeDLL_modGetLibraryDescriptor>(GetHandleDLL(), "modGetLibraryDescriptor");
	if(_FctPtr)
	{
		m_ModInfos = _FctPtr();
	}
	else
	{//assume no function have been define so we are loading OpenCV DLLs
		m_ModInfos= new LibraryDescriptor();
		m_ModInfos->SetVersionMajor(SGE::ToCharStr(CV_MAJOR_VERSION));
		m_ModInfos->SetVersionMinor(SGE::ToCharStr(CV_MINOR_VERSION));

//		m_modInfos->m_dllName = m_pCurrentDllFile->GetIDStr();
		m_ModInfos->SetImplementationName(GetStrImplemtationID(GPUCV_IMPL_OPENCV));
		m_ModInfos->SetBaseImplementationID(GPUCV_IMPL_OPENCV);

		m_ModInfos->SetStartColor(GPUCV_IMPL_OPENCV_COLOR_START);
		m_ModInfos->SetStopColor(GPUCV_IMPL_OPENCV_COLOR_STOP);


		m_ModInfos->SetUseGpu(false);
	}

	//Register dll implementation into GpuCV core global settings.
	GetGpuCVSettings()->RegisterNewImplementation(m_ModInfos);

#if 0
	//then init SugoiTracer singletons
	typedef void (*TypeDLL_RegisterTracerSingleton)(SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> *, SG_TRC::CL_TRACING_EVENT_LIST *); 
	TypeDLL_RegisterTracerSingleton _FctPtrRegister =  SGE::DLLGetProcAddress<TypeDLL_RegisterTracerSingleton>(m_handleDLL, "cvg_RegisterTracerSingletons");
	if(_FctPtrRegister)
	{
		_FctPtrRegister(&SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance(), &SG_TRC::CL_TRACING_EVENT_LIST::Instance());
	}
	else
	{//assume no function have been define to use benchmarks inside this library
	}
#endif
	return m_ModInfos;
}
//===================================================
/*virtual*/
TiXmlElement*	DllMod::XMLLoad(TiXmlElement* _XML_Root)
{
	SGE::XMLReadVal(_XML_Root, "prefix", m_Prefix);
	int tmpBool;
	SGE::XMLReadVal(_XML_Root, "enabled", tmpBool);
	SetEnabled( (tmpBool==1)? true:false );

	SGE::XMLLoadVector(_XML_Root, m_vecDLLFiles, "dllfiles", "dllfile");
	Load();
	return _XML_Root;
}
//===================================================
/*virtual*/
TiXmlElement*	DllMod::XMLSave(TiXmlElement* _XML_Root)
{
	SGE::XMLWriteVal(_XML_Root, "prefix", m_Prefix);
	int tmpBool = GetEnabled();
	SGE::XMLWriteVal(_XML_Root, "enabled", tmpBool);

	SGE::XMLSaveVector(_XML_Root, m_vecDLLFiles, "dllfiles", "dllfile");
	return _XML_Root;
}
/*===================================================
 =====================================================
 CLASS DllManager
 =====================================================
 ===================================================*/
	DllManager::DllManager()
:	SGE::CL_XML_MNGR<DllMod, std::string>(NULL, "dllMngr", "dll", "id")
	,CL_Singleton<DllManager>()
	,m_strXMLFilename("gcv_dlls.xml")//default file
{
	if(!XMLLoadFromFile(m_strXMLFilename))
	{
		std::string NewPath = GetGpuCVSettings()->GetShaderPath();
		NewPath+=m_strXMLFilename;
		GPUCV_WARNING("Trying a new path:"<<NewPath);
		if(!XMLLoadFromFile(NewPath))
		{
			GPUCV_ERROR("File "<< NewPath << " not found, no DLL will be loaded for switching!");
		}
		else
		{
			m_strXMLFilename = NewPath;
			NewPath+="2.xml";
			XMLSaveToFile(NewPath);
		}
	}
}
//===================================================
DllManager::~DllManager()
{
	XMLSaveToFile(m_strXMLFilename);
}
//===================================================
#if 0
DllMod::TpFunctObj DllManager::GetProcAddress(const std::string & _functionName, bool usePrefix/*=true*/)
{
	DllMod::TpFunctObj FunctionFound;
	SGE::CL_TEMPLATE_OBJECT_MANAGER<DllMod>::iterator itDllMod;
	for(itDllMod=this->GetFirstIter(); itDllMod!=this->GetLastIter(); itDllMod++)
	{
		FunctionFound = (*itDllMod).second->GetProcAddress(_functionName, usePrefix);
		if(FunctionFound)
			return FunctionFound;
	}
	GPUCV_WARNING("Given function '" << _functionName << "' not found in any modules loaded");
	return NULL;
}
#endif
//===================================================
CL_FctSw * DllManager::GetFunctionObj(const std::string _functionName, bool usePrefix/*=true*/)
{
	//get or create function sw obj.
	CL_FctSw * tmpFctSw = CL_FctSw_Mngr::GetSingleton()->Get(_functionName);
	if(!tmpFctSw)
	{
		tmpFctSw = CL_FctSw_Mngr::GetSingleton()->Add(_functionName);
	}

	//look for all implementations?
	SGE::CL_TEMPLATE_OBJECT_MANAGER<DllMod, std::string>::iterator itDllMod;
	for(itDllMod=this->GetFirstIter(); itDllMod!=this->GetLastIter(); itDllMod++)
	{
		switchFctStruct * TmpIpl= (*itDllMod).second->GetProcImpl(_functionName, usePrefix);
		if(TmpIpl)
		{
			tmpFctSw->AddImplementation(TmpIpl);
		}
	}
	//========================
	//GPUCV_WARNING("Given function '" << _functionName << "' not found in any modules loaded");
	return tmpFctSw;
}
//===================================================
DllMod * DllManager::AddLib(const std::string _lib, const std::string _prefix)
{
	DllMod *NewLib = Add(_lib);
	NewLib->SetPrefix(_prefix);
	return NewLib;
}
//===================================================
int DllManager::InitAllLibs(bool InitGLContext, bool isMultiThread)
{
	CL_FctSw *InitFct = GetFunctionObj("DLLInit", true);
	if(!InitFct)
	{
		GPUCV_ERROR("No cv*DLLInit found in any of the DLLS");
	}

	//look for all implementations?
	int InitCalled = 0;
	switchFctStruct * switchFct = NULL;
	for(unsigned int i=0; i< InitFct->GetImplSwitchNbr(); i++)
	{
		switchFct = InitFct->GetImplSwitch(i);
		if(switchFct!=NULL)
		{
			int(*InitCall)(bool, bool) = (int(*)(bool, bool))switchFct->m_ImplPtr;
			if(InitCall)
			{
				if(InitCall(InitGLContext, isMultiThread)!=-1)
					InitCalled++;
			}
		}
	}
	return InitCalled;
}
//===================================================
SG_TRC::ColorFilter & DllManager::GenerateLibraryColorFilters(SG_TRC::ColorFilter & rColorFilter, int _ColorNbr)
{
	//look for all implementations?
	SGE::CL_TEMPLATE_OBJECT_MANAGER<DllMod, std::string>::iterator itDllMod;
	for(itDllMod=this->GetFirstIter(); itDllMod!=this->GetLastIter(); itDllMod++)
	{
		
		if((*itDllMod).second)
		{
			(*itDllMod).second->GetModInfos()->GenerateDefaultColorFilter(rColorFilter, _ColorNbr);
		}
	}
	//========================
	return rColorFilter;
}
//===================================================
}//namespace GCV
