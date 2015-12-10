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
/**
*/
#include "StdAfx.h"
#include <GPUCVSwitch/Cl_FctSw_Mngr.h>
using namespace std;

namespace GCV{

std::string GpuCVInternalFunctionToForce []=
{
	//From cxCore
		//IplImages
		"cvCreateImage"
		,"cvReleaseImage"
		,"cvShowImage"
		,"cvCloneImage"
		//CvMat
		,"cvCreateMat"
		,"cvReleaseMat"
		,"cvCloneMat"
		//other
		,"cvCreateData"
		,"cvReleaseData"
		//,"cvSetData"
		,"cvGetRawData"
		//,"cvCopy"
		//,"cvSet"
		//,"cvSetZero"
		,"cvSetIdentity"
		,"cvQueryFrame"
		,"cvRetrieveFrame"

	//From CV

	//From Highgui
};

/*static*/
//template <> CL_FctSw_Mngr * GCV::CL_Singleton<CL_FctSw_Mngr>::m_registeredSingleton = NULL;
//=======================
CL_FctSw_Mngr::CL_FctSw_Mngr()
:SGE::CL_XML_MNGR<CL_FctSw, std::string>(NULL, "fctsw_mngr", "fctsw", "id")
,CL_Singleton<CL_FctSw_Mngr>()
,m_GlobalForcedImplemID(GPUCV_IMPL_AUTO)
,m_MinBenchNbr(5)
,m_strXMLFilename("gcv_FctSwManager.xml")//default file
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
			m_strXMLFilename = NewPath;
	}
}
//=======================
CL_FctSw_Mngr::~CL_FctSw_Mngr()
{
	XMLSaveToFile(m_strXMLFilename);
}
//=======================
void CL_FctSw_Mngr::PrintAllStatistics()
{
	GPUCV_NOTICE("==================================================");\
	GPUCV_NOTICE("			 (Processing)Number_____|_____AmntOfTimeSaved \n\n");
	CL_FctSw_Mngr::GetSingleton()->PrintAllObjects();
	GPUCV_NOTICE("==================================================");\
}
//=======================
/*virtual*/
TiXmlElement*	
CL_FctSw_Mngr::XMLLoad(TiXmlElement* _XML_Root)
{
	TiXmlElement*	CurElem = SGE::CL_XML_MNGR<CL_FctSw, std::string>::XMLLoad(_XML_Root);
	
	//force some function to use GpuCV instead of OpenCV (management functions)
	UpdateGpuCVSysFunctions(GpuCVInternalFunctionToForce, sizeof(GpuCVInternalFunctionToForce)/sizeof(std::string), "GLSL");
	int val = GPUCV_IMPL_AUTO;
	SGE::XMLReadVal(_XML_Root, "GlobalImpl", val);
	m_GlobalForcedImplemID = (BaseImplementation)val;

	SGE::XMLReadVal(_XML_Root, "minBenchNbr", m_MinBenchNbr);

	return CurElem;
}
//=======================
/*virtual*/
TiXmlElement*	
CL_FctSw_Mngr::XMLSave(TiXmlElement* _XML_Root)
{
	TiXmlElement*	CurElem = SGE::CL_XML_MNGR<CL_FctSw, std::string>::XMLSave(_XML_Root);

	//force some function to use GpuCV instead of OpenCV (management functions)
	UpdateGpuCVSysFunctions(GpuCVInternalFunctionToForce, sizeof(GpuCVInternalFunctionToForce)/sizeof(std::string), "GLSL");
	SGE::XMLWriteVal(_XML_Root, "minBenchNbr", m_MinBenchNbr);

	return CurElem;
}
//=======================
int CL_FctSw_Mngr::UpdateGpuCVSysFunctions(const std::string * _fctListName, int _nbr, std::string _strImplementationName)
{
	SG_Assert(_fctListName!=NULL && _nbr!=0, "Empty function list");

	GPUCV_WARNING("[DBG]CL_FctSw_Mngr::UpdateGpuCVSysFunctions()=> start ")
	int int_FctUpdated=0;
	CL_FctSw * CurFct=NULL;
	const ImplementationDescriptor * pCurImpl = GetGpuCVSettings()->GetImplementation(_strImplementationName);
	for(int i =0; i < _nbr; i++)
	{
		CurFct = Get(_fctListName[i]);
		if(CurFct)
		{
			GPUCV_WARNING("[DBG]Function found:" << _fctListName[i]);
			CurFct->SetForcedImpl(pCurImpl);
			int_FctUpdated++;
		}
		else
		{
			GPUCV_WARNING("[DBG]Function not found in XML File:" << _fctListName[i] << " we add it manually");
			Add(_fctListName[i])->SetForcedImpl(pCurImpl);
		}	
	}
	GPUCV_WARNING("[DBG]CL_FctSw_Mngr::UpdateGpuCVSysFunctions()=> stop")
	return int_FctUpdated;
}	
//=======================
}//namespace GCV
