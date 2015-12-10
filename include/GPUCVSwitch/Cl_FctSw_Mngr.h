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
 * \author Yannick Allusse
*/
#ifndef __GPUCV_SWITCH_CL_FCTSW_MNGR_H
#define __GPUCV_SWITCH_CL_FCTSW_MNGR_H

#include <GPUCVSwitch/config.h>
#include <GPUCVSwitch/Cl_FctSw.h>
#include <SugoiTools/cl_xml_mngr.h>

namespace GCV{

/** \brief Manager that stores, load, save all the CL_FctSw object used into XML files.
  * \author Yannick Allusse
*/
class _GPUCV_SWITCH_EXPORT CL_FctSw_Mngr
	: public SGE::CL_XML_MNGR<GCV::CL_FctSw, std::string>
	, public CL_Singleton<CL_FctSw_Mngr>
{
protected:
	//! \todo change type to a more flexible one later!!!

	//! Used to force(suggest) the selected implementation (GLSL/CUDA/OPENCL...) to all the operators, local forced Implementation ID keep priority.
	_DECLARE_MEMBER(BaseImplementation, GlobalForcedImplemID);
	
	//! Minimum nbr of loop to benchmark all implementations.
	_DECLARE_MEMBER(unsigned int,	MinBenchNbr);
	
	std::string m_strXMLFilename;
public:
	CL_FctSw_Mngr();
	~CL_FctSw_Mngr();

	void PrintAllStatistics();
	virtual
		TiXmlElement*	
		XMLLoad(TiXmlElement* _XML_Root);
	virtual
		TiXmlElement*	
		XMLSave(TiXmlElement* _XML_Root);
	
	int UpdateGpuCVSysFunctions(const std::string * _fctListName, int _nbr, std::string _strImplementationName);	
};

}//namespace GCV
#endif//__GPUCV_SWITCH_CL_FCTSW_MNGR_H
