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
#include "GPUCVHardware/CL_Options.h"
#include "GPUCVHardware/GlobalSettings.h"

namespace GCV{

//=================================================
CL_Options::CL_Options()
{
	m_currentValue = new OPTION_TYPE();
}
CL_Options::CL_Options(OPTION_TYPE _baseOPT)
{
	m_currentValue = new OPTION_TYPE();
	SetOption(_baseOPT, true);
}
//=================================================
CL_Options::~CL_Options()
{
	delete m_currentValue;
	//delete all others
}
//============================================================
void CL_Options::PushOptions()
{
	m_optionsVect.push_back(m_currentValue);
	OPTION_TYPE * newOption = new OPTION_TYPE();
	*newOption = *m_currentValue;//copy previous values
	m_currentValue = newOption;

}
//============================================================
void CL_Options::PopOptions()
{
	if(m_optionsVect.size() <1)
	{
		GPUCV_WARNING("m_optionsVect is empty, can not pop options");
		return;
	}
	delete m_currentValue;
	m_currentValue = m_optionsVect[m_optionsVect.size()-1];
	m_optionsVect.pop_back();
}
//============================================================
void CL_Options::SetOption(CL_Options::OPTION_TYPE _opt, bool val)
{
	*m_currentValue= (val)?
		(*m_currentValue) | _opt:
		(*m_currentValue) &(OPT_MAX_VAL ^ _opt);
}
/*virtual*/
void CL_Options::ForceAllOptions(OPTION_TYPE _val)
{
	*m_currentValue = _val;
}
//============================================================
CL_Options::OPTION_TYPE CL_Options::GetOption(CL_Options::OPTION_TYPE _opt)const
{
	OPTION_TYPE val = ((*m_currentValue) & _opt);
	if(val==_opt)
		return val;
	return 0;
}
//============================================================
CL_Options::OPTION_TYPE CL_Options::GetAllOptions()const
{
	return *m_currentValue;
}
//============================================================
void CL_Options::PushSetOptions(OPTION_TYPE _opt, bool val)
{
	PushOptions();
	SetOption(_opt, val);
}
//============================================================
}//namespace GCV
