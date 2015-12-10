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
#ifndef __GPUCV_HARDWARE_OPTIONS_H
#define __GPUCV_HARDWARE_OPTIONS_H
#include <GPUCVHardware/config.h>
#include <GPUCVHardware/Tools.h>

namespace GCV{
/**
\brief Supply a Push/Pop mechanism for options parameters.
\author Yannick Allusse
*/
class _GPUCV_HARDWARE_EXPORT_CLASS  CL_Options
	//: public SGE::CL_BASE_OBJ<std::string>
{
public:
	typedef unsigned long int	OPTION_TYPE;

	//use to filter specific message from a Cl_Options inherited class.
	enum GLOBAL_OPTIONS{
		LCL_OPT_DEBUG		= 0x100
		,LCL_OPT_WARNING	= 0x200
		,LCL_OPT_ERROR		= 0x400
		,LCL_OPT_NOTICE		= 0x800
		,OPT_MAX_VAL		= 0x00FFFFFF //!< Maximum value of a unsigned long int.(4BYTES)
	};

	/** \brief Default constructor
	*/
	CL_Options(void);


	CL_Options(OPTION_TYPE _baseOPT);

	/** \brief Destructor
	*/
	virtual
		~CL_Options();

	/** \brief Push the options values into the stack.
	*/
	virtual
		void PushOptions();
	/** \brief Pop the options values from the stack.
	*/
	virtual
		void PopOptions();

	/** \brief Set the given option to the given value.
	*	\param _opt => option ID to set.
	*	\param _val => new option value.
	*/
	virtual
		void SetOption(OPTION_TYPE _opt, bool _val);

	/** \brief Get the given option's value.
	*	\param _opt => option ID to get.
	*/
	virtual
		OPTION_TYPE GetOption(OPTION_TYPE _opt)const;

	/** \brief Get all given option's values.
	*/
	virtual
		OPTION_TYPE GetAllOptions()const;

	/** \brief Push all options and set the given one to value.
	*	\param _opt => option ID to set.
	*	\param val => new option value.
	*	\sa SetOption(), PushOptions().
	*/
	virtual void PushSetOptions(OPTION_TYPE _opt, bool val);

	/** \brief Set all options to the given values _val.
	*	\param _val => new option value.
	*	\sa SetOption(), PushOptions().
	*/
	virtual void ForceAllOptions(OPTION_TYPE _val);

protected:
	std::vector<OPTION_TYPE *>  m_optionsVect;	//!< Vectors of options that can be pop/push in the OpenGl way.
	OPTION_TYPE *				m_currentValue; //!< Pointer to current options into the stack
};

//===========================================

//===========================================
class _GPUCV_HARDWARE_EXPORT_CLASS CL_OptionStorage
	:public CL_Options
	,public SGE::CL_BASE_OBJ<std::string>
{
public:
	CL_OptionStorage(const std::string & _ID)
		:SGE::CL_BASE_OBJ<std::string>(_ID)
	{

	}
	~CL_OptionStorage()
	{
	}
};

/** @ingroup GPUCV_MACRO_LOGGING_GRP.
@{
\name Local debugging
*/
/** \brief Output debugging informations to the main target(file/console) when option GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG and specify flag FLAG are true.
*	Output format is:"[DBG] %INDENT_STRING %MSG\n"
*	\sa GPUCV_DEBUG
*/
#define GPUCV_LOCAL_DEBUG_FLAG(FLAG, msg)	{if(CL_Options::GetOption(FLAG))			{GPUCV_NOTICE(msg);}}
#define GPUCV_LOCAL_WARNING(msg)			{if(CL_Options::GetOption(LCL_OPT_WARNING))	{GPUCV_WARNING(msg);}}
#define GPUCV_LOCAL_ERROR(msg)				{if(CL_Options::GetOption(LCL_OPT_ERROR))	{GPUCV_ERROR(msg);}}
#define GPUCV_LOCAL_DEBUG(msg)				{if(CL_Options::GetOption(LCL_OPT_DEBUG))	{GPUCV_DEBUG(msg);}}
#define GPUCV_LOCAL_NOTICE(msg)				{if(CL_Options::GetOption(LCL_OPT_NOTICE))	{GPUCV_NOTICE(msg);}}
/** @} */

}//namespace GCV
#endif
