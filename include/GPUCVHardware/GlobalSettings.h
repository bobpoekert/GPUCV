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
#ifndef __GPUCV_HARDWARE_GLOBAL_SETTINGS_H
#define __GPUCV_HARDWARE_GLOBAL_SETTINGS_H

#include <GPUCVHardware/Profiling.h>
#include <GPUCVHardware/CL_Options.h>
#include <GPUCVHardware/moduleInfo.h>


namespace GCV{



/**
\brief CL_Singleton class used to store global settings about GpuCV library.
\author Yannick Allusse
*/
class _GPUCV_HARDWARE_EXPORT_CLASS GpuCVSettings
	:public CL_Options
{
public:
	enum Options
	{
		GPUCV_SETTINGS_GL_ERROR_CHECK			= 0x0001, //!< OpenGL error test are performed really often, which can slow down application.
		GPUCV_SETTINGS_GL_ERROR_RISE_EXCEPTION	= 0x0002, //!< OpenGL error will rise exceptions.
		GPUCV_SETTINGS_LOAD_ALL_SHADERS			= 0x0008, //!< Automatically load all shaders program from shader directories on the first call of cvgInit().
		GPUCV_SETTINGS_USE_OPENCV				= 0x0010, //!< Switch all the operators to use OpenCV if 1, GpuCV if 0.
		GPUCV_SETTINGS_GLOBAL_DEBUG				= 0x0020, //!< Enable the output of debugging informations to the console/log file.
		GPUCV_SETTINGS_GLOBAL_ERROR				= 0x0040, //!< Enable the output of error informations to the console/log file.
		GPUCV_SETTINGS_GLOBAL_WARNING			= 0x0080, //!< Enable the output of warning informations to the console/log file.
		GPUCV_SETTINGS_GLOBAL_NOTICE			= 0x0100, //!< Enable the output of standard informations to the console/log file, this flag should always be one unless you know what your are doing.
		//reserved from 0x100 to 0x800 by LCL_OPT_DEBUG????
		GPUCV_SETTINGS_CHECK_SHADER_UPDATE		= 0x0200, //!< Used to check that the shader file has not been changed before running it.
		GPUCV_SETTINGS_CHECK_IMAGE_ATTRIBS		= 0x0400, //!< Default is enabled to check the CvgArr attributes(size, pixel format...) before applying filters.
		GPUCV_SETTINGS_FILTER_DEBUG				= 0x0800, //!< Default is enabled to check the CvgArr attributes(size, pixel format...) before applying filters.
		GPUCV_SETTINGS_FILTER_SIMULATE			= 0x1000,
		GPUCV_SETTINGS_DEBUG_DATA_TRANSFER		= 0x2000,
		GPUCV_SETTINGS_SYNCHRONIZE_ON_ERROR		= 0x4000, //!< Use to synchronize data with CPU and call OpenCV operator when blocking error occurs in GpuCV operators.
		GPUCV_SETTINGS_PROFILING				= 0x10000, //!< Enable runtime profiling, _GPUCV_PROFILE must be set to 1.
		GPUCV_SETTINGS_PROFILING_TRANSFER		= 0x20000+GPUCV_SETTINGS_PROFILING, //!< Enable runtime profiling of data transfer, _GPUCV_PROFILE must be set to 1.
		GPUCV_SETTINGS_PROFILING_OPER			= 0x40000+GPUCV_SETTINGS_PROFILING, //!< Enable runtime profiling of GPUCV operators, _GPUCV_PROFILE must be set to 1.
		GPUCV_SETTINGS_PROFILING_CLASS			= 0x80000+GPUCV_SETTINGS_PROFILING, //!< Enable runtime profiling of GPUCV internal class, _GPUCV_PROFILE must be set to 1.
		GPUCV_SETTINGS_DEBUG_MEMORY				= 0x100000, //!< Enable runtime profiling of GPUCV internal class, _GPUCV_PROFILE must be set to 1.
		GPUCV_SETTINGS_SWITCH_LOG				= 0x200000,//!< Enable switch mechanism to be logged into console, see GPUCV_SWITCH_LOG.
		GPUCV_SETTINGS_PLUG_THROW_EXCEPTIONS	= 0x400000, //!< Enable plugin to throw exceptions to be catched by the main application
		GPUCV_SETTINGS_PLUG_AUTO_RESOLVE_ERROR	= 0x800000, //!< Enable plugin to try to auto resolve error in an operator by selecting another implementation.
		GPUCV_SETTINGS_LAST_ENUM				= 0x800000 //MUST ALWAYS BE THE LAST ONE
	};

	
	/** \brief Constructor.
	*/
	__GPUCV_INLINE
		GpuCVSettings();
	/** \brief Destructor.
	*/
	__GPUCV_INLINE
		~GpuCVSettings();

	/** \brief Set option _opt to val.
	*	\param _opt => option to set.
	*	\param _val	=> new value.
	*	\sa CL_Option.
	*/
	__GPUCV_INLINE
		virtual void SetOption(OPTION_TYPE _opt, bool _val);

	

	/** \brief Set the init flag value to 1.
	*/
	__GPUCV_INLINE
		void SetInitDone(){m_initDone=true;};

	/** \brief Get the init flag value.
	*/
	__GPUCV_INLINE
		bool IsInitDone(){return m_initDone;}

	/** \brief Return a string containing the current options of GpuCVSettings.
	*/
	std::string GetOptionsDescription(std::string lineStart="", std::string lineMiddle="\t - ",std::string lineEnd="\n");
	/** \brief Return a string containing the version full description of GpuCVSettings.
	*/
	std::string GetVersionDescription();

	std::string ExportToHTMLTable();

#if _GPUCV_GL_USE_GLUT
	void AddGlutWindowsID(int _ID);
#endif
	

	//! \name EXCEPTION MANAGEMENT
	//@{
	//-------------------------------------
	/** \brief Return the last object that might have raised an exception.
	*	\sa PopExceptionObj(), PushExceptionObj(), GetExceptionObjectTree().
	*/
	__GPUCV_INLINE
		const CL_Profiler	* GetLastExceptionObj()const;

	/** \brief Add a new object to the exception object stack.
	*	\param _obj => Pointer to current object that is being processed and that might rise some exceptions.
	*	\return _obj pointer.
	*	\sa PopExceptionObj(), GetLastExceptionObj(), GetExceptionObjectTree().
	*/
	__GPUCV_INLINE
		const CL_Profiler	* PushExceptionObj(const CL_Profiler	* _obj);

	/** \brief Remove and return the last object of the exception stack.
	*	\return Pointer to the last object of the exception stack.
	*	\sa PushExceptionObj(), GetLastExceptionObj(), GetExceptionObjectTree().
	*/
	__GPUCV_INLINE
		const CL_Profiler	* PopExceptionObj();

	/**	\brief Generate a string containing debugging informations about all objects present into the exception stack.
	*	\sa PopExceptionObj(), PushExceptionObj(), GetLastExceptionObj().
	*/
#ifdef _WINDOWS
	std::ostringstream::_Mystr GetExceptionObjectTree()const;
#else
	std::string 		GetExceptionObjectTree()const;
#endif
#if _GPUCV_GL_USE_GLUT

	//-------------------------------------
	//@}
	//! \name DEBUG MODE
	//@{
	//-------------------------------------
	/** \brief Set GLUT debug mode.
	*	GLUT debug mode is used to debug GLSL shaders. Result of the shader is displayed into a GLUT windows.
	*	\warning Must be called before cvgInit().
	*	\warning Windows size must be defined by SetWindowSize() before calling the first operator.
	*	\sa GetGlutDebug().
	*/
	__GPUCV_INLINE
		void EnableGlutDebug();
	/** \brief Get GLUT debug mode.
	*	\sa EnableGlutDebug().
	*/
	__GPUCV_INLINE
		bool GetGlutDebug()const;
#endif//GLUT
	/** \brief Get GLUT windows size, used with GLUT debug mode.
	*	\sa SetWindowSize(), EnableGlutDebug().
	*/
	__GPUCV_INLINE
		const unsigned int * GetWindowSize()const;
	/** \brief Set GLUT windows size, used with GLUT debug mode.
	*	\sa GetWindowSize(), EnableGlutDebug().
	*/
	__GPUCV_INLINE
		void SetWindowSize(unsigned int _width, unsigned int _height);

	//-------------------------------------
	//@}
	//! \name Customize object defaults settings
	//@{
	//-------------------------------------
	bool GetDefaultOption(std::string _objName, CL_Options::OPTION_TYPE & _opt);
	void SetDefaultOption(std::string _objName, CL_Options::OPTION_TYPE & _opt,  bool _val);
	void ForceAllOptions(std::string _objName, CL_Options::OPTION_TYPE &_opt);
	//-------------------------------------
	//@}
	//-------------------------------------

	//-------------------------------------
	//@}
	//! \name Registering implementation type
	//@{
	//-------------------------------------
	const ImplementationDescriptor* RegisterNewImplementation(LibraryDescriptor* _pDllInfos);
	const ImplementationDescriptor* GetImplementation(const std::string _name);
	//-------------------------------------
	//@}
	//-------------------------------------

//members, auto declaration
	//! Store the shader and data path.
	_DECLARE_MEMBER_BYREF__NOSET(std::string, ShaderPath);
	void SetShaderPath(const std::string _path);//defined manually

	//!< Major version number.
	_DECLARE_MEMBER_BYREF(std::string, Version);
	//!< Store the revision SVN number.
	_DECLARE_MEMBER_BYREF(std::string, Revision);
	//!< Store the revision SVN date.
	_DECLARE_MEMBER_BYREF(std::string, RevisionDate);
	//!< Store the SVN URL.
	_DECLARE_MEMBER_BYREF(std::string, URLSvn);
	//!< Store the Home page URL.
	_DECLARE_MEMBER_BYREF(std::string, URLHome);
protected:	
	bool			m_initDone;			//!< Store GpuCV init flag.
	unsigned int	m_windowSize[2];	//!< Size of the debug window.
#if _GPUCV_GL_USE_GLUT
	bool			m_glutDebug;			//!< Create a GLUT windows to debug GLSL shader execution.
	std::vector<int> 	m_GLutWindowsList;//!< List of OpenGL window ID
#endif
	std::vector<const CL_Profiler*> m_vExceptionParentObjects;//!< Stack of pointer to the object that raised the last exceptions

	SGE::CL_TEMPLATE_OBJECT_MANAGER<CL_OptionStorage, std::string> m_defaultOptionManager;
	std::vector<ImplementationDescriptor*>	m_vImplementations;//!< Stack to store all registered implementations references (OpenCV -> 1, CUDA -> 2, etc..)

};


/** \brief CL_Singleton function to access single GpuCVSettings object from anywhere in the application
*/
_GPUCV_HARDWARE_EXPORT
__GPUCV_INLINE GpuCVSettings * GetGpuCVSettings();

#define SET_GPUCV_OPTION(OPTION, VALUE) 	GetGpuCVSettings()->SetOption(OPTION, VALUE)
#define GET_GPUCV_OPTION(OPTION)		GetGpuCVSettings()->GetOption(OPTION)
#define SWITCH_GPUCV_OPTION(OPTION)		SET_GPUCV_OPTION(OPTION, !GET_GPUCV_OPTION(OPTION))
#define PUSH_GPUCV_OPTION()		GetGpuCVSettings()->PushOptions()
#define POP_GPUCV_OPTION()		GetGpuCVSettings()->PopOptions()

#define GPUCV_DEBUG_IMG_TRANSFER(MSG) if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_DEBUG_DATA_TRANSFER)){GPUCV_DEBUG(MSG);}


#if _GPUCV_DEPRECATED

//Push/PopExceptionObj() generate some errors in multi-threaded applications...do not use
#define CLASS_ASSERT(COND,MSG)\
{\
	GetGpuCVSettings()->PushExceptionObj(this);\
	SG_Assert(COND, MSG);\
	GetGpuCVSettings()->PopExceptionObj();\
}
#else
#define CLASS_ASSERT(COND,MSG){SG_Assert(COND, MSG);}
#define CLASS_ASSERT_FILE(COND,FILENAME, MSG){SG_AssertFile(COND, FILENAME, MSG);}
#endif

}//namespace GCV
#endif
