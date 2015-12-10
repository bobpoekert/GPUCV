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
#ifndef __GPUCV_SWITCH_CL_DLL_H
#define __GPUCV_SWITCH_CL_DLL_H

#include <GPUCVSwitch/config.h>
#include <GPUCVHardware/Singleton.h>
#include <GPUCVSwitch/Cl_FctSw.h>
#include <SugoiTools/dlls.h>

namespace GCV{

/** Describe a DLL file with:
<ul><li>DLL filename</li>
<li>DLL architecture(optionnal)</li>
<li>DLL OS(optionnal)</li>
<li>DLL version(optionnal)</li>
</ul>
 DllDesciptor is a base element used by DllMod.
*/
class DllDescriptor: public SGE::CL_XML_BASE_OBJ<std::string>
{
protected:
	//! Name of the release DLL.
	_DECLARE_MEMBER_BYREF(std::string, Release);
	//! Name of the Debug DLL.
	_DECLARE_MEMBER_BYREF(std::string, Debug);
	//! Architecture of the binary, x32/x64.
	_DECLARE_MEMBER_BYREF(std::string, Arch);
	//! Version of the DLL.
	_DECLARE_MEMBER_BYREF(std::string, Version);
	//! Target OS of the DLL.
	_DECLARE_MEMBER_BYREF(std::string, Os);
	//! Name of a required dependency, if the given library is not found, we will not load this plugin.
	_DECLARE_MEMBER_BYREF(std::string, Dependency);
public:
	DllDescriptor(const std::string _Name);
	DllDescriptor(void);
	~DllDescriptor();

	//! Read member values from XML stream.
	virtual
		TiXmlElement*	
		XMLLoad(TiXmlElement* _XML_Root, const std::string & _subTagName);
	
	//! Write member values to XML stream.
	virtual
		TiXmlElement*	
		XMLSave(TiXmlElement* _XML_Root, const std::string & _subTagName);
};
//===================================================

	
//===================================================
/**	DllMod describe and handle the dll file of one library. 
 Dll can be descibed by several files, depending on the version/OS/ARCH/config using DllDescriptor.
*/
class _GPUCV_SWITCH_EXPORT DllMod
	: public SGE::CL_XML_BASE_OBJ<std::string>
{
protected:
	//! Handle to the given module when loaded.
	_DECLARE_MEMBER(LibHandleType, HandleDLL);
	_DECLARE_MEMBER_BYREF__NOSET(LibraryDescriptor *, ModInfos);
	//! Enable/disable the usage of the lib into the application, can be set in the XML files.
	_DECLARE_MEMBER(bool, Enabled);
	//! Set the loading status of the DLL, EXEC_UNKNOWN means no loading has been trying, EXEC_FAILED_CRITICAL_ERROR means an error occured while loading so we do not enable this plugin and EXEC_SUCCESS means...success.
	_DECLARE_MEMBER(SG_TRC::ExecState, LoadStatus);
	
	//! Vector containing all possible version/name of the given DLL file.
	std::vector<DllDescriptor*> m_vecDLLFiles;
	
	//! Pointer to the DLL descriptor that have been used to successfuly open the DLL
	DllDescriptor * m_pCurrentDllFile;

	//! Prefixe of the function that are stored into the DLL, ex: cv, cvg or cvgsw.
	_DECLARE_MEMBER_BYREF_CST(std::string, Prefix);

public:
#ifdef _WINDOWS
	typedef PROC		TpFunctAdd;
	typedef TpFunctAdd	TpFunctObj;
#else
	typedef void*		TpFunctAdd;
	typedef TpFunctAdd	TpFunctObj;
#endif

	//! Constructor
	DllMod(const std::string _Name, const std::string _modPrefix="");
	
	//! Destructor
	~DllMod();

	//std::string GetDebugName();
	//void SetDebugName(const std::string _debugName);
	//! Load one DLL from the m_vecDLLFiles depending on current host configuration.
	virtual 
		SG_TRC::ExecState Load();

	/** Look into openned DLL for the given function.
	 \param _functionName -> name of the function to look for.
	 \param _usePrefix -> if true => append prefix value to the function name.
	*/
	GCV::switchFctStruct* 
		GetProcImpl(const std::string _functionName, bool usePrefix=true);

	//! Read member values from XML stream.
	virtual
		TiXmlElement*	
		XMLLoad(TiXmlElement* _XML_Root);
	
	//! Write member values to XML stream.
	virtual
		TiXmlElement*	
		XMLSave(TiXmlElement* _XML_Root);

protected:
	/** Read library informations related to GpuCV plug-in description.
	 */
	virtual 
		const LibraryDescriptor * ReadLibInformations();
};
	
//===================================================
//===================================================
//===================================================
/** This manager contains all DllMod object loaded from the XML file or added manually with function cvgswAddLib().
 
*/
class _GPUCV_SWITCH_EXPORT DllManager
	: public SGE::CL_XML_MNGR<DllMod, std::string>
	, public CL_Singleton<DllManager>
{
public:
	DllManager();
	~DllManager();

	/** Add a new library to the manager.
	 * \param _lib -> Library name.
	 * \param _prefix -> Library prefix such as: cv/cvg...
	 *\return Pointer on the loaded library object DllMod.
	*/
	DllMod * AddLib(const std::string _lib, const std::string _prefix);
		
	/** Look into all loading libraries to find all implementations of the given function, using library prefixe or not.
	 The function switch object is linked to the corresponding tracing object using a SG_TRC::CL_CLASS_TRACER<SG_TRC::CL_TimerVal>. 
	 * \param _functionName -> Library name.
	 * \param _usePrefix -> Use library prefix such as: cv/cvg or not.
	 * \return Pointer to Function switch object CL_FctSw.
	 */
	CL_FctSw * GetFunctionObj(const std::string _functionName, bool usePrefix=true);

	/** Search for function cv*DLLInit() in all libs and call it.
	\sa cvgDLLInit(), cvgCudaDllInit().
	 */
	int InitAllLibs(bool InitGLContext, bool isMultiThread);


	/** Add all libraries benchmarking color values to given color filter.
	 */
	SG_TRC::ColorFilter & GenerateLibraryColorFilters(SG_TRC::ColorFilter & rColorFilter, int _ColorNbr);


	
private:
	std::string m_strXMLFilename; //!< XML file name containing the dll list.
};

}//namespace GCV
#endif //__GPUCV_SWITCH_CL_DLL_H
