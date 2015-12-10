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
#ifndef __GPUCV_HARDWARE_MODULE_INFO_H
#define __GPUCV_HARDWARE_MODULE_INFO_H


#include <GPUCVHardware/config.h>
#include <SugoiTools/tools.h>
#include <SugoiTracer/svgGraph.h>
//#include <GPUCVHardware/GlobalSettings.h>

namespace GCV{

#define DLLINFO_DFT_URL		"https://picoforge.int-evry.fr/cgi-bin/twiki/view/Gpucv/Web/WebHome"
#define DLLINFO_DFT_AUTHOR	"gpucv-developers@picoforge.int-evry.fr"
	
	
/** Enum to select switch mode, we can choose a specific technology
*/
enum BaseImplementation
{
	GPUCV_IMPL_AUTO				= 0x0000, //!< Use OpenCV
	GPUCV_IMPL_OPENCV			= 0x0001, //!< Use OpenCV
	GPUCV_IMPL_GLSL				= 0x0002, //!< Use GpuCV_GLSL
	GPUCV_IMPL_CUDA				= 0x0004, //!< Use GpuCV_Cuda
	GPUCV_IMPL_OPENCL			= 0x0008, //!< Use GpuCV_OPENCL
	GPUCV_IMPL_DIRECTX			= 0x0010, //!< not used...
	GPUCV_IMPL_DIRECTCOMPUTE	= 0x0020, //!< not used...
	GPUCV_IMPL_OTHER			= 0x0040, //!< Unkown
};

#define GPUCV_IMPL_AUTO_STR		"AUTO"
#define GPUCV_IMPL_OPENCV_STR	"OpenCV"
#define GPUCV_IMPL_GLSL_STR		"GLSL"
#define GPUCV_IMPL_CUDA_STR		"CUDA"
#define GPUCV_IMPL_IPP_STR		"IPP"
#define GPUCV_IMPL_OPENCL_STR	"OpenCL"

/** An implementation describe both a techology and a library/algorithm to implement new operators cause some operators can have several implementation using CUDA but using different libraries. Ex:
cvgCudaAdd() and cvgNPPAdd() both implement the Add operator using CUDA, but the second one is using NVIDIA NPP library. They need to be distinguish by the switch so each new plugin will register itself and retrieve the associated implementation descriptor. 
*/
struct ImplementationDescriptor
{
	BaseImplementation		m_baseImplID;			//!< Base id of the implementation, see Implementation.
	unsigned char			m_dynImplID;			//!< Part of the ID that is affected dynamically, ie position in the stack.
	std::string				m_strImplName;			//!< Name of the registered implementation.
};


struct ModColor{
	unsigned int R;
	unsigned int G;
	unsigned int B;
	unsigned int A;
};


_GPUCV_HARDWARE_EXPORT ModColor CreateModColor(unsigned int R, unsigned int G, unsigned int B, unsigned int A);
//todo: _GPUCV_HARDWARE_EXPORT TiXmlElement* XMLRead(TiXmlElement* _XML_Root ,CreateModColor & _rModColor);
//todo: _GPUCV_HARDWARE_EXPORT TiXmlElement* XMLWrite(TiXmlElement* _XML_Root ,CreateModColor & _rModColor);

/** A function called modGetLibraryDescriptor() must be define in all GpuCV plugins.
\code
LibraryDescriptor* modGetLibraryDescriptor(void);
\endcode
*/
class  _GPUCV_HARDWARE_EXPORT LibraryDescriptor
{
protected:
	//! Major version.
	_DECLARE_MEMBER(std::string, VersionMajor);
	//! Minor version.
	_DECLARE_MEMBER(std::string, VersionMinor);
	//! Store the revision SVN number.
	_DECLARE_MEMBER(std::string, SvnRev);
	//! Store the revision SVN date.
	_DECLARE_MEMBER(std::string, SvnDate);	
	//! Store the revision SVN URL.
	_DECLARE_MEMBER(std::string, WebUrl);	
	//! Contact person(s)
	_DECLARE_MEMBER(std::string, Author);	
	//! Library name.
	_DECLARE_MEMBER(std::string, DllName);	
	//! Name of the processing technology or library used, shortest is best, ex: "CUDPP", "NPP", "CULAS", "CUDA", "GLSL", etc... 
	_DECLARE_MEMBER(std::string, ImplementationName);	
	//! ID of the processing technology used, ex: GPUCV_IMPL_GLSL, GPUCV_IMPL_CUDA, SSE, etc
	_DECLARE_MEMBER(BaseImplementation, BaseImplementationID);
	//! Flag to specify if this lib is using Gpu.
	_DECLARE_MEMBER(bool, UseGpu);
	//! Part of the ID that is affected dynamically, ie position in the stack, do not modify it manually.
//	_DECLARE_MEMBER(unsigned char, DynImplID);

	_DECLARE_MEMBER(ModColor, StartColor);
	_DECLARE_MEMBER(ModColor, StopColor);

	//! This descriptor is generated when registering the library using GpuCVSettings::RegisterNewImplementation()
	_DECLARE_MEMBER(const ImplementationDescriptor*, ImplementationDescriptor);
public:
	LibraryDescriptor();

	~LibraryDescriptor();
	/** Generate several color filters used into the benchmarks, and add them to the given filter.
	\param _rColorTable -> Color table to fill
	\return updated color filter.
	*/
	SG_TRC::ColorFilter & GenerateColorFilter(SG_TRC::ColorFilter &_rColorTable, ModColor & _rStartColor, ModColor & _rStopColor, unsigned int _ColorNbr); 
	
	/** Generate default color filters to use into the benchmarks. They can be changed within the plugin using properties m_StartColor/m_EndColor.
	*/
	SG_TRC::ColorFilter & GenerateDefaultColorFilter(SG_TRC::ColorFilter &_rColorTable, unsigned int _ColorNbr); 
	
};




/**
	*\brief takes enum value and returns the corresponding implementation string
	*\param _Impl => enum Implementation .
	*\return : implementation string.
	*\author : Cindula Saipriyadarshan.
	*/
_GPUCV_HARDWARE_EXPORT std::string GetStrImplemtation(const ImplementationDescriptor *_Impl);

_GPUCV_HARDWARE_EXPORT std::string GetStrImplemtationID(BaseImplementation _ID);

}//namespace GCV
#endif
