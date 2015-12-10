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
#ifndef __GPUCV_HARDWARE_GENERICGPU_H
#define __GPUCV_HARDWARE_GENERICGPU_H

#include <GPUCVHardware/GLExtension.h>
#include <GPUCVHardware/GlobalSettings.h>

namespace GCV{
class glGPUExtension;

/**
*	\brief Generic class to describe a GPU and its extensions.
*	This class should be inherited by specific class for each GPU brands using specific hardware tweaks.
*	\todo Most of the call to the GPU should go to threw this class.
*/
class _GPUCV_HARDWARE_EXPORT_CLASS GenericGPU
{
public:
	enum BusMode{
		BUS_UNKNOWN,
		BUS_AGP,
		BUS_PCI,
		BUS_PCI_EXP
	};

	enum MultiGPUMode{
		MGPU_NO_SLI,			//only one GPU, SLI is not activated
		MGPU_Single,	//SLI activated but we use only one GPU
		MGPU_AFR,		//Each GPU works on its own Frame.
		MGPU_SFR,		//Each frame is spitted for each GPU.
		MGPU_AUTO
	};

	/**
	*	\brief Define the main graphic card brands, the sub families must be defined into each of them specific GPU class.
	*/
	enum HardwareFamilyBase{
		HRD_FAM_MOBILE	=0x0001,//to add to other families
		HRD_FAM_UNKNOWN	=0x0000,
		HRD_FAM_NVIDIA	=0x1000,
		HRD_FAM_ATI		=0x2000,
		HRD_FAM_INTEL	=0x3000,
		HRD_FAM_S3		=0x4000,
		HRD_FAM_MATROX	=0x5000
	};

	enum HardwareProfile{
		HRD_PRF_0,// no GLSL support, only base OpengGL supported
		HRD_PRF_1,// eq fp20 -> GeForce 3, only VertexShader available
		HRD_PRF_2,// eq fp30 -> GeForce 5, Vertex & Fragment shaders, no dynamic branching
		HRD_PRF_3,// eq fp40 -> GeForce 6, V & F shaders, dynamic branching, MRT, Vertex shader texture look up
		HRD_PRF_4,// eq  -> GeForce 8, G7 V & F unified shaders, dynamic branching, MRT, Vertex shader texture look up
		HRD_PRF_CUDA
	};

	/**
	*	\brief Constructor
	*/
	__GPUCV_INLINE
		GenericGPU();
	/**
	*	\brief Constructor
	*/
	__GPUCV_INLINE virtual
		~GenericGPU();

	/**
	*	\brief Read informations about the GPU brand.
	*/
	virtual bool ReadBrandSettings();

	/**
	*	\brief Try to find hardware profile depending on available GLEW extensions.
	*	\sa HardwareProfile.
	*/
	HardwareProfile FindHardwareProfile();

	__GPUCV_INLINE
		const std::string &	GetBrand()const;
	__GPUCV_INLINE
		const std::string &	GetModel()const;
	__GPUCV_INLINE
		const std::string &	GetRenderer()const;
	/**
	*	\brief Read informations about the GPU settings and compatibilities.
	*/
	virtual bool ReadGPUSettings();

	/**
	*	\brief Set the GPU to work on a specific multi-GPU mode.
	*/
	virtual bool SetMultiGPUMode(MultiGPUMode _mode);
	MultiGPUMode			GetMULTIGPUMode()const;

	__GPUCV_INLINE
		int			GetGLSLProfile()const;
	__GPUCV_INLINE
		void		SetGLSLProfile(int _new_prof);
	/**
	*	\brief Check if the GPU is part of a hardware family.
	*/
	__GPUCV_INLINE
		bool		IsHardwareFamily(HardwareFamilyBase _fam);

	__GPUCV_INLINE
		GLuint		GetDefaultInternalFormat()const {return m_default_framebuffer_format;}

	__GPUCV_INLINE
		bool		IsDefaultInternalFormatForced()const {return m_force_framebuffer_format;}

	/** \brief Return information about double float precision format support on GPU.
	*	\todo Make answer dynamic.
	*/
	bool			IsDoubleSupported(void)const{return m_supportDouble;}

	/**
	*	\brief Convert openGL texture format and type to best Internal format match.
	*	This function try to match the OpenGL texture format and type to the best internal format
	*	independently from the hardware compatibilities or issues. This function is mainly call when we create textures from buffers or files.
	*	\attention Hardware compatibilities must be control in inherited class.
	*/
	virtual
		GLuint	ConvertGLFormatToInternalFormat(GLuint _format, GLuint _type)const;

	/** Return a string with memory usage (free/available)
	*/
	virtual std::string GetMemUsage()const{return "Unknown";}
	virtual
		std::string ExportToHTMLTable();
protected:
#ifdef _WINDOWS
	HINSTANCE LoadDriverLib(std::string _name);
#endif
	//members
public:
	glGPUExtension m_glExtension;
protected:
#ifdef _WINDOWS
	HINSTANCE m_driverLib;			//!< Handler to the driver DLL.
#endif
	BusMode m_busMode;				//!< Bus type AGP/PCI/PCI-EXPRESS.
	long	m_busTransferRate;		//!< Bus transfer rate.
	long	m_agpMemSize;			//!< AGP memory size
	long	m_videoMemSize;			//!< Video memory size
	MultiGPUMode	m_MultiGPUMode;	//!< GPU mode SLI.
	long	m_gpuNbr;				//!< GPU number on the system.
	int		m_hardwareFamily;
	int     m_glsl_profile;
	GLuint	m_default_framebuffer_format;	//!< This frame buffer format will be used when GPUCV is not sure about internal format compatibilities.
	bool	m_force_framebuffer_format;		//!< Flag to force textures to use the default frame buffer format(m_default_framebuffer_format).
	std::string  m_renderer;			//!< Graphic chipset name.
	std::string  m_openGL_version;		//!< OpenGL version compatible.
	std::string  m_shading_version;		//!< Shading language version compatible.
	std::string  m_brand;
	std::string  m_model;
	bool	m_supportDouble;		//!< GPU support double floating point operations.
	//extensions
	//model
};


const std::string GPUCV_BUS_STR[] = {
	"Unknown",
	"AGP",
	"PCI",
	"PCI-Express"
};

const std::string GPUCV_MULTIGPU_STR[] = {
	"No SLI/CROSSFIRE available",
	"Using single GPU",
	"AFR mode, each GPUs work on its own Frame",
	"SFR mode, each frame is splitted between all GPUs",
	"Auto mode, choose best performances"
};

}//namespace GCV
#endif//_GPUCV_HARDWARE_GENERICGPU_H
