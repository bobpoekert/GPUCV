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
#include <GPUCVHardware/GPU_ATI.h>

namespace GCV{

//=================================================
GPU_ATI::GPU_ATI()
:GenericGPU()
{

}
//=================================================
GPU_ATI::~GPU_ATI()
{

}
//=================================================
/*virtual*/
/** \todo Read GPU settings for ATI cards, see NVIDIA GPU.*/
bool GPU_ATI::ReadGPUSettings()
{
	GenericGPU::ReadGPUSettings();
	ReadBrandSettings();
	//get pointer to main functions from the DLL
	return true;
}
//=================================================
/** \todo Manage Cross fire mode for ATI GPUs*/
bool GPU_ATI::SetMultiGPUMode(MultiGPUMode _mode)
{
	GPUCV_WARNING("Multi GPU mode not yet supported for ATI hardware");
	return false;
}
//=================================================
bool GPU_ATI::ReadBrandSettings()
{
	if (m_renderer == "")
	{
		GPUCV_WARNING("Warning : can't parse empty renderer name.\n");
		return false;
	}
	std::string model;

	size_t Radeon= m_renderer.find("radeon");
	size_t FireGL= m_renderer.find("firegl");
	size_t FirePro= m_renderer.find("firepro");
	size_t Mobility = m_renderer.find("mobility");
	//mobility flag:
	if(Mobility!= std::string::npos)
		m_hardwareFamily = HRD_FAM_MOBILE;


	if (Radeon!= std::string::npos)
	{//radeon card, wich one.?
		size_t RadeonX = m_renderer.find("radeon x");
		size_t RadeonHD = m_renderer.find("radeon hd");
		
		if (RadeonX != std::string::npos)
		{
			model = m_renderer.substr(RadeonX+ strlen("radeon x"), 4);
			m_hardwareFamily += ATI_RADEON_X_FAM;
			m_model = "Radeon X";
		}
		else if (RadeonHD != std::string::npos)
		{
			model = m_renderer.substr(RadeonHD+ strlen("radeon hd") +1, 4);
			m_hardwareFamily += ATI_RADEON_HD_FAM;
			m_model = "Radeon HD";
		}
		else
		{
			model = m_renderer.substr(Radeon+7, 4);
			m_hardwareFamily += ATI_RADEON_FAM;
			m_model = "Radeon";
		}

		//model = m_renderer.substr(((RadeonX == std::string::npos)?Radeon:RadeonX)+8, 1);
		int m = atoi(model.data());

		//m_hardwareFamily += m;
		m_brand = "ATI";
		
		if(m_hardwareFamily >= ATI_RADEON_X_FAM)
		{
			GPUCV_DEBUG("Radeon card detected : all shaders should be compatible.");
			SetGLSLProfile(GenericGPU::HRD_PRF_3);
			m_glExtension.SetFloat32Compatible(true);//.??
		}
		else
		{
			GPUCV_WARNING("Radeon "<< model <<" detected.");
			GPUCV_WARNING("Warning : Card not recognized. Using GLEW extensions to choose profile.");
			SetGLSLProfile(FindHardwareProfile());
			m_glExtension.SetFloat32Compatible(false);//.??
		}
		return true;
	}
	else if (FireGL!= std::string::npos)
	{//FireGL card, wich one.?
		model = m_renderer.substr(FireGL+ strlen("firegl"), 4);
		m_hardwareFamily += ATI_FIREGL_FAM;
		m_model = "Firepro";
	
		m_brand = "ATI";
		GPUCV_DEBUG("FireGL card detected : all shaders should be compatible.");
		SetGLSLProfile(GenericGPU::HRD_PRF_3);
		m_glExtension.SetFloat32Compatible(true);//.??
		//todo: improve card detection
		return true;
	}
	else if (FirePro!= std::string::npos)
	{//FireGL card, wich one.?
		model = m_renderer.substr(FirePro+ strlen("firepro"), 4);
		m_hardwareFamily += ATI_FIREGL_FAM;
		m_model = "FirePro";
	
		m_brand = "ATI";
		GPUCV_DEBUG("FirePro card detected : all shaders should be compatible.");
		SetGLSLProfile(GenericGPU::HRD_PRF_3);
		m_glExtension.SetFloat32Compatible(true);//.??
		//todo: improve card detection
		return true;
	}
	return false;
}
//=================================================
#if _GPUCV_SUPPORT_GPU_PLUGIN
bool createGPU(GenericGPU * _GpuTable, int * _gpuNbr)
{
	return createGPU<GPU_ATI>(_GpuTable, _gpuNbr);
}
#endif
//=================================================
}//namespace GCV
