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
#include "GPUCVHardware/hardware.h"

#if !_GPUCV_SUPPORT_GPU_PLUGIN
#include <GPUCVHardware/GPU_ATI.h>
#include <GPUCVHardware/GPU_NVIDIA.h>
#endif

namespace GCV{

#define __HARDWARE_VERBOSE

//===================================================
HardwareProfile :: HardwareProfile()
:m_ProcessingGPU(NULL),
m_RenderingGPU(NULL),
m_MainGPU(NULL)
{
	//look for all the GPUs on the system
	CreateGPU();
	//===================================

	//affect GPUs
	if (m_GPUVector.size()==1)
	{//rendering and processing are on same GPU
		m_ProcessingGPU = m_RenderingGPU = m_MainGPU = *m_GPUVector.begin();
	}
	else
	{
		GPUCV_WARNING("NO GPU FOUND!!!");
		return;
	}
	//===================================

#if _DEBUG
	if(m_MainGPU)
		m_MainGPU->m_glExtension.PrintExtension();
#endif


	m_ProcessingGPU->SetMultiGPUMode(GenericGPU::MGPU_NO_SLI);
	GPUCV_NOTICE("Reading OpenGL Extensions");

	if(!m_ProcessingGPU->m_glExtension.IsFBOCompatible() && !m_ProcessingGPU->m_glExtension.IsFBOCompatible())
	{
		GPUCV_ERROR("\n\nERROR : Your graphic card is not compatible with 'Frame Buffer object' or 'Pixel Buffer'. GPUCV library can not run properly. Using OPENCV library instead!!!\n\n");
	}

	if(!m_ProcessingGPU->m_glExtension.IsTextRectCompatible() && !m_ProcessingGPU->m_glExtension.IsTextNOPCompatible())
	{
		GPUCV_ERROR("\n\nERROR : Your graphic card is not compatible GLEW_ARB_texture_rectangle and GLEW_ARB_texture_non_power_of_two extensions. To run properly GPUCV will used bigger texture size for each image, using more VRAM.\n\n");
	}

	SetTextType(0);//automatique.
	GPUCV_NOTICE("Reading OpenGL Extensions done");

}
//=================================================
HardwareProfile :: ~HardwareProfile()
{
	//delete TextureRenderBuffer::GetMainRenderer();
	for(unsigned int i=0; i< m_GPUVector.size(); i++)
	{
		delete m_GPUVector[i];
	}
	m_GPUVector.clear();
	/*

	if (m_ProcessingGPU != m_MainGPU)
		delete m_ProcessingGPU;
	if (m_RenderingGPU != m_MainGPU)
		delete m_RenderingGPU;
	delete m_MainGPU;

	m_RenderingGPU = m_MainGPU = m_ProcessingGPU = NULL;
*/
}
//===================================================
std::vector<GenericGPU*>::iterator HardwareProfile ::GetFirstGPUIter()
{
	return m_GPUVector.begin();
}
//===================================================
std::vector<GenericGPU*>::iterator HardwareProfile ::GetLastGPUIter()
{
	return m_GPUVector.end();
}
//===================================================
size_t HardwareProfile ::GetGPUNbr()const
{
	return m_GPUVector.size();
}
//===================================================
/**
*	\todo Create a manager that will create GPU(s) automatically.
*	\note Does not detect multiGPU yet.
*/
//=================================================
int	HardwareProfile :: CreateGPU()
{
	int GPUNbr = 0;
#if !_GPUCV_SUPPORT_GPU_PLUGIN

	//get GPU NAME
	std::string TmpStr = (char*)glGetString(GL_RENDERER);
	GPUCV_DEBUG("GPU found: " <<  TmpStr);
	std::string GpuName = SGE::StringToLower(TmpStr);


	GenericGPU * CurrentGPU = NULL;

	if(GpuName.find("ati") !=std::string::npos
		||	GpuName.find("radeon") !=std::string::npos
		||	GpuName.find("firegl")!=std::string::npos
		||	GpuName.find("firemv")!=std::string::npos)
	{//GPU is from ATI
		CurrentGPU = new GPU_ATI();
		CurrentGPU->ReadGPUSettings();

		if(CurrentGPU->IsHardwareFamily(GenericGPU::HRD_FAM_ATI))
		{//confirm ATI GPU
			AddGPU(CurrentGPU);
			return GPUNbr = 1;
		}
		else
			delete CurrentGPU;
	}
	else if(		GpuName.find("nvidia")!=std::string::npos
		||	GpuName.find("geforce")!=std::string::npos
		||	GpuName.find("quadro")!=std::string::npos)
	{//GPU is from NVIDIA
		CurrentGPU = new GPU_NVIDIA();
		CurrentGPU->ReadGPUSettings();

		if(CurrentGPU->IsHardwareFamily(GenericGPU::HRD_FAM_NVIDIA))
		{//confirm NVIDIA GPU
			AddGPU(CurrentGPU);
			return GPUNbr = 1;
		}
		else
			delete CurrentGPU;
	}
	else
	{//GPU is from another company
		CurrentGPU = new GenericGPU();
		CurrentGPU->ReadGPUSettings();
		AddGPU(CurrentGPU);
		return GPUNbr = 1;
	}
	//==========================
#endif
	GPUNbr = 0;
	return GPUNbr;
}
//===================================================
bool HardwareProfile::IsFBOCompatible()
{
	return m_ProcessingGPU->m_glExtension.IsFBOCompatible();
}
//===================================================
int HardwareProfile::GetTextType()
{
	return m_dfltTextType;
}
//===================================================
void HardwareProfile :: SetTextType(GLuint _type/*=0*/)
{

	//set default texture type
#if _GPUCV_TEXTURE_FORCE_TEX_RECT
	GPUCV_DEBUG("Forcing GL_TEXTURE_RECTANGLE_ARB\n");
	m_dfltTextType= GL_TEXTURE_RECTANGLE_ARB;
#endif
#if _GPUCV_TEXTURE_FORCE_TEX_NPO2
	GPUCV_DEBUG("Forcing GL_TEXTURE_2D\n");
	m_dfltTextType= GL_TEXTURE_2D;
#endif

	if(_type!= 0)
	{
		m_dfltTextType = _type;
		return;
	}

	if (m_ProcessingGPU->m_glExtension.IsTextNOPCompatible()==1)
	{
		m_dfltTextType= GL_TEXTURE_2D;
	}
	else if (m_ProcessingGPU->m_glExtension.IsTextRectCompatible()==1)
	{
		GPUCV_DEBUG("using GL_TEXTURE_RECTANGLE_ARB\n");
		m_dfltTextType=GL_TEXTURE_RECTANGLE_ARB;
	}
	else
	{
		m_dfltTextType= GL_TEXTURE_2D;
		GPUCV_DEBUG("using GL_TEXTURE_2D(default)\n");
	}
}
//===================================================
int  HardwareProfile :: GetGLSLProfile()
{
	return  m_ProcessingGPU->GetGLSLProfile();
}
//===================================================
bool HardwareProfile :: IsCompatible(int profile_needed)
{
	if (profile_needed <= ProcessingGPU()->GetGLSLProfile())
		return true;
	else
		return false;

	//else describe error
	/*
	int CurrentProfile = ProcessingGPU()->GetGLSLProfile();


	if (profile_needed == GenericGPU::HRD_PRF_0)
	{
	GPUCV_WARNING("Warning : your card doesn't support shaders.\n");
	return false;
	}
	else if (profile_needed <= CurrentProfile)
	{
	return true;
	}
	else
	{
	if (profile_needed == GenericGPU::HRD_PRF_2)
	{	GPUCV_WARNING("Warning : your card doesn't support shaders. Vertex and Fragment shaders requiered.\n");	}
	else if (profile_needed == GenericGPU::HRD_PRF_3)
	{	GPUCV_WARNING("Warning : your card doesn't support shaders. Vertex and Fragment shaders, dynamic branching or MTR requiered.\n");}

	return false;
	}
	*/
}
//===================================================
GenericGPU	*	HardwareProfile :: GetProcessingGPU()
{	return m_ProcessingGPU; }
//===================================================
GenericGPU	*	HardwareProfile :: GetRenderingGPU()
{	return m_RenderingGPU;		}
//===================================================
GenericGPU	*	HardwareProfile :: GetMainGPU()
{	return m_MainGPU;		}
//===================================================
void	HardwareProfile::SetProcessingGPU(GenericGPU	*_gpu)
{
	m_ProcessingGPU = _gpu;
}
//===================================================
void	HardwareProfile::SetRenderingGPU(GenericGPU	*_gpu)
{
	m_RenderingGPU = _gpu;
}
//===================================================
void	HardwareProfile::SetMainGPU(GenericGPU	*_gpu)
{
	m_MainGPU = _gpu;
}
//===================================================
void	HardwareProfile::AddGPU(GenericGPU	*_gpu)
{
	SG_Assert(_gpu, "Can not add empty GPU");
	m_GPUVector.push_back(_gpu);
}
//===================================================
HardwareProfile* GetHardProfile()
{
	static HardwareProfile HardSingleton;
	return &HardSingleton;
}
//===================================================
GenericGPU* ProcessingGPU()
{	return GetHardProfile()->GetProcessingGPU();}
//===================================================
GenericGPU* RenderingGPU()
{	return GetHardProfile()->GetRenderingGPU();}
//===================================================
GenericGPU* MainGPU()
{	return GetHardProfile()->GetMainGPU();}
//===================================================

}//namespace GCV
