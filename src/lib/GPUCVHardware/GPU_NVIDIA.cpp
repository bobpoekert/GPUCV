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
#include <GPUCVHardware/GPU_NVIDIA.h>

namespace GCV{


//=================================================
GPU_NVIDIA::GPU_NVIDIA()
:GenericGPU()
{
	m_force_framebuffer_format = false;//!< \todo just to test
	
}
//=================================================
GPU_NVIDIA::~GPU_NVIDIA()
{
	
}
//=================================================
/*virtual*/
GLuint	GPU_NVIDIA::ConvertGLFormatToInternalFormat(GLuint _format, GLuint _type)const
{
	//we process generic cases
	GLuint InternalFormat = GenericGPU::ConvertGLFormatToInternalFormat(_format, _type);

	if(IsDefaultInternalFormatForced())
		return InternalFormat;
	//========================

	GLuint Formatbackup = InternalFormat;
	//we process specific cases that main not work on NVIDIA hardware.
#if 1
	switch(_format)
	{
	case GL_LUMINANCE :
	case GL_RED:
	case GL_BLUE:
	case GL_GREEN:
		switch(_type)
		{
		case GL_UNSIGNED_BYTE:	//InternalFormat = GL_RGB8UI_EXT;break;
		case GL_BYTE:			//InternalFormat = GL_RGB8I_EXT;break;
		case GL_UNSIGNED_SHORT:	//InternalFormat = GL_RGB16UI_EXT;break;
		case GL_SHORT:			InternalFormat = GL_RGB16_EXT;break;
		case GL_UNSIGNED_INT:	//InternalFormat = GL_RGB32UI_EXT;break;
		case GL_INT:			InternalFormat = GL_RGB32I_EXT;break;
		case GL_FLOAT:
			if(m_glExtension.IsFloat32Compatible())
				InternalFormat = GL_RGB32F_ARB;
			else
				InternalFormat = GL_RGB16F_ARB;
			break;
		case GL_DOUBLE:
			if(1)//m_glExtension.IsFloat32Compatible())
				InternalFormat = GL_RGB32F_ARB;
			else
				InternalFormat = GL_RGB16F_ARB;
			GPUCV_WARNING("GpuCV doesnt support 64f format therefore converting it into 32f format");
			break;
		}
		break;
	case GL_RGB:
	case GL_BGR:
		switch(_type)
		{
		case GL_UNSIGNED_BYTE:
		case GL_BYTE:			//InternalFormat = GL_RGB8I_EXT;break;
		case GL_UNSIGNED_SHORT:
		case GL_SHORT:			InternalFormat = GL_RGB16_EXT;break;
		case GL_UNSIGNED_INT:
		case GL_INT:			InternalFormat = GL_RGB32I_EXT;break;
		case GL_FLOAT:
			if(m_glExtension.IsFloat32Compatible())
				InternalFormat = GL_RGB32F_ARB;
			else
				InternalFormat = GL_RGB16F_ARB;
			break;
		case GL_DOUBLE:
			if(1)//m_glExtension.IsFloat32Compatible())
				InternalFormat = GL_RGB32F_ARB;
			else
				InternalFormat = GL_RGB16F_ARB;
			GPUCV_WARNING("GpuCV doesnt support 64f format therefore converting it into 32f format");
			break;
		}
		break;
	case GL_RGBA:
	case GL_BGRA:
		switch(_type)
		{
		case GL_UNSIGNED_BYTE:
		case GL_BYTE:			InternalFormat = GL_RGB8I_EXT;break;
		case GL_UNSIGNED_SHORT:
		case GL_SHORT:			InternalFormat = GL_RGB16I_EXT;break;
		case GL_UNSIGNED_INT:
		case GL_INT:			InternalFormat = GL_RGBA32I_EXT;break;
		case GL_FLOAT:
			if(m_glExtension.IsFloat32Compatible())
				InternalFormat = GL_RGBA32F_ARB;
			else
				InternalFormat = GL_RGBA16F_ARB;
			break;
		case GL_DOUBLE:
			if(1)//m_glExtension.IsFloat32Compatible())
				InternalFormat = GL_RGB32F_ARB;
			else
				InternalFormat = GL_RGB16F_ARB;
			GPUCV_WARNING("GpuCV doesnt support 64f format therefore converting it into 32f format");
			break;
		}
		break;

	default :
		//GPUCV_ERROR("Critical : ConvertGLFormatToInternalFormat()=> Unknown texture internal format...Using GL_RGBA16");
		//	 InternalFormat = GL_RGBA16;
		break;
	}
#endif
	if(Formatbackup != InternalFormat)
	{
		GPUCV_WARNING("Getting OpenGL internal format: Format selected("<< GetStrGLInternalTextureFormat(Formatbackup)<< ") is not compatible with you graphic card, we force a compatible format("<< GetStrGLInternalTextureFormat(InternalFormat)<< ")");
	}
	return InternalFormat;

}
//=================================================
/*virtual*/
bool GPU_NVIDIA::ReadGPUSettings()
{
	GenericGPU::ReadGPUSettings();

	if(m_glExtension.IsFloat32Compatible())
		m_default_framebuffer_format= GL_RGBA32F_ARB;

	ReadBrandSettings();
	//get pointer to main functions from the DLL

#ifdef _WINDOWS

	if(!LoadDriverLib("NVCPL.dll"))
	{
		GPUCV_WARNING("Can't read the NVIDIA driver's DLL: NVCPL.dll");
		return false;
	}
	else
	{
		m_FctGetDataInt = (NvCplGetDataIntType)::GetProcAddress(m_driverLib, "NvCplGetDataInt");
		m_FctSetDataInt = (NvCplSetDataIntType)::GetProcAddress(m_driverLib, "NvCplSetDataInt");

		if (!m_FctSetDataInt || !m_FctGetDataInt)
		{
			GPUCV_WARNING("Cound not find function 'NvCplGetDataInt' and 'NvCplSetDataInt' from NVCPL.dll");
			return false;
		}
	}

	//========================================
	//read NVIDIA specific hardware settings :
	//========================================

	// Get bus mode
	std::string MsgOutput;
	long BusMode=0;
	if (m_FctGetDataInt(NVCPL_API_AGP_BUS_MODE, &BusMode) == FALSE)
	{
		m_busMode = BUS_UNKNOWN;
		MsgOutput="Unable to retrieve";
	}
	else {
		switch (BusMode) {
			case NV_PCI:
				m_busMode = BUS_PCI;
				MsgOutput="PCI";
				break;
			case NV_AGP:
				m_busMode = BUS_AGP;
				MsgOutput="AGP";
				break;
			case NV_PCI_EXPRESS:
				m_busMode = BUS_PCI_EXP;
				MsgOutput="PCI Express";
				break;
			default:
				m_busMode=BUS_UNKNOWN;
				MsgOutput="Unknown";
				break;
		}
	}
	GPUCV_DEBUG("- Bus mode: " << MsgOutput);

	// Get bus transfer rate
	MsgOutput="- Bus transfer rate: ";
	if (m_FctGetDataInt(NV_DATA_TYPE_BUS_TRANSFER_RATE, &m_busTransferRate) == FALSE)
	{
		GPUCV_DEBUG(MsgOutput << "Unable to retrieve");
	}
	else
	{
		GPUCV_DEBUG(MsgOutput << m_busTransferRate);
	}



	// Get AGP memory size
	MsgOutput ="- AGP memory size: ";
	if (m_FctGetDataInt(NV_DATA_TYPE_AGP_MEMORY_SIZE, &m_agpMemSize) == FALSE)
	{
		GPUCV_DEBUG(MsgOutput << "Unable to retrieve");
	}
	else
		// AGP Memory Size is reported in Bytes: convert to MB before printing.
		GPUCV_DEBUG(MsgOutput << m_agpMemSize/(1024*1024));

	// Get video memory size
	MsgOutput = "- Video memory size: ";
	if (m_FctGetDataInt(NV_DATA_TYPE_VIDEO_MEMORY_SIZE, &m_videoMemSize) == FALSE)
	{
		GPUCV_DEBUG(MsgOutput <<"Unable to retrieve");
	}
	else
	{
		GPUCV_DEBUG(MsgOutput << m_videoMemSize);
	}

	// Get antialiasing mode
	/*        long AAMode;
	printf("- Antialiasing mode: ");
	if (m_FctGetDataInt(NV_DATA_TYPE_ANTIALIASING_MODE, &AAMode) == FALSE)
	printf("Unable to retrieve\n");
	else {
	switch (AAMode) {
	case NV_AA_MODE_OFF:
	printf("Off\n");
	break;
	case NV_AA_MODE_2X:
	printf("2X\n");
	break;
	case NV_AA_MODE_QUINCUNX:
	printf("Quincunx\n");
	break;
	case NV_AA_MODE_4X:
	printf("4X\n");
	break;
	case NV_AA_MODE_4X9T:
	printf("4X 9T\n");
	break;
	case NV_AA_MODE_4XS:
	printf("4X Skewed\n");
	break;
	case NV_AA_MODE_6XS:
	printf("6XS\n");
	break;
	case NV_AA_MODE_8XS:
	printf("8XS\n");
	break;
	case NV_AA_MODE_16X:
	printf("16XS\n");
	break;
	default:
	printf("Unknown\n");
	break;
	}
	}
	*/
	// Get and Modify Number of Frames Buffered
	/*      NvCplSetDataIntType m_FctSetDataInt = (NvCplSetDataIntType)::GetProcAddress(hLib, "NvCplSetDataInt");
	if (m_FctSetDataInt == 0)
	printf("- Unable to get a pointer to m_FctSetDataInt\n");
	else
	{
	long numFramesBuffered;
	printf("- Number of Frames Buffered: ");
	if (m_FctGetDataInt(NVCPL_API_FRAME_QUEUE_LIMIT, &numFramesBuffered) == FALSE)
	printf("Unable to retrieve\n");
	else
	printf("%d Frame(s)\n", numFramesBuffered);

	long const kForceBufferedFrames = 1;
	printf("- Setting Number of Frames Buffered to %d: ", kForceBufferedFrames);
	if (m_FctSetDataInt(NVCPL_API_FRAME_QUEUE_LIMIT, kForceBufferedFrames) == FALSE)
	printf("Unable to set\n");
	else
	{
	long    readbackValue;
	if (m_FctGetDataInt(NVCPL_API_FRAME_QUEUE_LIMIT, &readbackValue) == FALSE)
	printf("Unable to retrieve\n");
	else
	printf("%d Frame(s)\n", readbackValue);
	}
	printf("- Resetting Number of Frames Buffered to %d: ", numFramesBuffered);
	if (m_FctSetDataInt(NVCPL_API_FRAME_QUEUE_LIMIT, numFramesBuffered) == FALSE)
	printf("Unable to set\n");
	else
	{
	long    readbackValue;
	if (m_FctGetDataInt(NVCPL_API_FRAME_QUEUE_LIMIT, &readbackValue) == FALSE)
	printf("Unable to retrieve\n");
	else
	printf("%d Frame(s)\n", readbackValue);
	}
	}
	*/
	// Get number of GPUs and number of SLI GPUs
	MsgOutput = "- Number of GPUs: ";
	if (m_FctGetDataInt(NVCPL_API_NUMBER_OF_GPUS, &m_gpuNbr) == FALSE)
	{
		GPUCV_DEBUG(MsgOutput <<"Unable to retrieve");
	}
	else
	{
		GPUCV_DEBUG(MsgOutput << m_gpuNbr);
	}


	MsgOutput = "- Number of GPUs in SLI mode: ";
	long NbrGpuSliMode= 0L;
	if (m_FctGetDataInt(NVCPL_API_NUMBER_OF_SLI_GPUS, &NbrGpuSliMode) == FALSE)
	{
		GPUCV_DEBUG(MsgOutput << "Unable to retrieve");
	}
	else
	{
		GPUCV_DEBUG(MsgOutput <<NbrGpuSliMode);
	}

	if (NbrGpuSliMode > 0L)
	{
		//convert it into generic format
		long MultiGPUMode=0;
		MsgOutput ="- SLI rendering mode: ";
		if (m_FctGetDataInt(NVCPL_API_SLI_MULTI_GPU_RENDERING_MODE, &MultiGPUMode) == FALSE)
			MsgOutput +="Unable to retrieve";
		else
		{
			if ((MultiGPUMode & NVCPL_API_SLI_ENABLED) == 0L)
			{
				m_MultiGPUMode = MGPU_NO_SLI;
				MsgOutput +="SLI is not enabled";
			}
			else
			{
				if ((MultiGPUMode & NVCPL_API_SLI_RENDERING_MODE_AFR) != 0L)
				{
					m_MultiGPUMode = MGPU_AFR;
					MsgOutput +="SLI is in AFR mode";
				}
				else if ((MultiGPUMode & NVCPL_API_SLI_RENDERING_MODE_SFR) != 0L)
				{
					m_MultiGPUMode = MGPU_SFR;
					MsgOutput +="SLI is in SFR mode";
				}
				else if ((MultiGPUMode & NVCPL_API_SLI_RENDERING_MODE_SINGLE_GPU) != 0L)
				{
					m_MultiGPUMode = MGPU_Single;
					MsgOutput +="SLI is in single GPU mode";
				}
				else
				{
					m_MultiGPUMode = MGPU_AUTO;
					MsgOutput +="SLI is in auto-select mode";
				}

				/*
				printf("Setting SLI to AFR mode.\n");
				long    newSLIMode = NVCPL_API_SLI_RENDERING_MODE_AFR;
				if (m_FctSetDataInt(NVCPL_API_SLI_MULTI_GPU_RENDERING_MODE, newSLIMode) == FALSE)
				printf("Unable to set\n");
				else
				{
				// saving old SLI state for resetting later
				newSLIMode = SLIMode & (~NVCPL_API_SLI_ENABLED); // turn off top bit;
				// Query current state, should be AFR
				if (m_FctGetDataInt(NVCPL_API_SLI_MULTI_GPU_RENDERING_MODE, &SLIMode) == FALSE)
				printf("Unable to retrieve\n");
				else
				{
				if ((SLIMode & NVCPL_API_SLI_RENDERING_MODE_AFR) != 0L)
				printf("SLI is in AFR mode.\n");
				else if ((SLIMode & NVCPL_API_SLI_RENDERING_MODE_SFR) != 0L)
				printf("SLI is in SFR mode.\n");
				else if ((SLIMode & NVCPL_API_SLI_RENDERING_MODE_SINGLE_GPU) != 0L)
				printf("SLI is in single GPU mode.\n");
				else
				printf("SLI is in auto-select mode.\n");
				}
				// Reset to initial SLI mode
				if (m_FctSetDataInt(NVCPL_API_SLI_MULTI_GPU_RENDERING_MODE, newSLIMode) == FALSE)
				printf("Unable to reset SLI mode.\n");
				else
				printf("Reset SLI mode to initial state.\n");
				}
				*/
			}
		}
		GPUCV_DEBUG(MsgOutput);
	}

	return true;
#else
	return true;
#endif
}
//=================================================
bool GPU_NVIDIA::SetMultiGPUMode(MultiGPUMode _mode)
{
#ifdef _WINDOWS
	GenericGPU::SetMultiGPUMode(_mode);
	//long    newSLIMode = NVCPL_API_SLI_RENDERING_MODE_AFR;
	if(_mode!=MGPU_NO_SLI && _mode!=MGPU_Single)
	{
		if (m_FctSetDataInt(NVCPL_API_SLI_MULTI_GPU_RENDERING_MODE, _mode) == FALSE)
		{
			GPUCV_NOTICE("Unable to set MultiGPU mode\n");
			return false;
		}
		else
		{
			return false;
		}
	}
	return true;
	/*
	// saving old SLI state for resetting later
	newSLIMode = SLIMode & (~NVCPL_API_SLI_ENABLED); // turn off top bit;
	// Query current state, should be AFR
	if (m_FctGetDataInt(NVCPL_API_SLI_MULTI_GPU_RENDERING_MODE, &SLIMode) == FALSE)
	printf("Unable to retrieve\n");
	else
	{
	if ((SLIMode & NVCPL_API_SLI_RENDERING_MODE_AFR) != 0L)
	printf("SLI is in AFR mode.\n");
	else if ((SLIMode & NVCPL_API_SLI_RENDERING_MODE_SFR) != 0L)
	printf("SLI is in SFR mode.\n");
	else if ((SLIMode & NVCPL_API_SLI_RENDERING_MODE_SINGLE_GPU) != 0L)
	printf("SLI is in single GPU mode.\n");
	else
	printf("SLI is in auto-select mode.\n");
	}
	// Reset to initial SLI mode
	if (m_FctSetDataInt(NVCPL_API_SLI_MULTI_GPU_RENDERING_MODE, newSLIMode) == FALSE)
	printf("Unable to reset SLI mode.\n");
	else
	printf("Reset SLI mode to initial state.\n");
	}
	*/
#else
	return false;
#endif
}
//=================================================
//!\todo Add mobile card detection
bool GPU_NVIDIA::ReadBrandSettings()
{
	if (m_renderer == "")
	{
		GPUCV_ERROR("can't parse empty renderer name.\n");
		return false;
	}
	// 	 string Msg = "Log : Renderer Found :" +  renderer+"\n";
	//	 GPUCV_DEBUG(Msg.data());
	std::string model;

	size_t geforce= m_renderer.find("geforce");
	size_t quadro= m_renderer.find("quadro");
	size_t tesla= m_renderer.find("tesla");

	if (geforce!= std::string::npos)
	{//gforce card, wich one.?
		size_t GTX_200	= m_renderer.find("gtx 2");
		size_t GTX_400	= m_renderer.find("gtx 4");
		int m = 0;//atoi(model.data());

		m_hardwareFamily = HRD_FAM_UNKNOWN;
		if (GTX_200 != std::string::npos)
		{
			m_hardwareFamily = NV_GEFORCE_FAM;
			model = m_renderer.substr(GTX_200+4, 1);
			m = atoi(model.data());
			m+=10 -2;//G200 start at 10
		}
		else if (GTX_400 != std::string::npos)
		{
			m_hardwareFamily = NV_GEFORCE_FAM;
			model = m_renderer.substr(GTX_400+4, 1);
			m = atoi(model.data());
			m+=12 -2;//G400 start at 11?
		}
		else
		{
			m_hardwareFamily = NV_GEFORCE_FAM;
			model = m_renderer.substr(geforce + 8, 1);
			m = atoi(model.data());
		}

		m_hardwareFamily += m;
		m_brand = "NVIDIA";
		m_model = "GeForce " + ('0'+m);

		switch (m)
		{
		case 3 :
			{
				GPUCV_DEBUG("GeForce 3 detected : many shaders aren't compatible.");
				SetGLSLProfile(GenericGPU::HRD_PRF_1);
				break;
			}
		case 4 :
			{
				GPUCV_DEBUG("GeForce 4 detected : many shaders aren't compatible.");
				SetGLSLProfile(GenericGPU::HRD_PRF_1);
				break;
			}
		case 5 :
			{
				GPUCV_DEBUG("GeForce 5 detected : some shaders aren't compatible.");
				SetGLSLProfile(GenericGPU::HRD_PRF_2);
				break;
			}
		case 6 :
			{
				GPUCV_DEBUG("GeForce 6 detected : all shaders should be compatible.");
				SetGLSLProfile(GenericGPU::HRD_PRF_3);
				m_glExtension.SetFloat32Compatible(true);
				break;
			}

		case 7 :
			{
				GPUCV_DEBUG("GeForce 7 detected : all shaders should be compatible.");
				SetGLSLProfile(GenericGPU::HRD_PRF_3);
				m_glExtension.SetFloat32Compatible(true);
				break;
			}
		case 8 :
			{
				GPUCV_DEBUG("GeForce 8 detected : all shaders should be compatible.");
				SetGLSLProfile(GenericGPU::HRD_PRF_4);
				m_glExtension.SetFloat32Compatible(true);
				break;
			}
		case 9 :
			{
				GPUCV_DEBUG("GeForce 9 detected : all shaders should be compatible.");
				SetGLSLProfile(GenericGPU::HRD_PRF_4);
				m_glExtension.SetFloat32Compatible(true);
				break;
			}
		case 10 : //G200
			{
				GPUCV_DEBUG("GeForce G200 detected : all shaders should be compatible.");
				SetGLSLProfile(GenericGPU::HRD_PRF_4);
				m_glExtension.SetFloat32Compatible(true);
				m_supportDouble = true;
				break;
			}
		case 14 : //G400
			{
				GPUCV_DEBUG("GeForce G400 detected : all shaders should be compatible.");
				SetGLSLProfile(GenericGPU::HRD_PRF_4);
				m_glExtension.SetFloat32Compatible(true);
				m_supportDouble = true;
				break;
			}
		default :
			{
				GPUCV_WARNING("Warning : Card not recognize("<<m_renderer<<"). Using GLEW extensions to choose profile.");
				SetGLSLProfile(FindHardwareProfile());
				break;
			}
		}
		return true;
	}
	else if (quadro!= std::string::npos)
	{
		m_brand = "NVIDA";
		size_t Quadro = m_renderer.find("quadro fx");
		// According to GLView : all QuadroFX should work with gpuCV,
		// Other ones only have vertex shader
		if (m_renderer.find("quadro fx") != std::string::npos)
		{
			m_hardwareFamily = NV_QUADRO_FX_FAM;
			GPUCV_DEBUG("Quadro FX detected : all shaders should be compatible.");
			SetGLSLProfile(GenericGPU::HRD_PRF_3);
			m_model = "Quadro FX";
		}
		else if (m_renderer.find("Quadro NVS") != std::string::npos)
		{
			m_hardwareFamily = NV_QUADRO_NVS_FAM;
			GPUCV_DEBUG("Quadro NVS detected : all shaders should be compatible.");
			SetGLSLProfile(GenericGPU::HRD_PRF_3);
			m_model = "Quadro NVS";
		}
		else
		{
			m_hardwareFamily = NV_QUADRO_FAM;
			m_model = "Quadro";
			GPUCV_WARNING("Warning : Card not recognize. Using GLEW extensions to choose profile.");
			SetGLSLProfile(FindHardwareProfile());
		}
		return true;
	}
	return false;
}
//=================================================
#if _GPUCV_SUPPORT_GPU_PLUGIN

bool createGPU(GenericGPU * _GpuTable, int * _gpuNbr)
{
	return createGPU<GPU_NVIDIA>(_GpuTable, _gpuNbr);
}
#endif
//=================================================
}//namespace GCV
