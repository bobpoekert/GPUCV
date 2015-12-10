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
#include "GPUCVHardware/GenericGPU.h"

#if _WINDOWS 
	#include <windows.h>
	#include <strsafe.h>
#endif //_WINDOWS

namespace GCV{

//=====================================
GenericGPU::GenericGPU()//int _id)
://SGE::CL_BASE_OBJ<int>(_id),
#ifdef _WINDOWS
m_driverLib(NULL),
#endif
m_busMode(BUS_AGP),
m_busTransferRate(0),
m_agpMemSize(0),
m_videoMemSize(0),
m_MultiGPUMode(MGPU_NO_SLI),
m_gpuNbr(1),
m_hardwareFamily(HRD_FAM_UNKNOWN),
m_glsl_profile(HRD_PRF_0)
,m_default_framebuffer_format(_GPUCV_FRAMEBUFFER_DFLT_FORMAT)
,m_force_framebuffer_format(true)
,m_supportDouble(false)
{
}
//=====================================
GenericGPU::~GenericGPU()
{}
//=====================================
#ifdef _WINDOWS
HINSTANCE GenericGPU::LoadDriverLib(std::string _name)
{
#ifdef _WINDOWS
	//#define GPUCV_EXPORT "C" __declspec(dllexport)
#ifdef _VCC2005
	WCHAR  * a2 = NULL;
	int bufferSize = MultiByteToWideChar(CP_ACP, NULL, _name.c_str(), -1,
		(LPWSTR)a2, 0);
	a2 = (WCHAR  *)new TCHAR[bufferSize];

	int conversion = MultiByteToWideChar(CP_ACP, NULL, _name.c_str(), -1,
		(LPWSTR)a2, bufferSize);

	m_driverLib = ::LoadLibrary((LPCWSTR)a2);
#else
	m_driverLib = ::LoadLibrary(_name.c_str());

#endif
#else
	GUPCV_ERROR("GenericGPU::LoadDriverLib()=> not done fot LINUX and MAC OS");
#endif
	if (m_driverLib == 0)
	{
		GPUCV_ERROR ("Unable to load driver:" << _name);
#ifdef _WINDOWS
		DWORD dw = GetLastError(); 
		LPVOID lpMsgBuf;
		LPVOID lpDisplayBuf;
		
		FormatMessage(
			FORMAT_MESSAGE_ALLOCATE_BUFFER | 
			FORMAT_MESSAGE_FROM_SYSTEM |
			FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL,
			dw,
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
			(LPTSTR) &lpMsgBuf,
			0, NULL );

		// Display the error message and exit the process
		LPTSTR lpszFunction = TEXT("Loading DLL");
		lpDisplayBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT, 
			(lstrlen((LPCTSTR)lpMsgBuf) + lstrlen((LPCTSTR)(lpszFunction)) + 40) * sizeof(TCHAR)); 
		StringCchPrintf((LPTSTR)lpDisplayBuf, 
			LocalSize(lpDisplayBuf) / sizeof(TCHAR),
			TEXT("%s failed with error %d: %s"), 
			lpszFunction, dw, lpMsgBuf); 
		GPUCV_ERROR((LPCTSTR)lpDisplayBuf); 

		LocalFree(lpMsgBuf);
		LocalFree(lpDisplayBuf);
#endif
		return NULL;
	}
	return m_driverLib;
}
#endif
//=====================================
/*virtual*/
bool GenericGPU::SetMultiGPUMode(MultiGPUMode _mode)
{
	GPUCV_DEBUG("Changing MultiGPU mode.\n");
	std::string Text = "MultiGPU mode : ";
	m_MultiGPUMode = _mode;
	switch(_mode)
	{
	case MGPU_NO_SLI:	Text += "Only one GPU, SLI is not activated.";break;
	case MGPU_Single:	Text += "SLI activated but we use only one GPU.";break;
	case MGPU_AFR:		Text += "Each GPU works on its own Frame.";break;
	case MGPU_SFR:		Text += "Each frame is splitted for each GPU.";break;
	case MGPU_AUTO:		Text += "AUTO.";break;
	}
	Text+="\n";
	GPUCV_DEBUG(Text);

	return false;
}
//=====================================

/*virtual*/
bool GenericGPU::ReadGPUSettings()
{
	//read general opengl settings
	m_renderer = (char*)glGetString(GL_RENDERER);
	m_renderer = SGE::StringToLower (m_renderer);
	GPUCV_DEBUG("GenericGPU::ReadGPUSettings()=> renderer:"<< m_renderer);
	m_openGL_version = (char*)glGetString(GL_VERSION);
	GPUCV_DEBUG("GenericGPU::ReadGPUSettings()=> OpenGL version:"<< m_openGL_version);

	//std::ostringstream oss;
	std::istringstream input(m_openGL_version);
	float glVersion;
	input >> glVersion;

	if(glVersion>=2.0)
	{
		m_shading_version	= (char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
		m_shading_version = SGE::StringToLower (m_shading_version);
		GPUCV_DEBUG("GenericGPU::ReadGPUSettings()=> shading version:"<< m_shading_version);
	}

	m_glExtension.ReadExtension();

	if (m_renderer!="" && m_shading_version!="")
		return true;
	else
		return false;
}
//=====================================
/*virtual*/
bool GenericGPU::ReadBrandSettings()
{
	if (m_renderer == "")
	{
		GPUCV_ERROR("can't parse empty renderer name.\n");
		return false;
	}

	std::string TempRenderer=m_renderer;


	if (TempRenderer.find("intel"))
	{
		m_hardwareFamily = HRD_FAM_INTEL;
	}
	else if (TempRenderer.find("s3"))
	{
		m_hardwareFamily = HRD_FAM_S3;
	}
	else if (TempRenderer.find("matrox"))
	{
		m_hardwareFamily = HRD_FAM_MATROX;
	}
	else
	{
		GPUCV_WARNING("Graphics card not recognized by GpuCV ("<< m_renderer << ")");
		GPUCV_WARNING("Using GLEW extensions to choose profile.");
	}
	SetGLSLProfile(FindHardwareProfile());
	return true;
}
//=====================================
bool GenericGPU::IsHardwareFamily(HardwareFamilyBase _fam)
{
	int Res = m_hardwareFamily - _fam;
	if(Res < 0)
		return false;
	else if (Res >= 0x1000)
		return false;
	else
		return true;
}
//=====================================
const std::string &	GenericGPU::GetBrand()const
{	return m_brand;	}
//=====================================
const std::string &	GenericGPU::GetModel()const
{	return m_model;	}
//=====================================
const std::string &	GenericGPU::GetRenderer()const
{	return m_renderer;	}
//=====================================
int		GenericGPU::GetGLSLProfile()const
{	return m_glsl_profile;	}
//=====================================
void		GenericGPU::SetGLSLProfile(int _new_prof)
{	m_glsl_profile = _new_prof; }
//=====================================
GenericGPU::MultiGPUMode GenericGPU::GetMULTIGPUMode()const
{		return m_MultiGPUMode;	}
//=====================================
GLuint GenericGPU::ConvertGLFormatToInternalFormat(GLuint _format, GLuint _type)const
{
	if (IsDefaultInternalFormatForced())
		return GetDefaultInternalFormat();

	//==========================
	GLuint GLInternalFormat = 0;
	switch(_format)
	{
	case GL_BGR :
	case GL_RGB :
		switch(_type)
		{
		case GL_UNSIGNED_BYTE:	GLInternalFormat = GL_RGB8;break;//UI_EXT;break;
		case GL_BYTE:			GLInternalFormat = GL_RGB8;break;//I_EXT;break;
		case GL_UNSIGNED_SHORT:	GLInternalFormat = GL_RGB16UI_EXT;break;//UI_EXT;break;
		case GL_SHORT:			GLInternalFormat = GL_RGB16I_EXT;break;//I_EXT;break;
		case GL_UNSIGNED_INT:	GLInternalFormat = GL_RGB32UI_EXT;break;//UI_EXT;break;
		case GL_INT:			GLInternalFormat = GL_RGB32I_EXT;break;//I_EXT;break;
		case GL_FLOAT:
			if(1)//m_glExtension.IsFloat32Compatible())
				GLInternalFormat = GL_RGB32F_ARB;
			else
				GLInternalFormat = GL_RGB16F_ARB;
			break;
		case GL_DOUBLE:
			if(1)//m_glExtension.IsFloat32Compatible())
				GLInternalFormat = GL_RGB32F_ARB;
			else
				GLInternalFormat = GL_RGB16F_ARB;
			GPUCV_WARNING("GpuCV doesnt support 64f format therefore converting it into 32f format");
			break;
		}
		break;
	case GL_BGRA:
	case GL_RGBA :
		switch(_type)
		{
		case GL_UNSIGNED_BYTE:	GLInternalFormat = GL_RGBA8;break;//UI_EXT;break;
		case GL_BYTE:			GLInternalFormat = GL_RGBA8;break;//I_EXT;break;
		case GL_UNSIGNED_SHORT:	GLInternalFormat = GL_RGBA16;break;//UI_EXT;break;
		case GL_SHORT:			GLInternalFormat = GL_RGBA16;break;//I_EXT;break;
		case GL_UNSIGNED_INT:	GLInternalFormat = GL_RGBA32UI_EXT;break;//UI_EXT;break;
		case GL_INT:			GLInternalFormat = GL_RGBA32I_EXT;break;//I_EXT;break;
		case GL_FLOAT:
			if(1)//m_glExtension.IsFloat32Compatible())
				GLInternalFormat = GL_RGBA32F_ARB;
			else
				GLInternalFormat = GL_RGBA16F_ARB;
			break;
		case GL_DOUBLE:
			if(1)//m_glExtension.IsFloat32Compatible())
				GLInternalFormat = GL_RGB32F_ARB;
			else
				GLInternalFormat = GL_RGB16F_ARB;
			GPUCV_WARNING("GpuCV doesnt support 64f format therefore converting it into 32f format");
			break;
		}
		break;
	case GL_ALPHA :
		switch(_type)
		{
		case GL_UNSIGNED_BYTE:	GLInternalFormat = GL_ALPHA8;break;//UI_EXT;break;
		case GL_BYTE:			GLInternalFormat = GL_ALPHA8;break;//I_EXT;break;
		case GL_UNSIGNED_SHORT:	GLInternalFormat = GL_ALPHA16;break;//UI_EXT;break;
		case GL_SHORT:			GLInternalFormat = GL_ALPHA16;break;//I_EXT;break;
		case GL_UNSIGNED_INT:	GLInternalFormat = GL_ALPHA32UI_EXT;break;
		case GL_INT:			GLInternalFormat = GL_ALPHA32I_EXT;break;
		case GL_FLOAT:
			if(1)//m_glExtension.IsFloat32Compatible())
				GLInternalFormat = GL_ALPHA32F_ARB;
			else
				GLInternalFormat = GL_ALPHA16F_ARB;
			break;
		case GL_DOUBLE:
			if(1)//m_glExtension.IsFloat32Compatible())
				GLInternalFormat = GL_RGB32F_ARB;
			else
				GLInternalFormat = GL_RGB16F_ARB;
			GPUCV_WARNING("GpuCV doesnt support 64f format therefore converting it into 32f format");
			break;
		}
		break;
	case GL_LUMINANCE :
	case GL_RED:
	case GL_BLUE:
	case GL_GREEN:
		switch(_type)
		{
		case GL_UNSIGNED_BYTE:	GLInternalFormat = GL_LUMINANCE8UI_EXT;break;
		case GL_BYTE:			GLInternalFormat = GL_LUMINANCE8I_EXT;break;
		case GL_UNSIGNED_SHORT:	GLInternalFormat = GL_LUMINANCE16UI_EXT;break;
		case GL_SHORT:			GLInternalFormat = GL_LUMINANCE16I_EXT;break;
		case GL_UNSIGNED_INT:	GLInternalFormat = GL_LUMINANCE32UI_EXT;break;
		case GL_INT:			GLInternalFormat = GL_LUMINANCE32I_EXT;break;
		case GL_FLOAT:
			if(1)//m_glExtension.IsFloat32Compatible())
				GLInternalFormat = GL_LUMINANCE32F_ARB;
			else
				GLInternalFormat = GL_LUMINANCE16F_ARB;
			break;
		case GL_DOUBLE:
			if(1)//m_glExtension.IsFloat32Compatible())
				GLInternalFormat = GL_RGB32F_ARB;
			else
				GLInternalFormat = GL_RGB16F_ARB;
			GPUCV_WARNING("GpuCV doesnt support 64f format therefore converting it into 32f format");
			break;
		}
		break;
	case GL_LUMINANCE_ALPHA:
		switch(_type)
		{
		case GL_UNSIGNED_BYTE:	GLInternalFormat = GL_LUMINANCE8_ALPHA8_EXT;break;
		case GL_BYTE:			GLInternalFormat = GL_LUMINANCE8_ALPHA8_EXT;break;
		case GL_UNSIGNED_SHORT:	GLInternalFormat = GL_LUMINANCE16_ALPHA16_EXT;break;
		case GL_SHORT:			GLInternalFormat = GL_LUMINANCE16_ALPHA16_EXT;break;
		case GL_UNSIGNED_INT:	GLInternalFormat = GL_LUMINANCE_ALPHA_INTEGER_EXT;break;
		case GL_INT:			GLInternalFormat = GL_LUMINANCE_ALPHA_INTEGER_EXT;break;
		case GL_FLOAT:
			if(1)//m_glExtension.IsFloat32Compatible())
				GLInternalFormat = GL_LUMINANCE_ALPHA_FLOAT16_ATI;
			else
				GLInternalFormat = GL_LUMINANCE_ALPHA_FLOAT32_ATI;
			break;
		case GL_DOUBLE:
			if(1)//m_glExtension.IsFloat32Compatible())
				GLInternalFormat = GL_RGB32F_ARB;
			else
				GLInternalFormat = GL_RGB16F_ARB;
			GPUCV_WARNING("GpuCV doesnt support 64f format therefore converting it into 32f format");
			break;
		}
		break;

	default :
		GPUCV_ERROR("Critical : ConvertGLFormatToInternalFormat()=> Unknown texture internal format...Using default");
		GLInternalFormat = GetDefaultInternalFormat();
		break;
	}
	return GLInternalFormat;
}
//=====================================
GenericGPU::HardwareProfile GenericGPU::FindHardwareProfile()
{
	HardwareProfile hardProf = HRD_PRF_0;

	//Check for Profile 1
	if(
		//vertex shader
		GLEW_ARB_vertex_program
		||	GLEW_ARB_vertex_shader
		//	||	GLEW_ATI_vertex_shader
		||	GLEW_NV_vertex_program
		||	GLEW_EXT_vertex_shader
		)
	{
		hardProf = HRD_PRF_1;
	}

	//Check for Profile 2
	if(	hardProf == HRD_PRF_1 &&
		//fragment shader
		GLEW_ARB_fragment_program
		||	GLEW_ARB_fragment_shader
		||	GLEW_ATI_fragment_shader
		||	GLEW_NV_fragment_program
		//||	GLEW_EXT_fragment_shader
		)
	{
		hardProf = HRD_PRF_2;
	}

	//Check for Profile 3
	//! \todo Find a way to check that dynamic branching is available for profile HRD_PRF_3
	//! \todo Find a way to check that vertex shader lookup is available for profile HRD_PRF_3
	if(	hardProf == HRD_PRF_2 &&
		//dynamic branching
		1//assume it is true
		//vertex shader lookup
		||	1//assume it is true
		//Multiple render target
		||	m_glExtension.m_multipleRenderTarget
		)
	{
		hardProf = HRD_PRF_3;
	}

	//Check for Profile 3
#if 0
	if(	hardProf == HRD_PRF_3 &&
		//unified shaders
		1//m_multipleRenderTarget
		)
	{
		hardProf = HRD_PRF_4;
	}
#endif
	return hardProf;
}

//=====================================
std::string GenericGPU::ExportToHTMLTable()
{
	std::string TempStr = "<H2>Graphic card description:</H2>";
	TempStr += HTML_OPEN_TABLE;
	TempStr += HTML_OPEN_ROW + HTML_CELL("Brand") + HTML_CELL(m_brand)+ HTML_CLOSE_ROW;
	//TempStr += HTML_OPEN_ROW + HTML_CELL("Model") + HTML_CELL(m_model)+ HTML_CLOSE_ROW;
	TempStr += HTML_OPEN_ROW + HTML_CELL("Full description") + HTML_CELL(m_renderer)+ HTML_CLOSE_ROW;
	//TempStr += HTML_OPEN_ROW + HTML_CELL("AGP aperture Memory size") + HTML_CELL(SGE::ToCharStr(m_agpMemSize))+ HTML_CLOSE_ROW;
	TempStr += HTML_OPEN_ROW + HTML_CELL("Video memory size") + HTML_CELL(SGE::ToCharStr(m_videoMemSize))+ HTML_CLOSE_ROW;
	TempStr += HTML_OPEN_ROW + HTML_CELL("Graphic port") + HTML_CELL(GPUCV_BUS_STR[m_busMode])+ HTML_CLOSE_ROW;
	//TempStr += HTML_OPEN_ROW + HTML_CELL("Nbr. of GPU(s)") + HTML_CELL(SGE::ToCharStr(m_gpuNbr))+ HTML_CLOSE_ROW;
	//TempStr += HTML_OPEN_ROW + HTML_CELL("Multi-GPUs mode") + HTML_CELL(GPUCV_MULTIGPU_STR[m_MultiGPUMode])+ HTML_CLOSE_ROW;
	//TempStr += HTML_OPEN_ROW + HTML_CELL("Shading langage version") + HTML_CELL(m_shading_version)+ HTML_CLOSE_ROW;
	//TempStr += HTML_OPEN_ROW + HTML_CELL("Driver version") + HTML_CELL("")+ HTML_CLOSE_ROW;
	//TempStr += HTML_OPEN_ROW + HTML_CELL("Driver date") + HTML_CELL("")+ HTML_CLOSE_ROW;
	TempStr += HTML_CLOSE_TABLE;
	return TempStr;
}
//=====================================
}//namespace GCV
