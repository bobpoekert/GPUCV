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
#include "GPUCVTexture/pbuffer.h"

namespace GCV{


#if _GPUCV_DEBUG_MODE
#ifndef PB_FPF
#if defined DEBUG || defined _GPUCV_DEBUG_MODE
#define PB_FPF fprintf
#endif
#endif
#endif

#ifndef PB_FPF
#if _MSC_VER == 1310
#define PB_FPF()//...)
#else
//#define PB_FPF(...) //do { } while (false)
#define PB_FPF(...)do { } while (false)
#endif
#endif

#ifdef _WINDOWS
PBuffer::PBuffer(const char *strMode, bool managed)
: m_hDC(0), m_hGLRC(0), m_hPBuffer(0), m_hOldGLRC(0), m_hOldDC(0),
m_bIsTexture(false), m_iWidth(0), m_iHeight(0), m_strMode(strMode),
m_bSharedContext(false), m_bShareObjects(false), m_bIsBound(false),
m_bIsActive(false), m_bManaged(managed)
{
	m_pfAttribList.push_back(WGL_DRAW_TO_PBUFFER_ARB);
	m_pfAttribList.push_back(true);
	m_pfAttribList.push_back(WGL_SUPPORT_OPENGL_ARB);
	m_pfAttribList.push_back(true);

	//Yann.A. 18/10/05 : test Pbuffer with ATI...
	if(ProcessingGPU()->IsHardwareFamily(GenericGPU::HRD_FAM_ATI))
	{
		m_pfAttribList.push_back(WGL_BIND_TO_TEXTURE_RGBA_ARB);
		m_pfAttribList.push_back(true);
		m_pfAttribList.push_back(WGL_PBUFFER_LARGEST_ARB);
		m_pfAttribList.push_back(true);
		m_pfAttribList.push_back(WGL_RED_BITS_ARB);
		m_pfAttribList.push_back(8);
		m_pfAttribList.push_back(WGL_GREEN_BITS_ARB);
		m_pfAttribList.push_back(8);
		m_pfAttribList.push_back(WGL_BLUE_BITS_ARB);
		m_pfAttribList.push_back(8);
		m_pfAttribList.push_back(WGL_ALPHA_BITS_ARB);
		m_pfAttribList.push_back(8);
		m_pfAttribList.push_back(WGL_DEPTH_BITS_ARB);
		m_pfAttribList.push_back(24);

	}
	//====================================================

	m_pbAttribList.push_back(WGL_PBUFFER_LARGEST_ARB);
	m_pbAttribList.push_back(true);
	//Yann.A. 18/10/05 : test Pbuffer with ATI...
	if(ProcessingGPU()->IsHardwareFamily(GenericGPU::HRD_FAM_ATI))
	{
		m_pbAttribList.push_back(WGL_TEXTURE_FORMAT_ARB);
		m_pbAttribList.push_back(WGL_TEXTURE_RGBA_ARB);
		m_pbAttribList.push_back(WGL_TEXTURE_TARGET_ARB);
		m_pbAttribList.push_back(WGL_TEXTURE_2D_ARB);
	}
	//====================================================

	PB_FPF(stdout, "Declare a Pbuffer with \"%s\" parameters\n", strMode);
	m_strMode = strMode;
	parseModeString(m_strMode, &m_pfAttribList, &m_pbAttribList);

	m_pfAttribList.push_back(0);
	m_pbAttribList.push_back(0);
}

PBuffer::~PBuffer()
{
	if (m_bManaged)
		Destroy();
}


bool PBuffer::Initialize(int iWidth, int iHeight, bool bShareContexts, bool bShareObjects)
{
	HDC hdc = wglGetCurrentDC();
	HGLRC hglrc = wglGetCurrentContext();
	int format = 0;
	unsigned int nformats=0;
	int nfattribs = 0;
	int niattribs = 0;

	m_iWidth = iWidth;
	m_iHeight = iHeight;

	m_bSharedContext = bShareContexts;
	m_bShareObjects = bShareObjects;

	if (m_bSharedContext || ProcessingGPU()->IsHardwareFamily(GenericGPU::HRD_FAM_ATI))//Yann.Al. shared context ignored with ATI RADEON...???fix it later...
	{
		// Get the pixel format for the on-screen window.
		format = GetPixelFormat(hdc);
		if (format == 0)
		{
			PB_FPF(stderr, "pbuffer creation error:  GetPixelFormat() failed\n");
			return false;
		}
	}
	else
	{
		wglChoosePixelFormatARB(hdc, &m_pfAttribList[0], NULL, 1, &format, &nformats);
		if (nformats == 0)
		{//Yann.Al. does not work on ATI RADEON 9700...???
			PB_FPF(stderr, "pbuffer creation error:  Couldn't find a suitable pixel format.\n");
			return false;
		}
	}

	m_hPBuffer = wglCreatePbufferARB(hdc, format, m_iWidth, m_iHeight, &m_pbAttribList[0]);
	if (!m_hPBuffer)
	{
		DWORD err = GetLastError();
		PB_FPF(stderr, "pbuffer creation error:  wglCreatePbufferARB() failed\n");
		if (err == ERROR_INVALID_PIXEL_FORMAT)
		{
			PB_FPF(stderr, "error:  ERROR_INVALID_PIXEL_FORMAT\n");
		}
		else if (err == ERROR_NO_SYSTEM_RESOURCES)
		{
			PB_FPF(stderr, "error:  ERROR_NO_SYSTEM_RESOURCES\n");
		}
		else if (err == ERROR_INVALID_DATA)
		{
			PB_FPF(stderr, "error:  ERROR_INVALID_DATA\n");
		}
		else  PB_FPF(stderr, "error:  Unknwon\n");

		return false;
	}

	// Get the device context.
	m_hDC = wglGetPbufferDCARB(m_hPBuffer);
	if (!m_hDC)
	{
		PB_FPF(stderr, "pbuffer creation error:  wglGetPbufferDCARB() failed\n");
		return false;
	}

	if (m_bSharedContext)
	{
		// Let's use the same gl context..
		// Since the device contexts are compatible (i.e. same pixelformat),
		// we should be able to use the same gl rendering context.
		m_hGLRC = hglrc;
	}
	else
	{
		// Create a new gl context for the p-buffer.
		m_hGLRC = wglCreateContext(m_hDC);
		if (!m_hGLRC)
		{
			PB_FPF(stderr, "pbuffer creation error:  wglCreateContext() failed\n");
			return false;
		}

		if(m_bShareObjects)
		{
			if(!wglShareLists(hglrc, m_hGLRC))
			{
				PB_FPF(stderr, "pbuffer: wglShareLists() failed\n");
				return false;
			}
		}
	}

	GLint texFormat = WGL_NO_TEXTURE_ARB;
	wglQueryPbufferARB(m_hPBuffer, WGL_TEXTURE_FORMAT_ARB, &texFormat);

	if (texFormat != WGL_NO_TEXTURE_ARB)
		m_bIsTexture = true;

	// Determine the actual width and height we were able to create.
	wglQueryPbufferARB(m_hPBuffer, WGL_PBUFFER_WIDTH_ARB, &m_iWidth);
	wglQueryPbufferARB(m_hPBuffer, WGL_PBUFFER_HEIGHT_ARB, &m_iHeight);

	PB_FPF(stdout, "Created a %d x %d pbuffer\n", m_iWidth, m_iHeight);

#if _GPUCV_DEBUG_MODE
	// query pixel format
	int iattributes[] = {
		WGL_RED_BITS_ARB,
		WGL_GREEN_BITS_ARB,
		WGL_BLUE_BITS_ARB,
		WGL_ALPHA_BITS_ARB,
		WGL_FLOAT_COMPONENTS_NV,
		WGL_DEPTH_BITS_ARB,
		WGL_SAMPLES_EXT,
		WGL_AUX_BUFFERS_ARB
	};
	int ivalues[sizeof(iattributes) / sizeof(int)];

	if (wglGetPixelFormatAttribivARB(m_hDC, format, 0, sizeof(iattributes) / sizeof(int), iattributes, ivalues)) {
		PB_FPF(stdout, "r:%d g:%d b:%d a:%d float:%d depth:%d samples:%d aux:%d\n",
			ivalues[0], ivalues[1], ivalues[2], ivalues[3], ivalues[4], ivalues[5], ivalues[6], ivalues[7]);
	}
#endif

	return true;
}

void PBuffer::Destroy()
{
	if (m_hPBuffer)
	{
		if (!m_bSharedContext) wglDeleteContext(m_hGLRC);
		wglReleasePbufferDCARB(m_hPBuffer, m_hDC);
		wglDestroyPbufferARB(m_hPBuffer);
	}
}

void PBuffer::parseModeString(const char *modeString, vector<int> *pfAttribList, vector<int> *pbAttribList)
{
	if (!modeString || strcmp(modeString, "") == 0)
		return;

	m_iBitsPerComponent = 8;
	m_iNComponents = 0;
	bool bIsFloatBuffer = false;
	bool bIsATIFloatBuffer = false;
	bool bIsTexture = false;
	bool bNeedAlpha = false;

	char *mode = _strdup(modeString);

	vector<string> tokens;
	char *buf = strtok(mode, " ");
	while (buf != NULL)
	{
		if (strstr(buf, "ati_float") != NULL)
			bIsATIFloatBuffer = true;
		else if (strstr(buf, "float") != NULL)
			bIsFloatBuffer = true;

		if (strstr(buf, "texture") != NULL)
			bIsTexture = true;

		if (strstr(buf, "alpha") != NULL)
			bNeedAlpha = true;

		tokens.push_back(buf);
		buf = strtok(NULL, " ");
	}

	pfAttribList->push_back(WGL_PIXEL_TYPE_ARB);
#ifdef WGL_ATI_pixel_format_float
	if (bIsATIFloatBuffer) {
		pfAttribList->push_back(WGL_TYPE_RGBA_FLOAT_ATI);
	} else
#endif
	{
		pfAttribList->push_back(WGL_TYPE_RGBA_ARB);
	}

	for (unsigned int i = 0; i < tokens.size(); i++)
	{
		string token = tokens[i];

		if (token == "rgb" && (m_iNComponents <= 1))
		{
			/*            pfAttribList->push_back(WGL_RED_BITS_ARB);
			pfAttribList->push_back(m_iBitsPerComponent);
			pfAttribList->push_back(WGL_GREEN_BITS_ARB);
			pfAttribList->push_back(m_iBitsPerComponent);
			pfAttribList->push_back(WGL_BLUE_BITS_ARB);
			pfAttribList->push_back(m_iBitsPerComponent);*/
			m_iNComponents += 3;
			continue;
		}
		else if (token == "rgb") PB_FPF(stderr, "warning : mistake in components definition (rgb + %d)\n", m_iNComponents);


		if (token == "rgba" && (m_iNComponents == 0))
		{
			/*pfAttribList->push_back(WGL_RED_BITS_ARB);
			pfAttribList->push_back(m_iBitsPerComponent);
			pfAttribList->push_back(WGL_GREEN_BITS_ARB);
			pfAttribList->push_back(m_iBitsPerComponent);
			pfAttribList->push_back(WGL_BLUE_BITS_ARB);
			pfAttribList->push_back(m_iBitsPerComponent);
			pfAttribList->push_back(WGL_ALPHA_BITS_ARB);
			pfAttribList->push_back(m_iBitsPerComponent);*/
			m_iNComponents = 4;
			continue;
		}
		else if (token == "rgba") PB_FPF(stderr, "warning : mistake in components definition (rgba + %d)\n", m_iNComponents);

		if (token == "alpha" && (m_iNComponents <= 3))
		{
			/*pfAttribList->push_back(WGL_ALPHA_BITS_ARB);
			pfAttribList->push_back(m_iBitsPerComponent);*/
			m_iNComponents++;
			continue;
		}
		else if (token == "alpha") PB_FPF(stderr, "warning : mistake in components definition (alpha + %d)\n", m_iNComponents);


		if (token == "r" && (m_iNComponents <= 1))// && bIsFloatBuffer)
		{
			/*pfAttribList->push_back(WGL_RED_BITS_ARB);
			pfAttribList->push_back(m_iBitsPerComponent);*/
			m_iNComponents++;
			continue;
		}
		else if (token == "r") PB_FPF(stderr, "warning : mistake in components definition (r + %d)\n", m_iNComponents);

		if (token == "rg" && (m_iNComponents <= 1))// && bIsFloatBuffer)
		{
			/*pfAttribList->push_back(WGL_RED_BITS_ARB);
			pfAttribList->push_back(m_iBitsPerComponent);
			pfAttribList->push_back(WGL_GREEN_BITS_ARB);
			pfAttribList->push_back(m_iBitsPerComponent);*/
			m_iNComponents += 2;
			continue;
		}
		else if (token == "r") PB_FPF(stderr, "warning : mistake in components definition (rg + %d)\n", m_iNComponents);

		if (token.find("depth") == 0)
		{
			pfAttribList->push_back(WGL_DEPTH_BITS_ARB);
			pfAttribList->push_back(getIntegerValue(token));

			continue;
		}

		if (token.find("stencil") == 0)
		{
			pfAttribList->push_back(WGL_STENCIL_BITS_ARB);
			pfAttribList->push_back(8);

			continue;
		}

		if (token.find("samples") == 0)
		{
			pfAttribList->push_back(WGL_SAMPLE_BUFFERS_ARB);
			pfAttribList->push_back(1);
			pfAttribList->push_back(WGL_SAMPLES_ARB);
			pfAttribList->push_back(getIntegerValue(token));

			continue;
		}

		if (token.find("aux") == 0)
		{
			pfAttribList->push_back(WGL_AUX_BUFFERS_ARB);
			pfAttribList->push_back(getIntegerValue(token));
			continue;
		}

		if (token == "double")
		{
			pfAttribList->push_back(WGL_DOUBLE_BUFFER_ARB);
			pfAttribList->push_back(true);

			continue;
		}

		if (token.find("ati_float") == 0)
		{
			m_iBitsPerComponent = getIntegerValue(token);
			// type already set above
			continue;

		} else if (token.find("float") == 0)
		{
			m_iBitsPerComponent = getIntegerValue(token);
			//bIsFloatBuffer = true; done previously
			pfAttribList->push_back(WGL_FLOAT_COMPONENTS_NV);
			pfAttribList->push_back(true);

			continue;
		}

		if (token.find("texture") == 0)
		{
			if (token.find("textureRECT") == 0 || bIsFloatBuffer)
			{
				pbAttribList->push_back(WGL_TEXTURE_TARGET_ARB);
				pbAttribList->push_back(WGL_TEXTURE_RECTANGLE_NV);
			}
			else if (token.find("textureCUBE") == 0)
			{
				pbAttribList->push_back(WGL_TEXTURE_TARGET_ARB);
				pbAttribList->push_back(WGL_TEXTURE_CUBE_MAP_ARB);
			}
			else
			{
				pbAttribList->push_back(WGL_TEXTURE_TARGET_ARB);
				pbAttribList->push_back(WGL_TEXTURE_2D_ARB);
			}

			if (bIsFloatBuffer || bIsATIFloatBuffer)
			{
				if(m_iNComponents == 0)
				{
					PB_FPF(stderr, "components not specified. assuming rgba...\n");
					pfAttribList->push_back(WGL_RED_BITS_ARB);
					pfAttribList->push_back(m_iBitsPerComponent);
					pfAttribList->push_back(WGL_GREEN_BITS_ARB);
					pfAttribList->push_back(m_iBitsPerComponent);
					pfAttribList->push_back(WGL_BLUE_BITS_ARB);
					pfAttribList->push_back(m_iBitsPerComponent);
					pfAttribList->push_back(WGL_ALPHA_BITS_ARB);
					pfAttribList->push_back(m_iBitsPerComponent);
					m_iNComponents = 4;
				}
			}

			if (bIsFloatBuffer) {
				switch(m_iNComponents)
				{
				case 1:
					pfAttribList->push_back(WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_R_NV);
					pfAttribList->push_back(true);

					pbAttribList->push_back(WGL_TEXTURE_FORMAT_ARB);
					pbAttribList->push_back(WGL_TEXTURE_FLOAT_R_NV);
					break;
				case 2:
					pfAttribList->push_back(WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RG_NV);
					pfAttribList->push_back(true);

					pbAttribList->push_back(WGL_TEXTURE_FORMAT_ARB);
					pbAttribList->push_back(WGL_TEXTURE_FLOAT_RG_NV);
					break;
				case 3:
					pfAttribList->push_back(WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGB_NV);
					pfAttribList->push_back(true);

					pbAttribList->push_back(WGL_TEXTURE_FORMAT_ARB);
					pbAttribList->push_back(WGL_TEXTURE_FLOAT_RGB_NV);
					break;
				case 4:
					pfAttribList->push_back(WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV);
					pfAttribList->push_back(true);

					pbAttribList->push_back(WGL_TEXTURE_FORMAT_ARB);
					pbAttribList->push_back(WGL_TEXTURE_FLOAT_RGBA_NV);
					break;
				default:
					PB_FPF(stderr, "Bad number of components (r=1,rg=2,rgb=3,rgba=4): %d\n", m_iNComponents);
					break;
				}
			}
			else
			{
				switch(m_iNComponents)
				{
				case 3:
					pfAttribList->push_back(WGL_BIND_TO_TEXTURE_RGB_ARB);
					pfAttribList->push_back(true);

					pbAttribList->push_back(WGL_TEXTURE_FORMAT_ARB);
					pbAttribList->push_back(WGL_TEXTURE_RGB_ARB);
					break;
				case 4:
					pfAttribList->push_back(WGL_BIND_TO_TEXTURE_RGBA_ARB);
					pfAttribList->push_back(true);

					pbAttribList->push_back(WGL_TEXTURE_FORMAT_ARB);
					pbAttribList->push_back(WGL_TEXTURE_RGBA_ARB);
					break;
				default:
					PB_FPF(stderr, "Bad number of components (r=1,rg=2,rgb=3,rgba=4): %d\n", m_iNComponents);
					break;
				}
			}

			string option = getStringValue(token);
			if (option == "depth")
			{
				pfAttribList->push_back(WGL_BIND_TO_TEXTURE_DEPTH_NV);
				pfAttribList->push_back(true);

				pbAttribList->push_back(WGL_DEPTH_TEXTURE_FORMAT_NV);
				pbAttribList->push_back(WGL_TEXTURE_DEPTH_COMPONENT_NV);
			}

			continue;
		}

		if (token.find("mipmap") == 0 && bIsTexture)
		{
			pbAttribList->push_back(WGL_MIPMAP_TEXTURE_ARB);
			pbAttribList->push_back(true);

			continue;
		}

		PB_FPF(stderr, "unknown pbuffer attribute: %s\n", token.c_str());
	}

	if (m_iNComponents > 0)
	{
		pfAttribList->push_back(WGL_RED_BITS_ARB);
		pfAttribList->push_back(m_iBitsPerComponent);
	}
	if (m_iNComponents > 1)
	{
		pfAttribList->push_back(WGL_GREEN_BITS_ARB);
		pfAttribList->push_back(m_iBitsPerComponent);
	}
	if (m_iNComponents > 2)
	{
		pfAttribList->push_back(WGL_BLUE_BITS_ARB);
		pfAttribList->push_back(m_iBitsPerComponent);
	}
	if (m_iNComponents > 3)
	{
		pfAttribList->push_back(WGL_ALPHA_BITS_ARB);
		pfAttribList->push_back(m_iBitsPerComponent);
	}
}


void PBuffer::HandleModeSwitch()
{
	int lost = 0;

	wglQueryPbufferARB(m_hPBuffer, WGL_PBUFFER_LOST_ARB, &lost);

	if (lost)
	{
		this->~PBuffer();
		Initialize(m_iWidth, m_iHeight, m_bSharedContext, m_bShareObjects);
	}
}

int PBuffer::Bind(int iBuffer)
{
	if (!m_bIsTexture)
	{
		PB_FPF(stderr, "PBuffer::Bind() failed - pbuffer format does not support render to texture!\n");
		return 0;
	}

#if 0
	// SGG - with MRT it is legal to bind different buffers of a pbuffer simultaneously
	if (m_bIsBound)
	{
		PB_FPF(stderr, "PBuffer::Bind() failed - pbuffer is already bound.\n");
		return 0;
	}
#endif

	int ret = wglBindTexImageARB(m_hPBuffer, iBuffer);
	if (!ret)
		PB_FPF(stderr, "PBuffer::Bind() failed.\n");

	m_bIsBound = true;

	return ret;
}

int PBuffer::Release(int iBuffer)
{
	if (!m_bIsTexture)
	{
		PB_FPF(stderr, "PBuffer::Release() failed - pbuffer format does not support render to texture!\n");
		return 0;
	}

#if 0
	// SGG - with MRT it is legal to bind different buffers of a pbuffer simultaneously
	if (!m_bIsBound)
	{
		PB_FPF(stderr, "PBuffer::Release() failed - pbuffer is not bound.\n");
		return 0;
	}
#endif

	int ret = wglReleaseTexImageARB(m_hPBuffer, iBuffer);
	if (!ret)
		PB_FPF(stderr, "PBuffer::Release() failed.\n");

	m_bIsBound = false;

	return ret;
}


void PBuffer::Activate(PBuffer *current /* = NULL */)
{
	if (current == this)
	{
		return; // no switch necessary
	}

	if (NULL == current || !current->m_bIsActive)
	{
		if (m_bIsActive)
			return;

		m_hOldGLRC = wglGetCurrentContext();
		m_hOldDC = wglGetCurrentDC();
	}
	else
	{
		m_hOldGLRC = current->m_hOldGLRC;
		m_hOldDC = current->m_hOldDC;
		current->m_hOldGLRC = 0;
		current->m_hOldDC = 0;
		current->m_bIsActive = false;
	}

	if (!wglMakeCurrent(m_hDC, m_hGLRC))
		PB_FPF(stderr, "PBuffer::Activate() failed.\n");

	m_bIsActive = true;
}

void PBuffer::Deactivate()
{
	if (!m_bIsActive)
		return;

	if (!wglMakeCurrent(m_hOldDC, m_hOldGLRC))
		PB_FPF(stderr, "PBuffer::Deactivate() failed.\n");

	m_hOldGLRC = 0;
	m_hOldDC = 0;
	m_bIsActive = false;
}

#elif defined(_LINUX)

PBuffer::PBuffer(const char *strMode, bool managed)
: m_pDisplay(0), m_glxPbuffer(0), m_glxContext(0), m_pOldDisplay(0), m_glxOldDrawable(0),
m_glxOldContext(0), m_iWidth(0), m_iHeight(0), m_strMode(strMode),
m_bSharedContext(false), m_bShareObjects(false), m_bManaged(managed)
{
	m_pfAttribList.push_back(GLX_DRAWABLE_TYPE);
	m_pfAttribList.push_back(GLX_PBUFFER_BIT);
	m_pfAttribList.push_back(GLX_RENDER_TYPE);
	m_pfAttribList.push_back(GLX_RGBA_BIT);

	m_pbAttribList.push_back(GLX_LARGEST_PBUFFER);
	m_pbAttribList.push_back(true);
	m_pbAttribList.push_back(GLX_PRESERVED_CONTENTS);
	m_pbAttribList.push_back(true);

	m_strMode = strMode;
	parseModeString(m_strMode, &m_pfAttribList, &m_pbAttribList);

	m_pfAttribList.push_back(0);
	m_pbAttribList.push_back(0);
}

PBuffer::~PBuffer()
{
	if (m_bManaged)
		Destroy();
}

bool PBuffer::Initialize(int iWidth, int iHeight, bool bShareContexts, bool bShareObjects)
{
	Display *pDisplay = glXGetCurrentDisplay();
	int iScreen = DefaultScreen(pDisplay);
	GLXContext glxContext = glXGetCurrentContext();

	GLXFBConfig *glxConfig;
	int iConfigCount;

	m_bSharedContext = bShareContexts;
	m_bShareObjects = bShareObjects;

	m_iWidth = iWidth;
	m_iHeight = iHeight;

	if (m_bSharedContext)
	{
		glxConfig = glXGetFBConfigs(pDisplay, iScreen, &iConfigCount);
		if (!glxConfig)
		{
			PB_FPF(stderr, "pbuffer creation error:  glXGetFBConfigs() failed\n");
			return false;
		}
	}
	else
	{
		glxConfig = glXChooseFBConfigSGIX(pDisplay, iScreen, &m_pfAttribList[0], &iConfigCount);
		if (!glxConfig)
		{
			PB_FPF(stderr, "pbuffer creation error:  glXChooseFBConfig() failed\n");
			return false;
		}
	}

	m_glxPbuffer = glXCreateGLXPbufferSGIX(pDisplay, glxConfig[0], m_iWidth, m_iHeight, &m_pbAttribList[0]);

	if (!m_glxPbuffer)
	{
		PB_FPF(stderr, "pbuffer creation error:  glXCreatePbuffer() failed\n");
		return false;
	}

	if (m_bSharedContext)
	{
		m_glxContext = glxContext;
	}
	else
	{
		if (m_bShareObjects)
			m_glxContext = glXCreateContextWithConfigSGIX(pDisplay, glxConfig[0], GLX_RGBA_TYPE, glxContext, true);
		else
			m_glxContext = glXCreateContextWithConfigSGIX(pDisplay, glxConfig[0], GLX_RGBA_TYPE, NULL, true);

		if (!glxConfig)
		{
			PB_FPF(stderr, "pbuffer creation error:  glXCreateNewContext() failed\n");
			return false;
		}
	}

	m_pDisplay = pDisplay;

	unsigned int w, h;
	w = h = 0;

	glXQueryGLXPbufferSGIX(m_pDisplay, m_glxPbuffer, GLX_WIDTH, &w);
	glXQueryGLXPbufferSGIX(m_pDisplay, m_glxPbuffer, GLX_HEIGHT, &h);
	m_iWidth = w;
	m_iHeight = h;

	PB_FPF(stdout, "Created a %d x %d pbuffer\n", m_iWidth, m_iHeight);

	return true;
}

void PBuffer::Destroy()
{
	if (m_glxContext && !m_bSharedContext)
		glXDestroyContext(m_pDisplay, m_glxContext);

	if (m_glxPbuffer)
		glXDestroyGLXPbufferSGIX(m_pDisplay, m_glxPbuffer);

	m_glxContext = 0;
	m_glxPbuffer = 0;
	m_pDisplay = 0;
}

void PBuffer::parseModeString(const char *modeString, vector<int> *pfAttribList, vector<int> *pbAttribList)
{
	if (!modeString || strcmp(modeString, "") == 0)
		return;

	m_iBitsPerComponent = 8;
	m_iNComponents = 0;
	bool bIsFloatBuffer = false;
	bool bNeedAlpha = false;

	char *mode = strdup(modeString);

	vector<string> tokens;
	char *buf = strtok(mode, " ");
	while (buf != NULL)
	{
		if (strstr(buf, "float") != NULL)
			bIsFloatBuffer = true;

		if (strstr(buf, "alpha") != NULL)
			bNeedAlpha = true;

		tokens.push_back(buf);
		buf = strtok(NULL, " ");
	}

	for (unsigned int i = 0; i < tokens.size(); i++)
	{
		string token = tokens[i];

		if (token == "rgb" && !bIsFloatBuffer)
		{
			pfAttribList->push_back(GLX_RED_SIZE);
			pfAttribList->push_back(m_iBitsPerComponent);
			pfAttribList->push_back(GLX_GREEN_SIZE);
			pfAttribList->push_back(m_iBitsPerComponent);
			pfAttribList->push_back(GLX_BLUE_SIZE);
			pfAttribList->push_back(m_iBitsPerComponent);
			m_iNComponents += 3;
			continue;
		}
		else if (token == "rgb") PB_FPF(stderr, "warning : mistake in components definition (rgb + %d)\n", m_iNComponents);

		if (token == "rgba" && (m_iNComponents == 0))
		{
			pfAttribList->push_back(GLX_RED_SIZE);
			pfAttribList->push_back(m_iBitsPerComponent);
			pfAttribList->push_back(GLX_GREEN_SIZE);
			pfAttribList->push_back(m_iBitsPerComponent);
			pfAttribList->push_back(GLX_BLUE_SIZE);
			pfAttribList->push_back(m_iBitsPerComponent);
			pfAttribList->push_back(GLX_ALPHA_SIZE);
			pfAttribList->push_back(m_iBitsPerComponent);
			m_iNComponents = 4;
			continue;
		}
		else if (token == "rgba") PB_FPF(stderr, "warning : mistake in components definition (rgba + %d)\n", m_iNComponents);

		if (token.find("alpha") != token.npos)
		{
			pfAttribList->push_back(GLX_ALPHA_SIZE);
			pfAttribList->push_back(m_iBitsPerComponent);
			m_iNComponents++;
			continue;
		}
		else if (token == "alpha") PB_FPF(stderr, "warning : mistake in components definition (alpha + %d)\n", m_iNComponents);

		if (token.find("depth") != token.npos)
		{
			pfAttribList->push_back(GLX_DEPTH_SIZE);
			pfAttribList->push_back(getIntegerValue(token));

			continue;
		}

		if (token.find("stencil") != token.npos)
		{
			pfAttribList->push_back(GLX_STENCIL_SIZE);
			pfAttribList->push_back(getIntegerValue(token));

			continue;
		}

		if (token.find("samples") != token.npos)
		{
			pfAttribList->push_back(GLX_SAMPLE_BUFFERS_ARB);
			pfAttribList->push_back(1);
			pfAttribList->push_back(GLX_SAMPLES_ARB);
			pfAttribList->push_back(getIntegerValue(token));

			continue;
		}

		if (token == "double")
		{
			pfAttribList->push_back(GLX_DOUBLEBUFFER);
			pfAttribList->push_back(true);

			continue;
		}

		if (token.find("float") == 0)
		{
			m_iBitsPerComponent = getIntegerValue(token);
			//bIsFloatBuffer = true; done previously
			pfAttribList->push_back(GLX_FLOAT_COMPONENTS_NV);
			pfAttribList->push_back(true);
			continue;
		}

		PB_FPF(stderr, "unknown pbuffer attribute: %s\n", token.c_str());
	}
}

void PBuffer::Activate(PBuffer *current /* = NULL */)
{
	if (current == this)
	{
		return; // no switch necessary
	}

	if (NULL == current || !current->m_bIsActive)
	{
		m_pOldDisplay = glXGetCurrentDisplay();
		m_glxOldDrawable = glXGetCurrentDrawable();
		m_glxOldContext = glXGetCurrentContext();
	}
	else
	{
		m_pOldDisplay = current->m_pOldDisplay;
		m_glxOldDrawable = current->m_glxOldDrawable;
		m_glxOldContext = current->m_glxOldContext;
		current->m_pOldDisplay = 0;
		current->m_glxOldDrawable = 0;
		current->m_glxOldContext = 0;
	}

	if (!glXMakeCurrent(m_pDisplay, m_glxPbuffer, m_glxContext))
		PB_FPF(stderr, "PBuffer::Activate() failed.\n");
}

void PBuffer::Deactivate()
{
	if (!glXMakeCurrent(m_pOldDisplay, m_glxOldDrawable, m_glxOldContext))
		PB_FPF(stderr, "PBuffer::Deactivate() failed.\n");

	m_pOldDisplay = 0;
	m_glxOldDrawable = 0;
	m_glxOldContext = 0;
}

int PBuffer::Bind(int iBuffer)
{
	return 0;
}


#elif defined(MACOS)

PBuffer::PBuffer(const char *strMode)
:
m_iWidth(0), m_iHeight(0), m_strMode(strMode),
m_bSharedContext(false), m_bShareObjects(false)
{
	PB_FPF(stderr, "pbuffer not implemented under Mac OS X yet\n");
}

PBuffer::~PBuffer()
{
	PB_FPF(stderr, "pbuffer not implemented under Mac OS X yet\n");
}

bool PBuffer::Initialize(int iWidth, int iHeight, bool bShareContexts, bool bShareObjects)
{
	PB_FPF(stderr, "pbuffer not implemented under Mac OS X yet\n");

	return false;
}

void PBuffer::Activate()
{
	PB_FPF(stderr, "pbuffer not implemented under Mac OS X yet\n");
}

void PBuffer::Deactivate()
{
	PB_FPF(stderr, "pbuffer not implemented under Mac OS X yet\n");
}

#endif

string PBuffer::getStringValue(std::string token)
{
	size_t pos;
	if ((pos = token.find("=")) != token.npos)
	{
		string value = token.substr(pos+1, token.length()-pos+1);
		return value;
	}
	else
		return "";
}

int PBuffer::getIntegerValue(std::string token)
{
	size_t pos;
	if ((pos = token.find("=")) != token.npos)
	{
		string value = token.substr(pos+1, token.length()-pos+1);
		if (value.empty())
			return 1;
		return atoi(value.c_str());
	}
	else
		return 1;
}

unsigned int PBuffer::GetSizeInBytes()
{
	return m_iWidth * m_iHeight * (m_iNComponents/8);
}
unsigned int PBuffer::CopyToBuffer(void *ptr, int w, int h)
{
	GLenum format=0;
	GLenum type=0;
	switch(m_iNComponents)
	{
	case 1: //
		format = GL_LUMINANCE; // is it right to ask for Red only component ?
		break;
	case 2:
		format = GL_LUMINANCE_ALPHA; //How to ask for GL_RG ??
		break;
	case 3:
		format = GL_RGB;
		break;
	case 4:
		format = GL_RGBA;
		break;
	}
	switch(m_iBitsPerComponent)
	{
	case 8:
		type = GL_UNSIGNED_BYTE;
		break;
	case 32:
		type = GL_FLOAT;
		break;
#ifdef GL_NV_half_float
	case 16:
		type = GL_HALF_FLOAT_NV;
		break;
#endif
	default:
		PB_FPF(stderr, "unknown m_iBitsPerComponent\n");
#	if defined(_WINDOWS)
		//_asm { int 3; }
#	endif
	}
	Activate();
	if((w < 0) || (w > m_iWidth))
		w = m_iWidth;
	if((h < 0) || (h > m_iHeight))
		h = m_iHeight;

	glReadPixels(0,0,w,h, format, type, ptr);
	Deactivate();
	return w * h * (m_iNComponents/8);
}

PBufferManager :: PBufferManager(const char *strMode, bool managed):active(0), unique_context(false)
{
	princ = new PBuffer(strMode, managed);
	sec   = new PBuffer(strMode, managed);
	princ->Initialize(_GPUCV_TEXTURE_MAX_SIZE_X, _GPUCV_TEXTURE_MAX_SIZE_Y, true, true);
	sec->Initialize(_GPUCV_TEXTURE_MAX_SIZE_X, _GPUCV_TEXTURE_MAX_SIZE_Y, false, false);
	//     princ->Initialize(512, 512, true, true);
	//     sec->Initialize(512, 512, false, false);
}

void PBufferManager :: ActivateFirst(GLuint tex_bind)
{
	if (!unique_context)
	{
		getGLContext()->PushAttribs();
		if (tex_bind != 0)
			princ->Bind(tex_bind);
		princ->Activate();
	}
}

void PBufferManager :: ForceFirst(GLuint tex_bind)
{
	if (!unique_context)   ActivateFirst(tex_bind);
	active = 1;
}

void PBufferManager :: ForceSecond(GLuint tex_bind)
{
	if (!unique_context)   ActivateSecond(tex_bind);
	active = 2;

}

void PBufferManager :: ActivateSecond(GLuint tex_bind)
{
	if (!unique_context)
	{
		getGLContext()->PushAttribs();
		if (tex_bind != 0)
			princ->Bind(tex_bind);
		sec->Activate();
	}
}

void PBufferManager :: Desactivate()
{

	if (!unique_context) // if unique context, dont desactivate the PBuffer
	{
		princ->Deactivate();
		sec->Deactivate();
		getGLContext()->PopAttribs();
	}

	active = 0;
}

void PBufferManager::SetUniqueContext()
{
	ActivateFirst();
	unique_context = true;
}

void PBufferManager::UnsetUniqueContext()
{
	Desactivate();
	unique_context = false;
}

int PBufferManager :: IsActive()
{ return active; }

bool PBufferManager :: IsUnique()
{ return unique_context; }

PBufferManager * PBuffersManager()
{
	static PBufferManager *PBuffSingleton;
	if (!PBuffSingleton)
		//  if(GetHardProfile()->IsFloat32Compatible())
		PBuffSingleton = new PBufferManager("float=32 rgba", false);
	/*  else//Yann.A. 18/10/05...to test ????
	PBuffSingleton = new PBufferManager("float=16 rgba", false);//float=16
	*/

	//  if (!PBuffSingleton) PBuffSingleton = new PBufferManager("rgba", false);
	return PBuffSingleton;
}

}//namespace GCV


