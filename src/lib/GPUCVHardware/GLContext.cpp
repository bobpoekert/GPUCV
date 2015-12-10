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
#include "GPUCVHardware/GLContext.h"
#include "GPUCVHardware/hardware.h"
#include "GPUCVHardware/GlobalSettings.h"

namespace GCV{
//=================================================
void InitGLView(TextSize<GLsizei> & _size)
{
	InitGLView(0, _size._GetWidth(), 0, _size._GetHeight());
}
//=================================================
void InitGLView(int x_max, int y_max)
{
	InitGLView(0, x_max, 0, y_max);
}
//=================================================
void InitGLView(int x_min, int x_max, int y_min, int y_max)
{
	_GPUCV_GL_ERROR_TEST();
	glEnable(GL_SCISSOR_TEST);
	glScissor(x_min,y_min,x_max, y_max);
	glViewport(x_min, y_min, x_max, y_max);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(-1, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	//   glEnable(GetHardProfile()->GetTextType());
	//glClearDepth(0.);
	//	glClearColor(0., 0., 0., 0.);
	_GPUCV_GL_ERROR_TEST();

	//rise unknown error 1286...most of the time, this is due to wrong framebuffer format
	//if(!GetHardProfile()->GetMainGPU()->IsHardwareFamily(GenericGPU::HRD_FAM_ATI))
	{//don't know why, this crash on my ATI radeon 9700

		//  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT|GL_ACCUM_BUFFER_BIT);
		glClear(GL_COLOR_BUFFER_BIT);//| GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT|GL_ACCUM_BUFFER_BIT);
		_GPUCV_GL_ERROR_TEST();
	}
}
//=================================================
void InitGLViewPerspective(TextSize<GLsizei> & _size, float _near/*=0.01*/, float _far/*=1000.*/)
{
	InitGLViewPerspective(_size._GetWidth(), _size._GetHeight(),_near, _far);
}
void InitGLViewPerspective(int width, int height, float _near/*=0.01*/, float _far/*=1000.*/)
{
	_GPUCV_GL_ERROR_TEST();
	glEnable(GL_SCISSOR_TEST);
	glScissor(0,0,width,height);
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(30., 4/3., _near, _far);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glEnable(GetHardProfile()->GetTextType());
	glEnable(GL_DEPTH_TEST);
	glDepthRange(_near,_far);
	glDepthMask(true);
	glDepthFunc(GL_LEQUAL);
	_GPUCV_GL_ERROR_TEST();

	//glClearDepth(0.);
	//glClearColor(0., 0., 0., 0.);
	//??rise unknown error 1286...
	//if(!GetHardProfile()->GetMainGPU()->IsHardwareFamily(GenericGPU::HRD_FAM_ATI))
	{//don't know why, this crash on my ATI radeon 9700
		glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT|GL_ACCUM_BUFFER_BIT);
	}
	_GPUCV_GL_ERROR_TEST();
}
//=================================================








/*static*/
GLContext*	GLContext::m_CurrentContext = NULL;
bool		GLContext::m_MultiContext = false;
std::vector <GLContext*> GLContext::m_ContextVector;
/*static*/
void GLContext::PushCurrentGLContext()
{
	if (!m_MultiContext)
		return;

	GPUCV_DEBUG(blue << "Push Current GL context"<< white);
	LogIndentIncrease();

	glFinish();
	if(m_CurrentContext)//not the first time
		m_ContextVector.push_back(m_CurrentContext);

#ifdef _WINDOWS
	m_CurrentContext = new GLContext(wglGetCurrentDC(), wglGetCurrentContext());
	m_CurrentContext->MakeCurrent();
#endif
}
//=================================================
/*static*/
void GLContext::PopCurrentGLContext()
{
	if (!m_MultiContext)
		return;

	LogIndentDecrease();
	GPUCV_DEBUG(blue << "Pop Current GL context" << white);


	glFinish();
	if(m_ContextVector.size() ==0)
	{
		GPUCV_WARNING("GLContext::PopCurrentGLContext() => no more context to pop");
		return;
	}
	delete m_CurrentContext;
	m_CurrentContext = m_ContextVector[m_ContextVector.size() -1];
	m_ContextVector.pop_back();
	m_CurrentContext->MakeCurrent();
}
//=================================================
/*static*/
GLContext* GLContext::GetActiveContext()
{
	SG_Assert(m_ContextVector.size() >= 1, "GLContext::GetActiveContext() => No active context.");
	return m_CurrentContext;
}
//=================================================
/*static*/
void GLContext::SetMultiContext(bool _multi)
{
	m_MultiContext = _multi;
}
//=================================================
/*__GPUCV_INLINE*/
void GLContext::MakeCurrent()
{
#ifdef _WINDOWS
	//SG_Assert(m_hDC!=0 && m_hRC!=0, "GLContext::MakeCurrent() => Empty context.");
	if(m_hDC!=0 && m_hRC!=0)
		wglMakeCurrent( m_hDC, m_hRC );
	else
		GPUCV_WARNING("GLContext::MakeCurrent() => Empty context.");
#endif
}
//=================================================
GLContext::GLContext()
#ifdef _WINDOWS
:
m_hDC(0),
m_hRC(0)
#endif
{
	if(!m_CurrentContext)
		m_CurrentContext = this;
}
//=================================================
#ifdef _WINDOWS
GLContext::GLContext(HDC hDC , HGLRC hRC)
:m_hDC(0),
m_hRC(0)
{
	m_hDC = hDC;
	m_hRC = hRC;
	if(!m_CurrentContext)
		m_CurrentContext = this;
}
#endif
//=================================================
GLContext::~GLContext()
{}
//=================================================
void GLContext :: PushAttribs()
{
	//GPUCV_DEBUG("====>GLContext :: PushAttribs()")
	glPushAttrib(
		//##################################
		//when some of this flags are set, FBO does not work any more
		//	GL_ACCUM_BUFFER_BIT|
		//	GL_COLOR_BUFFER_BIT|
		//	GL_CURRENT_BIT|
		//	GL_DEPTH_BUFFER_BIT|
		//##################################
		//	to be tested, but we don't use all of them in GPUCV
		//	GL_EVAL_BIT|
		//	GL_FOG_BIT|
		//	GL_HINT_BIT|
		//	GL_LINE_BIT|
		//	GL_LIST_BIT|
		//	GL_STENCIL_BUFFER_BIT|
		//	GL_TRANSFORM_BIT|
		/*	GL_PIXEL_MODE_BIT|
		GL_POINT_BIT|
		GL_POLYGON_BIT|
		GL_POLYGON_STIPPLE_BIT|
		*/
		//##################################
		GL_ENABLE_BIT|		//ok
		GL_LIGHTING_BIT|	//ok
		GL_SCISSOR_BIT|		//OK
		GL_TEXTURE_BIT|		//OK
		GL_VIEWPORT_BIT		//OK
		);
}
//=================================================
void GLContext :: PopAttribs()
{
	//	GPUCV_DEBUG("<====GLContext :: PopAttribs()")
	glPopAttrib();
}
//=================================================
GLContext* getGLContext()
{
	static GLContext GLContextSingleton;
	return &GLContextSingleton;
}
//=================================================
bool ShowOpenGLError(char * File, long Line)
{
	if(!GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_CHECK))
		return false;

	int GLERROR = glGetError();
	if (GLERROR==GL_NO_ERROR) return false;
	Beep(5000, 20);

	std::string FullMsg = "OPENGL ERROR===";
	//FullMsg += "\nFILE  : ";
	//FullMsg += File;
	//FullMsg += "\nLINE  : ";
	//FullMsg += SGE::ToCharStr(Line);
	FullMsg += "\nCODE  : ";
	FullMsg += SGE::ToCharStr(GLERROR);

	string Msg="";
	const GLubyte *errStr;
	errStr = gluErrorString(GLERROR);
	if (errStr)
	{
		Msg = "\nMSG(OpenGL): ";
		Msg +=(char*)errStr;
	}
	else
	{
		Msg = "\nMSG(Custom):";
		switch(GLERROR)
		{
		case GL_INVALID_ENUM 		: Msg += "GLenum argument out of range.";break;
		case GL_INVALID_VALUE 		: Msg += "Numeric argument out of range.";break;
		case GL_INVALID_OPERATION 	: Msg += "Operation illegal in current state.";break;
		case GL_STACK_OVERFLOW 	: Msg += "Function would cause a stack overflow.";break;
		case GL_STACK_UNDERFLOW 	: Msg += "Function would cause a stack underflow.";break;
		case GL_OUT_OF_MEMORY 		: Msg += "Not enough memory left to execute function.";break;
		case 1286 					: Msg += "It seems that glClear() or glClearColor() raised an error, due to wrong framebuffer format.";break;
		default : Msg += "Unknown OpenGL error message.";
		}
	}
	FullMsg+= Msg;

	GLint green=0, blue=0, red=0, alpha=0, depth=0;
	glGetIntegerv(GL_GREEN_BITS,	&green);
	glGetIntegerv(GL_BLUE_BITS,		&blue);
	glGetIntegerv(GL_RED_BITS,		&red);
	glGetIntegerv(GL_ALPHA_BITS,	&alpha);
	glGetIntegerv(GL_DEPTH_BITS,	&depth);

	if(green*blue*red == 0)
		FullMsg+="\nFrameBuffer bits (Red, Green, Blue) are empty => May be using incompatible framebuffer format.";
	if(alpha == 0)
		FullMsg+="\nFrameBuffer bits (Alpha) is empty => May be using incompatible framebuffer format.";
	if(depth == 0)
		FullMsg+="\nFrameBuffer bits (Depth) is empty => May be using incompatible framebuffer format.";

	//FullMsg+="\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";

	if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_RISE_EXCEPTION))
	{
		GPUCV_ERROR(FullMsg);
		//SG_Assert(0, FullMsg);
		//manual assert:
		throw ::SGE::CAssertException	(File, Line, FullMsg);
	}
	else
	{	GPUCV_ERROR(FullMsg);}
	return true;
}
//=================================================
std::string GetStrGLTextureFormat(const GLuint format)
{
	string FormatStr;
	switch (format)
	{
	case GL_RED: 		FormatStr="GL_RED";		break;
	case GL_BLUE: 		FormatStr="GL_BLUE";		break;
	case GL_GREEN: 	FormatStr="GL_GREEN";		break;
	case GL_LUMINANCE: FormatStr="GL_LUMINANCE";break;
	case GL_LUMINANCE_ALPHA: FormatStr="GL_LUMINANCE_ALPHA";break;
	case GL_BGR: 		FormatStr="GL_BGR";		break;
	case GL_RGB: 		FormatStr="GL_RGB";		break;
	case GL_BGRA: 		FormatStr="GL_BGRA";	break;
	case GL_RGBA: 		FormatStr="GL_RGBA";	break;
	case GL_RGBA_FLOAT16_ATI: FormatStr="GL_RGBA_FLOAT16_ATI";	break;
	case GL_RGBA_FLOAT32_ATI: FormatStr="GL_RGBA_FLOAT32_ATI";	break;
	default: FormatStr = "UNKOWN"; break;
	}
	return FormatStr;
}
//=================================================
std::string GetStrGLInternalTextureFormat(const GLuint format)
{
	string FormatStr;
	switch (format)
	{
		//alpha
	case GL_ALPHA4: 		FormatStr="GL_ALPHA4";		break;
	case GL_ALPHA8: 		FormatStr="GL_ALPHA8";		break;
	case GL_ALPHA8UI_EXT: 	FormatStr="GL_ALPHA8UI_EXT";	break;
	case GL_ALPHA8I_EXT: 	FormatStr="GL_ALPHA8I_EXT";	break;
	case GL_ALPHA12: 		FormatStr="GL_ALPHA12";		break;
	case GL_ALPHA16: 		FormatStr="GL_ALPHA16";		break;
	case GL_ALPHA16UI_EXT: 	FormatStr="GL_ALPHA16UI_EXT";		break;
	case GL_ALPHA16I_EXT: 	FormatStr="GL_ALPHA16I_EXT";		break;
	case GL_ALPHA16F_ARB: 	FormatStr="GL_ALPHA16F_ARB";		break;
	case GL_ALPHA32F_ARB: 	FormatStr="GL_ALPHA32F_ARB";		break;

		//luminance
	case GL_LUMINANCE4:		FormatStr="GL_LUMINANCE4";break;
	case GL_LUMINANCE8:		FormatStr="GL_LUMINANCE8";break;
	case GL_LUMINANCE8UI_EXT:		FormatStr="GL_LUMINANCE8UI_EXT";break;
	case GL_LUMINANCE8I_EXT:		FormatStr="GL_LUMINANCE8I_EXT";break;
	case GL_LUMINANCE12:	FormatStr="GL_LUMINANCE12";break;
	case GL_LUMINANCE16:	FormatStr="GL_LUMINANCE16";break;
	case GL_LUMINANCE16UI_EXT:	FormatStr="GL_LUMINANCE16UI_EXT";break;
	case GL_LUMINANCE16I_EXT:	FormatStr="GL_LUMINANCE16I_EXT";break;
	case GL_LUMINANCE32UI_EXT:	FormatStr="GL_LUMINANCE32UI_EXT";break;
	case GL_LUMINANCE32I_EXT:	FormatStr="GL_LUMINANCE32I_EXT";break;
	case GL_LUMINANCE16F_ARB:	FormatStr="GL_LUMINANCE16F_ARB";break;
	case GL_LUMINANCE32F_ARB:	FormatStr="GL_LUMINANCE32F_ARB";break;


		/*add GL_LUMINANCE16_ALPHA16*/
	case GL_LUMINANCE8_ALPHA8_EXT:		FormatStr="GL_LUMINANCE8_ALPHA8_EXT";break;
	case GL_LUMINANCE16_ALPHA16:		FormatStr="GL_LUMINANCE16_ALPHA16";break;
	case GL_LUMINANCE_ALPHA16F_ARB:		FormatStr="GL_LUMINANCE_ALPHA16F_ARB";break;
		//GL_LUMINANCE_ALPHA_FLOAT16_APPLE
		// &&
		//GL_LUMINANCE_ALPHA_FLOAT16_ATI
		// are equal to GL_LUMINANCE_ALPHA16I_EXT
	case GL_LUMINANCE_ALPHA16I_EXT:		FormatStr="GL_LUMINANCE_ALPHA16I_ARB";break;
	case GL_LUMINANCE_ALPHA16UI_EXT:		FormatStr="GL_LUMINANCE_ALPHA16UI_ARB";break;
		//GL_LUMINANCE_ALPHA_FLOAT32_APPLE
		// &&
		//GL_LUMINANCE_ALPHA_FLOAT32_ATI
		// are equal to GL_LUMINANCE_ALPHA32F_ARB
	case GL_LUMINANCE_ALPHA32F_ARB:				FormatStr="GL_LUMINANCE_ALPHA32F_ARB";break;
	case GL_LUMINANCE_ALPHA32I_EXT:				FormatStr="GL_LUMINANCE_ALPHA32I_EXT";break;
	case GL_LUMINANCE_ALPHA32UI_EXT:			FormatStr="GL_LUMINANCE_ALPHA32UI_EXT";break;
	case GL_INTENSITY4:		FormatStr="GL_INTENSITY4";break;
	case GL_INTENSITY8:		FormatStr="GL_INTENSITY8";break;
	case GL_INTENSITY12:	FormatStr="GL_INTENSITY12";break;
	case GL_INTENSITY16:	FormatStr="GL_INTENSITY16";break;



		//rgb
	case GL_R3_G3_B2:		FormatStr="GL_R3_G3_B2";break;
	case GL_RGB4:			FormatStr="GL_RGB4";break;
	case GL_RGB5:			FormatStr="GL_RGB5";break;
	case GL_RGB8:			FormatStr="GL_RGB8";break;
	case GL_RGB8UI_EXT:		FormatStr="GL_RGB8UI_EXT";break;
	case GL_RGB8I_EXT:		FormatStr="GL_RGB8I_EXT";break;
	case GL_RGB12:			FormatStr="GL_RGB12";break;
	case GL_RGB16:/*GL_RGB16_EXT*/			FormatStr="GL_RGB16";break;
	case GL_RGB16UI_EXT:		FormatStr="GL_RGB16UI_EXT";break;
	case GL_RGB16I_EXT:			FormatStr="GL_RGB16I_EXT";break;
	case GL_RGB32UI_EXT:		FormatStr="GL_RGB32UI_EXT";break;
	case GL_RGB32I_EXT:			FormatStr="GL_RGB32I_EXT";break;
	case GL_RGB16F_ARB:			FormatStr="GL_RGB16F_ARB";break;
	case GL_RGB32F_ARB:			FormatStr="GL_RGB32F_ARB";break;

		//rgba
	case GL_RGBA2:			FormatStr="GL_RGBA2";break;
	case GL_RGBA4:			FormatStr="GL_RGBA4";break;
	case GL_RGB5_A1:		FormatStr="GL_RGB5_A1";break;
	case GL_RGBA8:			FormatStr="GL_RGBA8";break;
	case GL_RGBA8UI_EXT:	FormatStr="GL_RGBA8UI_EXT";break;
	case GL_RGBA8I_EXT:		FormatStr="GL_RGBA8I_EXT";break;
	case GL_RGB10_A2:		FormatStr="GL_RGB10_A2";break;
	case GL_RGBA12:			FormatStr="GL_RGBA12";break;
	case GL_RGBA16:/*GL_RGBA16_EXT*/			FormatStr="GL_RGBA16";break;
	case GL_RGBA16UI_EXT:		FormatStr="GL_RGBA16UI_EXT";break;
	case GL_RGBA16I_EXT:		FormatStr="GL_RGBA16I_EXT";break;
	case GL_RGBA32UI_EXT:		FormatStr="GL_RGBA32UI_EXT";break;
	case GL_RGBA32I_EXT:		FormatStr="GL_RGBA32I_EXT";break;
	case GL_RGBA16F_ARB:/*GL_RGBA_FLOAT16_ATI*/		FormatStr="GL_RGBA16F_ARB";break;
	case GL_RGBA32F_ARB:/*GL_RGBA_FLOAT32_ATI*/			FormatStr="GL_RGBA32F_ARB";break;

		//depth
	case GL_DEPTH_COMPONENT:	FormatStr="GL_DEPTH_COMPONENT";break;
	case GL_DEPTH_COMPONENT16:	FormatStr="GL_DEPTH_COMPONENT16";break;
	case GL_DEPTH_COMPONENT24:	FormatStr="GL_DEPTH_COMPONENT24";break;
	case GL_DEPTH_COMPONENT32:	FormatStr="GL_DEPTH_COMPONENT32";break;
	case GL_DEPTH_COMPONENT32F_NV :	FormatStr="GL_DEPTH_COMPONENT32F_NV";break;

	default: FormatStr = "UNKOWN"; break;
	}
	return FormatStr;
}
//=================================================
std::string GetStrGLTexturePixelType(const GLuint format)
{
	string FormatStr;
	switch (format)
	{
	case GL_UNSIGNED_BYTE: 	FormatStr="GL_UNSIGNED_BYTE";		break;
	case GL_BYTE: 				FormatStr="GL_BYTE";		break;
	case GL_BITMAP: 			FormatStr="GL_BITMAP";		break;
	case GL_UNSIGNED_SHORT:	FormatStr="GL_UNSIGNED_SHORT";break;
	case GL_SHORT: 			FormatStr="GL_SHORT";		break;
	case GL_UNSIGNED_INT: 		FormatStr="GL_UNSIGNED_INT";		break;
	case GL_INT: 				FormatStr="GL_INT";	break;
	case GL_FLOAT: 			FormatStr="GL_FLOAT";	break;
	case GL_DOUBLE: 			FormatStr="GL_DOUBLE";	break;
	default: FormatStr = "UNKOWN"; break;
	}
	return FormatStr;
}
//=================================================
std::string GetStrGLTextureType(const GLuint type)
{
	string FormatStr;
	switch (type)
	{
	case GL_TEXTURE_1D: 			FormatStr="GL_TEXTURE_1D";		break;
	case GL_TEXTURE_2D: 			FormatStr="GL_TEXTURE_2D";		break;
	case GL_TEXTURE_2D_ARRAY_EXT: 	FormatStr="GL_TEXTURE_2D_ARRAY_EXT";break;
	case GL_TEXTURE_3D: 			FormatStr="GL_TEXTURE_3D";		break;
	default:						FormatStr="UNKOWN";				break;
	}
	return FormatStr;
}
//=================================================
GLenum GetGLNbrComponent(const GLenum format)
{
	char Res = 0;
	switch(format)
	{
	case GL_GREEN:
	case GL_BLUE:
	case GL_RED:
	case GL_LUMINANCE: Res = 1; break;
	case GL_LUMINANCE_ALPHA: Res = 2;break;
	case GL_RGB :
	case GL_BGR : 		Res = 3; break;
	case GL_RGBA :
	case GL_BGRA : 	Res = 4; break;

	default : GPUCV_ERROR("Critical : GetGLNbrComponent()=> Unknown texture format...Please update function GetGLNbrComponent().");
		Res = 4;
		break;
	}
	return Res;
}
//=================================================
/**
\todo fomat GL_BITMAP not managed.
\todo fomat GL_RGBAFLOAT not managed.
*/
GLenum GetGLTypeSize(GLenum type)
{
	char Res = 0;
	switch (type)
	{
	case GL_UNSIGNED_BYTE:
	case GL_BYTE: 				Res = sizeof(char);		break;
		//		 case GL_BITMAP: 			FormatStr="GL_BITMAP";		break;
	case GL_UNSIGNED_SHORT:
	case GL_SHORT: 			Res = sizeof(short int);break;
	case GL_UNSIGNED_INT:
	case GL_INT: 				Res = sizeof(int);break;
	case GL_FLOAT: 			Res = sizeof(float);break;
	case GL_DOUBLE: 			Res = sizeof(double);break;
		//case GL_RGBAFLOAT: 		Res = sizeof(float);break;
	default:
		GPUCV_ERROR("Critical : GetGLPixelSize()=> Unknown texture format...Using GL_BYTE");
		Res = sizeof(char);
		break;
	}
	return Res;
}
//=================================================
std::string GetStrGLColorAttachment(GLuint format)
{
	string FormatStr;
	switch (format)
	{
	case 0:	 						FormatStr="NO Attachement";		break;
	case GL_COLOR_ATTACHMENT0_EXT: 	FormatStr="GL_COLOR_ATTACHMENT0_EXT";		break;
	case GL_COLOR_ATTACHMENT1_EXT: 	FormatStr="GL_COLOR_ATTACHMENT1_EXT";		break;
	case GL_COLOR_ATTACHMENT2_EXT: 	FormatStr="GL_COLOR_ATTACHMENT2_EXT";		break;
	case GL_COLOR_ATTACHMENT3_EXT: 	FormatStr="GL_COLOR_ATTACHMENT3_EXT";		break;
	case GL_COLOR_ATTACHMENT4_EXT: 	FormatStr="GL_COLOR_ATTACHMENT4_EXT";		break;
	case GL_COLOR_ATTACHMENT5_EXT: 	FormatStr="GL_COLOR_ATTACHMENT5_EXT";		break;
	case GL_COLOR_ATTACHMENT6_EXT: 	FormatStr="GL_COLOR_ATTACHMENT6_EXT";		break;
	case GL_COLOR_ATTACHMENT7_EXT: 	FormatStr="GL_COLOR_ATTACHMENT7_EXT";		break;
	case GL_COLOR_ATTACHMENT8_EXT: 	FormatStr="GL_COLOR_ATTACHMENT8_EXT";		break;
	case GL_DEPTH_ATTACHMENT_EXT: 		FormatStr="GL_DEPTH_ATTACHMENT_EXT";		break;
	case GL_STENCIL_ATTACHMENT_EXT: 	FormatStr="GL_STENCIL_ATTACHMENT_EXT";		break;
	default: FormatStr = "UNKOWN"; break;
	}
	return FormatStr;
}
//=================================================
GLuint GetGeometryTypeFromStr(const std::string & _typeStr)
{
	GLuint GeoType=GL_POINTS;
	if(_typeStr=="GL_POINTS")			GeoType=GL_POINTS;
	else if(_typeStr=="GL_LINES")		GeoType=GL_LINES;
	else if(_typeStr=="GL_LINE_LOOP")	GeoType=GL_LINE_LOOP;
	else if(_typeStr=="GL_LINE_STRIP")	GeoType=GL_LINE_STRIP;
	else if(_typeStr=="GL_TRIANGLES")	GeoType=GL_TRIANGLES;
	else if(_typeStr=="GL_TRIANGLE_STRIP")GeoType=GL_TRIANGLE_STRIP;
	else if(_typeStr=="GL_TRIANGLE_FAN")GeoType=GL_TRIANGLE_FAN;
	else if(_typeStr=="GL_QUADS")		GeoType=GL_QUADS;
	else if(_typeStr=="GL_QUAD_STRIP")	GeoType=GL_QUAD_STRIP;
	else if(_typeStr=="GL_POLYGON")		GeoType=GL_POLYGON;
	else
	{
		SG_Assert(0, "GetGeometryTypeFromStr(): unknown geometry type");
	}
	return GeoType;
}
//=================================================
const char * GetGeometryStrFromType(const GLuint & _type)
{
	std::string Str="";

	switch (_type)
	{
	case GL_POINTS:	 		Str="GL_POINTS";		break;
	case GL_LINES: 			Str="GL_LINES";			break;
	case GL_LINE_LOOP: 		Str="GL_LINE_LOOP";		break;
	case GL_LINE_STRIP: 	Str="GL_LINE_STRIP";	break;
	case GL_LINES_ADJACENCY_EXT: 	Str="GL_LINES_ADJACENCY_EXT";	break;
	case GL_TRIANGLES: 		Str="GL_TRIANGLES";		break;
	case GL_TRIANGLE_STRIP:	Str="GL_TRIANGLE_STRIP";break;
	case GL_TRIANGLE_FAN: 	Str="GL_TRIANGLE_FAN";	break;
	case GL_TRIANGLES_ADJACENCY_EXT: 	Str="GL_TRIANGLES_ADJACENCY_EXT";	break;
	case GL_QUADS: 			Str="GL_QUADS";			break;
	case GL_QUAD_STRIP: 	Str="GL_QUAD_STRIP";	break;
	case GL_POLYGON: 		Str="GL_POLYGON";		break;
	default:				Str="UNKOWN";			break;
	}
	return Str.data();
}
//=================================================
}//namespace GCV
