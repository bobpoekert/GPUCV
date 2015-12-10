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
#include "GPUCVHardware/GLExtension.h"
#include "GPUCVHardware/GlobalSettings.h"
#include <SugoiTools/cl_base_obj.h>
#include <SugoiTools/tools.h>

namespace GCV{

typedef void (* PFnVoid)();
static PFnVoid gpucvGetProcAddress(const char * function)
{
#ifdef _WINDOWS
	return (PFnVoid) wglGetProcAddress(function);
#else
	return (PFnVoid) glXGetProcAddress((const GLubyte *) function);
#endif
}

//=====================================
glGPUExtension::glGPUExtension()
{}
//=====================================
glGPUExtension::~glGPUExtension()
{}
//=====================================
bool glGPUExtension::ReadExtension()
{
	m_fbo_compatible = (GLEW_EXT_framebuffer_object)? true: false;
	m_tex_rect_compatible = (GLEW_ARB_texture_rectangle)? true: false;
	m_text_non_power_of_two_compatible = (GLEW_ARB_texture_non_power_of_two)?true:false;
	m_float32_Compatible = false;
	m_pbo_compatible = (GLEW_EXT_pixel_buffer_object )?true:false;
	m_vbo_compatible = (GL_ARB_vertex_buffer_object)?true:false;
	m_multipleRenderTarget = (GL_ARB_draw_buffers)? true:false;
	glGetIntegerv(GL_MAX_DRAW_BUFFERS, &m_multipleRenderTarget_max_nbr);
	glGetIntegerv(GL_MAX_TEXTURE_UNITS_ARB, &m_MAX_TEXTURE_UNITS_ARB);
	m_geometryShader = (GLEW_EXT_geometry_shader4)? true:false;

#ifdef _WINDOWS
	m_render_depth_texture = WGL_NV_render_depth_texture;
#endif


	m_gpu_affinity  = false;//(WGL_NV_gpu_affinity)? true: false;


#if defined(_WINDOWS)
	m_pbuffer_compatible = (WGL_ARB_pbuffer)?true:false;
#elif defined(_LINUX)
	m_pbuffer_compatible = (GLX_SGIX_pbuffer)?true:false;
#elif defined(_MACOS)
	m_pbuffer_compatible = (GL_APPLE_pixel_buffer)?true:false;
#endif

	return true;
}
//=====================================
void glGPUExtension::PrintExtension()const
{
	GPUCV_NOTICE("\nOpenGL compatible extensions : " );
	GPUCV_NOTICE("\t"<< m_pbuffer_compatible	<<"\tGLEW_ARB_pbuffer" );
	GPUCV_NOTICE("\t"<< m_fbo_compatible		<< "\tGLEW_EXT_framebuffer_object" );
	GPUCV_NOTICE("\t"<< m_pbo_compatible		<<"\tGLEW_EXT_pixel_buffer_object" );
	GPUCV_NOTICE("\t"<< m_vbo_compatible		<<"\tGL_ARB_vertex_buffer_object" );
	GPUCV_NOTICE("\t"<< m_tex_rect_compatible	<<  "\tGLEW_ARB_texture_rectangle" );
	GPUCV_NOTICE("\t"<< m_text_non_power_of_two_compatible << "\tGLEW_ARB_texture_non_power_of_two" );
	GPUCV_NOTICE("\t"<< m_render_depth_texture	<< "\tWGL_NV_render_depth_texture" );
	GPUCV_NOTICE("\t"<< GL_ARB_fragment_shader	<<"\tGL_ARB_fragment_shader" );
	GPUCV_NOTICE("\t"<< GL_ARB_vertex_shader	<< "\tGL_ARB_vertex_shader" );
	GPUCV_NOTICE("\t"<< GL_ARB_occlusion_query	<<"\tGL_ARB_occlusion_query" );

	GPUCV_NOTICE("\t"<< GL_ARB_draw_buffers		<<"\tGL_ARB_draw_buffers" );
	GPUCV_NOTICE("\t"<< m_multipleRenderTarget	<<"\tGL_ARB_draw_buffers(MultipleRenderTarget)" );
	GPUCV_NOTICE("\t"<< m_multipleRenderTarget_max_nbr <<"\tMultipleRenderTarget max target." );
	GPUCV_NOTICE("\t"<< GL_EXT_gpu_shader4		<<"\tGL_EXT_gpu_shader4" );
	GPUCV_NOTICE("\t"<< GL_EXT_geometry_shader4		<<"\tGL_EXT_geometry_shader4" );
	GPUCV_NOTICE("\t"<< m_gpu_affinity			<<"\tWGL_NV_gpu_affinity." );
	GPUCV_NOTICE("\t"<< GetSGISTextureLOD()		<<"\tGLEW_SGIS_texture_lod." );
	GPUCV_NOTICE("\t"<< GLEW_ARB_imaging		<<"\tGL_ARB_imaging." );

#ifdef _WINDOWS
	if (WGL_ATI_render_texture_rectangle)
		GPUCV_NOTICE("\t"<< WGL_ATI_render_texture_rectangle <<"\tWGL_ATI_render_texture_rectangle" );
	if (WGL_ATI_pixel_format_float)
		GPUCV_NOTICE("\t"<< WGL_ATI_pixel_format_float <<"\tWGL_ATI_pixel_format_float" );
#endif

}
//=====================================
void glGPUExtension::glActiveTextureARB(GLenum texture)
{
	static PFNGLACTIVETEXTUREARBPROC _stc_glActiveTextureARB = (PFNGLACTIVETEXTUREARBPROC) gpucvGetProcAddress("glActiveTextureARB");
	if(	_stc_glActiveTextureARB)
		_stc_glActiveTextureARB(texture);
	else
	{
		SG_ERROR_LOG("glGPUExtension::glActiveTextureARB() => no extensiong for glActiveTextureARB")
	}
}
//=====================================
void glGPUExtension::glMultiTexCoord2dARB( GLenum target, GLdouble s, GLdouble t )
{
	static PFNGLMULTITEXCOORD2DARBPROC _stc_glMultiTexCoord2dARB = (PFNGLMULTITEXCOORD2DARBPROC) gpucvGetProcAddress("glMultiTexCoord2dARB");
	if(	_stc_glMultiTexCoord2dARB)
		_stc_glMultiTexCoord2dARB(target, s, t);
	else
	{
		SG_ERROR_LOG("glGPUExtension::glMultiTexCoord2dARB() => no extensiong for glMultiTexCoord2dARB");
	}
}
//=====================================
void glGPUExtension::glClientActiveTextureARB(GLenum texture)
{
	static PFNGLCLIENTACTIVETEXTUREARBPROC _stc_glClientActiveTextureARB = (PFNGLCLIENTACTIVETEXTUREARBPROC) gpucvGetProcAddress("glClientActiveTextureARB");
	if(	_stc_glClientActiveTextureARB)
		_stc_glClientActiveTextureARB(texture);
	else
	{
		SG_ERROR_LOG("glGPUExtension::glClientActiveTextureARB() => no extensiong for glClientActiveTextureARB");
	}
}
//=====================================
bool glGPUExtension::CheckAttachmentFormat(GLuint _InternalFormat, GLuint _Attach)
{
	switch(_Attach)
	{
	case GL_DEPTH_ATTACHMENT_EXT:
		switch (_InternalFormat)
		{//http://oss.sgi.com/projects/ogl-sample/registry/ARB/depth_texture.txt
		case GL_DEPTH_COMPONENT:
		case GL_DEPTH_COMPONENT16:// same as GL_DEPTH_COMPONENT16_ARB and GL_DEPTH_COMPONENT16_SGIX
			//				case GL_DEPTH_COMPONENT16_ARB:
			//				case GL_DEPTH_COMPONENT16_SGIX:
		case GL_DEPTH_COMPONENT24:
			//				case GL_DEPTH_COMPONENT24_ARB:
			//				case GL_DEPTH_COMPONENT24_SGIX:
		case GL_DEPTH_COMPONENT32:
			//				case GL_DEPTH_COMPONENT32_ARB:
			//				case GL_DEPTH_COMPONENT32_SGIX:
		case GL_DEPTH_COMPONENT32F_NV:
			return true;break;
		default:return false;
		}break;
	default:return true;
	}
	return false;
}
//=====================================
//=====================================
}//namespace GCV
