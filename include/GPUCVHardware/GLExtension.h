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



#ifndef __GPUCV_HARDWARE_GLEXTENSION_H
#define	__GPUCV_HARDWARE_GLEXTENSION_H


#include <GPUCVHardware/config.h>
#include <GPUCVHardware/GLContext.h>
#include <GPUCVHardware/ToolsGL.h>

namespace GCV{
/**
*	\brief Class use to store locally OpenGL and GLEW extension compatibilities.
*	\todo Create a genericGlExtension class, so it can be redefined for each GPU brands.
*/
class _GPUCV_HARDWARE_EXPORT_CLASS glGPUExtension
{
public://protected:
	bool    m_fbo_compatible;					//!< FBO compatibility flag.
	bool	m_pbuffer_compatible;				//!< PBuffer compatibility flag.
	bool    m_tex_rect_compatible;				//!< Texture Rectangle compatibility flag.
	bool 	m_text_non_power_of_two_compatible; //!< Texture NON Power of Two compatibility flag.
	bool    m_float32_Compatible;				//!< Float32 compatibility flag.
	bool    m_pbo_compatible;					//!< PBO compatibility flag.
	bool	m_vbo_compatible;					//!< Vertex Buffer Object flag.
	bool	m_multipleRenderTarget;				//!< MultipleRenderTarget compatibility flag.
	GLint		m_multipleRenderTarget_max_nbr;		//!< Maximum MultipleRenderTarget number compatibility flag(Shader output).
	GLint		m_MAX_TEXTURE_UNITS_ARB;			//!< Maximum Texture unit number compatibility flag(Shader input).
	bool	m_gpu_affinity;						//!< WGL_NV_gpu_affinity flag.
	bool	m_render_depth_texture;				//!< WGL_NV_render_depth_texture.
	bool	m_geometryShader;					//!< GLEW_EXT_geometry_shader4
public:

	/**
	*	\brief Default constructor.
	*/
	glGPUExtension();
	/**
	*	\brief Default destructor.
	*/
	~glGPUExtension();

	/**
	*	\brief Read all opengl and glew extension required or used by the library.
	*/
	bool ReadExtension();

	/**
	*	\brief Print to the console all OpenGL and GLEW extensions required or used by the library.
	*/
	void PrintExtension()const;

	//access functions
	__GPUCV_INLINE
		bool	IsFBOCompatible()const{return m_fbo_compatible;};
	__GPUCV_INLINE
		bool	IsPBufferCompatible()const{return m_pbuffer_compatible;};
	__GPUCV_INLINE
		bool	IsFloat32Compatible()const{return m_float32_Compatible;};
	__GPUCV_INLINE
		void    SetFloat32Compatible(bool _val){m_float32_Compatible=_val;};
	__GPUCV_INLINE
		bool	IsPBOCompatible()const{return m_pbo_compatible;};
	__GPUCV_INLINE
		bool	IsVBOCompatible()const{return m_vbo_compatible;};
	__GPUCV_INLINE
		bool	IsTextRectCompatible()const{return m_tex_rect_compatible;};
	__GPUCV_INLINE
		bool	IsTextNOPCompatible()const{return m_text_non_power_of_two_compatible;};

	bool GetSGISTextureLOD()const {return (GLEW_SGIS_texture_lod)? true:false;}
	//==========================


	/*!
	*	\brief Control texture attachment compatibility with internal format.
	*	\return TRUE if format is known as compatible with the corresponding attachment.
	*/
	bool CheckAttachmentFormat(GLuint _InternalFormat, GLuint _Attach);

	//-----------------
	//openglARB functions
	__GPUCV_INLINE
		void glActiveTextureARB(GLenum texture);
	__GPUCV_INLINE
		void glMultiTexCoord2dARB( GLenum target, GLdouble s, GLdouble t );
	__GPUCV_INLINE
		void glClientActiveTextureARB(GLenum texture);
	//
	//====================

	friend class GenericGPU;
};
}//namespace GCV
#endif
