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



#ifndef __GPUCV_TEXTURE_FBO_H
#define __GPUCV_TEXTURE_FBO_H
#include "GPUCVTexture/config.h"
namespace GCV{

/** @addtogroup RENDERTOTEXT_GRP
*  @{
*/

#define NO_FORCED     0
#define TEX_FORCED    1
#define RENDER_FORCED 2
#define FBO_SUPPORT_DEPTH	0//??remove this
#define FBO_SUPPORT_STENCIL 0
#define FBO_MRT 1		//!< Multiple render target features

enum TYPE{
	TYPE_undefined,
	TYPE_renderBuffer,
	TYPE_texture
};

struct ST_TEX_DEST //!< store a texture id and its destination color
{

	GLuint	m_id;
	GLuint	m_colorattachment;
	TYPE	m_type;
	GLuint	m_textureType;
	GLuint	m_width;
	GLuint	m_height;	
};

/**
*	\brief manage FBO and OpenGL context
*	\author Erwan Guehenneux
*	\author Jean-Philippe Farrugia
*	\author Yannick Allusse
*  Automatize and simplify FBO use and integration in an existing OpenGL context.
*/
class _GPUCV_TEXTURE_EXPORT FBO
	: CL_Profiler
{
public :


	/**
	*	\brief  Default constructor : renderbuffer attached, GL-RGB type
	*/
	FBO();

	/*! 
	*	\brief  Second constructor : allows to chose renderbuffers or not, and type
	*	\param  type -> format of the render buffer
	*	\param  init_bool_rb -> if true, a render buffer will be create, if false, FBO will not present any renderbuffer
	*/		
	FBO(GLenum type,bool init_bool_rb);

	/*! 
	*	\brief  destructor
	*/
	~FBO();


	virtual std::ostringstream & operator << (std::ostringstream & _stream)const;
	/*! 
	*	\brief  check initialization of FBO and log any error
	*/
	void init();


	/*! 
	*	\brief  Attach texture to the FBO and sets it for rendering
	*	\param  texid -> openGL texture to attach
	*	\param  width -> image width
	*	\param  height -> image height
	*/
	void SetTexture(GLuint texid,int width, int height, GLuint _type, int colorAttachment=GL_COLOR_ATTACHMENT0_EXT);

	/*!
	*	\brief  unset in texture rendering
	*/
	void UnsetTexture();

	/*! 
	*	\brief  check the size of renderbuffer and resize it if necessary
	*	\param  width -> new renderbuffer width
	*	\param  height -> new renderbuffer height
	*/
	void ResizeRb(int width, int height);

	/*!
	*	\brief  set configuration for renderbuffer rendering
	*	\param width -> image width
	*	\param height -> image height
	*/
	void SetRender(int width, int height);


	/*!
	*	\brief  unset renderbuffer rendering
	*/
	void UnsetRender();

	/*! 
	*	\brief  is one kind of rendering forced ? (when render is force, all default rendering context changes are no more done before UnFoce function call)
	*	\return int -> kind of rendering forced (see fbo.h for type definitions)
	*/
	int  IsForced();


	/*! 
	*	\brief  force renderbuffer rendering
	*	\param  width -> image width
	*	\param  height -> image height
	*/
	void ForceRender(int width, int height);

	/*!
	*	\brief  stop forced rendering, reactivate default GL contexts comportment 
	*/
	void UnForce();

	GLuint GetFBOId(void)const {return framebuf;};

private : 

#if FBO_SUPPORT_DEPTH
	GLuint renderBuffer_Depth;//depth buffer, yannick
#endif

#if FBO_SUPPORT_STENCIL
	GLuint render_Stencil; // stencill render target
#endif

	bool bool_rb;		// true if renderbuffer, false either


	GLenum	FBOinternalFormat;
	int active;
	GLuint framebuf;
#if !FBO_MRT
	GLenum tex_attach;	// color attachment for texture rendering (0 if none)
	GLenum rb_attach;	// color attachment for renderbuffer (0 if none)	
	GLuint renderbuf;
#else
	//yannick
	ST_TEX_DEST*	m_renderBuff;
	ST_TEX_DEST*	m_firstText;
	GLuint			m_currentNbrTextAttached;
	GLint			m_maxDrawBuffer;
	ST_TEX_DEST*	m_drawBuffers;	//!< Struct that contains the texture ID and the color attachment of the draw buffers...

	void	InitColorattachment(bool init_bool_rb);
	int		AttachAllTexture();
	void	ClearAttachedTexture();
	int		DettachAllTexture();
public:	
	int		AddAttachedTexture(GLuint texid,int width, int height, GLuint _type, int colorAttachment=GL_COLOR_ATTACHMENT0_EXT);
#endif
	void	ForceTexture(GLuint texid, int width, int height, GLuint _type, int colorAttachment=GL_COLOR_ATTACHMENT0_EXT);
	int		ForceMultiTexture();


	bool CheckFramebufferStatus(char * File, long Line);
#if FBO_SUPPORT_DEPTH
	bool AddDepthFBO(GLuint depthFormat, GLuint width, GLuint height);
#endif
};


/*! 
*	\brief get one unique occurrence of one FBO
*	\return pointer to a unique FBO.
*/
_GPUCV_TEXTURE_EXPORT 
FBO* FBOManager();

/** @} */ // end of RENDERTOTEXT_GRP
}//namespace GCV
#endif
