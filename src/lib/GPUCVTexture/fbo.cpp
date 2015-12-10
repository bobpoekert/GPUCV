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
#include "GPUCVTexture/fbo.h"

namespace GCV{

#if _GPUCV_DEBUG_FBO
	#define	_GPUCV_FBO_ERROR_CHECK()CheckFramebufferStatus((char*)__FILE__, __LINE__);
#else
	#define	_GPUCV_FBO_ERROR_CHECK()
#endif
//====================================================
FBO :: FBO()
: 
CL_Profiler("FBO")
,FBOinternalFormat(_GPUCV_FRAMEBUFFER_DFLT_FORMAT)
,framebuf(0)
,m_currentNbrTextAttached (0)
,m_maxDrawBuffer (0)
,m_drawBuffers(NULL)
{
	_GPUCV_FBO_ERROR_CHECK();

#if !FBO_MRT
	bool_rb = true;
	rb_attach = GL_COLOR_ATTACHMENT0_EXT;
	tex_attach = GL_COLOR_ATTACHMENT1_EXT;

	glGenFramebuffersEXT(1, &framebuf);
	glGenRenderbuffersEXT(1, &renderbuf);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuf);

	// Initializing Renderbuffer
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderbuf);
#if 1 //def _USE_IMG_FORMAT 
	//Yann.A. 15/11/05 : added texture format and pixel type to cvgimage...		
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, 
		FBOinternalFormat, _GPUCV_TEXTURE_MAX_SIZE_X, _GPUCV_TEXTURE_MAX_SIZE_Y);
#else
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, 
		//_GPUCV_FRAMEBUFFER_DFLT_FORMAT/*GL_RGB*/,_GPUCV_TEXTURE_MAX_SIZE_X,_GPUCV_TEXTURE_MAX_SIZE_Y);
		GL_RGB,_GPUCV_TEXTURE_MAX_SIZE_X,_GPUCV_TEXTURE_MAX_SIZE_Y);
#endif

	// Attaching renderbuffer to fbo
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, 
		rb_attach, GL_RENDERBUFFER_EXT, 
		renderbuf);

#else
	glGenFramebuffersEXT(1, &framebuf);
	InitColorattachment(true);
#endif
#if FBO_SUPPORT_DEPTH
	AddDepthFBO(GL_DEPTH_COMPONENT16, _GPUCV_TEXTURE_MAX_SIZE_X,_GPUCV_TEXTURE_MAX_SIZE_Y);
#endif

	_GPUCV_FBO_ERROR_CHECK();
}

//====================================================
FBO :: FBO(GLenum type,bool init_bool_rb)
: 
CL_Profiler("FBO")
,FBOinternalFormat(type)
,m_currentNbrTextAttached (0)
,m_maxDrawBuffer (0)
,m_drawBuffers(NULL)
{
	_GPUCV_FBO_ERROR_CHECK();
	glPushAttrib(GL_DEPTH_TEST);
	glDisable(GL_DEPTH_TEST);
	bool_rb = init_bool_rb;
	glGenFramebuffersEXT(1, &framebuf);

#if FBO_MRT
	InitColorattachment(bool_rb);
#else
	if(bool_rb)
	{
		// Initializing Renderbuffer
		glGenRenderbuffersEXT(1, &renderbuf);
		rb_attach = GL_COLOR_ATTACHMENT0_EXT;
		tex_attach = GL_COLOR_ATTACHMENT1_EXT;

		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuf);
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderbuf);
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, 
			FBOinternalFormat,_GPUCV_TEXTURE_MAX_SIZE_X,_GPUCV_TEXTURE_MAX_SIZE_Y);

		// Attaching renderbuffer to fbo
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, 
			rb_attach, GL_RENDERBUFFER_EXT, 
			renderbuf);
	}
	else
	{
		rb_attach = 0;
		tex_attach = GL_COLOR_ATTACHMENT0_EXT;
		// glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuf);
	}
#endif

#if FBO_SUPPORT_DEPTH
	AddDepthFBO(GL_DEPTH_COMPONENT16, _GPUCV_TEXTURE_MAX_SIZE_X,_GPUCV_TEXTURE_MAX_SIZE_Y);
#endif
	glPopAttrib();
	_GPUCV_FBO_ERROR_CHECK();
}
//====================================================
#if FBO_MRT
void FBO ::InitColorattachment(bool init_bool_rb)
{
	bool_rb = init_bool_rb;
	//get number of possible attachment
	glGetIntegerv(GL_MAX_DRAW_BUFFERS_ATI, &m_maxDrawBuffer);
	if (m_maxDrawBuffer > 0)
	{	
		m_drawBuffers = new ST_TEX_DEST[m_maxDrawBuffer];
		ClearAttachedTexture();
	}
	else
	{
		GPUCV_ERROR("ERROR : no GL_MAX_DRAW_BUFFERS_ATI extension\n");
		return;
	}

	if(bool_rb)
	{
		//generate renderBuffer
		m_renderBuff = &m_drawBuffers[0];
		m_renderBuff->m_type = TYPE_renderBuffer;
		glGenRenderbuffersEXT(1, &m_renderBuff->m_id);

		// Initializing Renderbuffer
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuf);
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, m_renderBuff->m_id);
#if 1//def _USE_IMG_FORMAT 
		//Yann.A. 15/11/05 : added texture format and pixel type to cvgimage...		
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, 
			FBOinternalFormat, _GPUCV_TEXTURE_MAX_SIZE_X, _GPUCV_TEXTURE_MAX_SIZE_Y);
#else
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, 
			//_GPUCV_FRAMEBUFFER_DFLT_FORMAT/*GL_RGB*/,_GPUCV_TEXTURE_MAX_SIZE_X,_GPUCV_TEXTURE_MAX_SIZE_Y);
			GL_RGB,_GPUCV_TEXTURE_MAX_SIZE_X,_GPUCV_TEXTURE_MAX_SIZE_Y);
#endif

		// Attaching renderbuffer to fbo
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, 
			m_renderBuff->m_colorattachment, GL_RENDERBUFFER_EXT, 
			m_renderBuff->m_id);

	}
	else
		m_renderBuff=NULL;
}
#endif//FBO_MRT

//====================================================
void FBO ::ClearAttachedTexture()
{
	//glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuf);
	for (int i=0; i < m_maxDrawBuffer; i++)
	{
		if(bool_rb && i==0)
			continue;
		/*
		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
		m_drawBuffers[i].m_colorattachment, 
		GetHardProfile()->GetTextType(),
		0, 0);
		*/
		m_drawBuffers[i].m_id	= 0;
		m_drawBuffers[i].m_colorattachment = GL_COLOR_ATTACHMENT0_EXT + i;
		m_drawBuffers[i].m_type		= TYPE_undefined;
		m_drawBuffers[i].m_textureType	= 0;
		m_drawBuffers[i].m_width	= 0;
		m_drawBuffers[i].m_height	= 0;


	}

	m_firstText = NULL;
	m_currentNbrTextAttached = (bool_rb)? 1:0;
}
//====================================================
int FBO :: AddAttachedTexture(GLuint texid,int width, int height, GLuint _type, int colorAttachment)
{
	if ((int)m_currentNbrTextAttached == m_maxDrawBuffer)
		return 0;

	m_drawBuffers[m_currentNbrTextAttached].m_id = texid;
	if(colorAttachment != GL_COLOR_ATTACHMENT0_EXT && colorAttachment!=0)		
		//
		m_drawBuffers[m_currentNbrTextAttached].m_colorattachment = colorAttachment;
	else
		//review the way we affect ID..???
		m_drawBuffers[m_currentNbrTextAttached].m_colorattachment 
		= GL_COLOR_ATTACHMENT0_EXT+m_currentNbrTextAttached;

	m_drawBuffers[m_currentNbrTextAttached].m_type = TYPE_texture;
	m_drawBuffers[m_currentNbrTextAttached].m_textureType = _type;
	m_drawBuffers[m_currentNbrTextAttached].m_width	 = width;
	m_drawBuffers[m_currentNbrTextAttached].m_height = height;

	if (m_firstText==NULL)
		m_firstText = &m_drawBuffers[m_currentNbrTextAttached];

	m_currentNbrTextAttached++;
	return m_drawBuffers[m_currentNbrTextAttached-1].m_colorattachment;
}
//====================================================
int FBO :: AttachAllTexture()
{
	//std::cout <<std::endl<< "FBO::AttachAllTexture:"<<std::endl;

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuf);
	GLenum BuffersAttachedID[8];
	for (GLuint i=0; i < m_currentNbrTextAttached; i++)
	{
		BuffersAttachedID[i] = 0;
		if(bool_rb && i==0)
			continue;

		/*std::cout << "Attaching Tex(" << m_drawBuffers[i].m_id 
		<< ") with Attach("<< GetStrGLColorAttachment(m_drawBuffers[i].m_colorattachment)
		<< ") to framebuffer("<<framebuf<<")" << std::endl;
		*/
		if (m_drawBuffers[i].m_colorattachment == GL_DEPTH_ATTACHMENT_EXT && !glIsEnabled(GL_DEPTH_TEST))
		{
			GPUCV_WARNING("FBO with Dpeth attachment, but GL_DEPTH_TEST is disable");
		}

		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
			m_drawBuffers[i].m_colorattachment, 
			m_drawBuffers[i].m_textureType,
			m_drawBuffers[i].m_id, 0);
		//glDrawBuffer(m_firstText->m_colorattachment);
		//glReadBuffer(m_firstText->m_colorattachment);
		BuffersAttachedID[i] = m_drawBuffers[i].m_colorattachment;
	}

	//enable draw buffers:
	if(m_currentNbrTextAttached>1)
		glDrawBuffers(m_currentNbrTextAttached, BuffersAttachedID);

	//delete [] BuffersAttachedID;
	//?? do we have to do it with several attachment??
	//	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuf);

	//glDrawBuffer(m_firstText->m_colorattachment);
	//glReadBuffer(m_firstText->m_colorattachment);

	return m_currentNbrTextAttached;
}
//====================================================
int FBO :: DettachAllTexture()
{
	if(!framebuf)
		return 0;

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuf);
	for (int i=0; i < m_maxDrawBuffer; i++)
	{
		if(bool_rb && i==0)
			continue;

		glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT,
			m_drawBuffers[i].m_colorattachment, 
			m_drawBuffers[i].m_textureType,
			0, 0);

		glDrawBuffer(GL_BACK);
		glReadBuffer(GL_BACK);

	}
	return m_currentNbrTextAttached;
}
//====================================================
FBO :: ~FBO()
{
	if(bool_rb)
		glDeleteRenderbuffersEXT(1, &m_renderBuff->m_id);
	//CRITICAL??? glDeleteFramebuffersEXT cause segmentation fault..!!
	//	if(framebuf>=1)
	//		glDeleteFramebuffersEXT(1, &framebuf);
}
//===============================================
void FBO :: init()
{
	//_GPUCV_FBO_ERROR_CHECK(); not needed here
	bool fbo_return = CheckFramebufferStatus((char*)__FILE__, __LINE__);

	if (fbo_return)
		cout << "FrameBufferObject init : Ok" << endl;
	_GPUCV_FBO_ERROR_CHECK();
}
//===============================================
void FBO :: SetTexture(GLuint texid,int width, int height, GLuint _type, int colorAttachment)
{
	_GPUCV_FBO_ERROR_CHECK();
	// Attaching texture
	// Checking rb and resizing if necessary
	if(bool_rb)
		ResizeRb(width,height);

	AddAttachedTexture(texid,width, height, _type, colorAttachment);

	AttachAllTexture();


	/*	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuf);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, 
	GL_COLOR_ATTACHMENT0_EXT, 
	GetHardProfile()->GetTextType(), 
	texid, 0);

	// Setting buffer for reading and drawing
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuf);
	glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
	glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
	//glPopAttrib();
	*/
	/*	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuf);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, m_firstText->m_colorattachment, 
	GetHardProfile()->GetTextType(), m_firstText->m_id, 0);

	// Setting buffer for reading and drawing
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuf);
	glDrawBuffer(m_firstText->m_colorattachment);
	glReadBuffer(m_firstText->m_colorattachment);
	*/
	//_GPUCV_FBO_ERROR_CHECK();
}
//===============================================
void FBO :: ResizeRb(int width, int height)
{
	_GPUCV_FBO_ERROR_CHECK();
	int temp_width, temp_height;
	// Getting renderbuffer dimensions
	glGetRenderbufferParameterivEXT(GL_RENDERBUFFER_EXT,GL_RENDERBUFFER_WIDTH_EXT,&temp_width);
	glGetRenderbufferParameterivEXT(GL_RENDERBUFFER_EXT,GL_RENDERBUFFER_HEIGHT_EXT,&temp_height);

	// If different from width and height, reaffecting
	if((temp_width != width)||(temp_height != height))
	{
		// Defining new renderbuffer
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, m_renderBuff->m_id);
#if 1//ef _USE_IMG_FORMAT 
		//Yann.A. 15/11/05 : added texture format and pixel type to cvgimage...		
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, 
			FBOinternalFormat,width,height);
#else
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, 
			//_GPUCV_FRAMEBUFFER_DFLT_FORMAT/*GL_RGB*/,width,height);
			GL_RGB,width,height);
#endif

		// Attaching renderbuffer to fbo
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, 
			m_renderBuff->m_colorattachment, GL_RENDERBUFFER_EXT, 
			m_renderBuff->m_id);
	}
	_GPUCV_FBO_ERROR_CHECK();
}
//===============================================
void FBO :: SetRender(int width, int height)
{
	_GPUCV_FBO_ERROR_CHECK();
	if(bool_rb)
	{
		ResizeRb(width,height);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, framebuf);
		glDrawBuffer(m_renderBuff->m_colorattachment);
		glReadBuffer(m_renderBuff->m_colorattachment);
	}
	else
		GPUCV_WARNING("Warning : cannot use fbo->SetRender without renderBuffer\n");
	_GPUCV_FBO_ERROR_CHECK();
}
//===============================================
void FBO :: UnsetRender()
{
	_GPUCV_FBO_ERROR_CHECK();
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	// Resetting buffers
	glDrawBuffer(GL_BACK);
	glReadBuffer(GL_BACK);
}
//===============================================
void FBO :: UnsetTexture()
{
	_GPUCV_FBO_ERROR_CHECK();
	ClearAttachedTexture();
	UnsetRender();
	_GPUCV_FBO_ERROR_CHECK();
}
//===============================================
int  FBO :: IsForced()
{
	return active;
}
//===============================================
void FBO :: ForceTexture(GLuint texid, int width, int height, GLuint _type, int colorAttachment)
{
	_GPUCV_FBO_ERROR_CHECK();
	SetTexture(texid, width, height, _type, colorAttachment);
	//AttachAllTexture();, works only when put directly into SetTexture() ???
	active = TEX_FORCED;
	_GPUCV_FBO_ERROR_CHECK();
}
//===============================================
int FBO :: ForceMultiTexture()
{
	_GPUCV_FBO_ERROR_CHECK();
	// Attaching texture

	// Checking rb and resizing if necessary
	if(bool_rb)
		ResizeRb(m_firstText->m_width,m_firstText->m_height);

	AttachAllTexture();

	active = TEX_FORCED;
	_GPUCV_FBO_ERROR_CHECK();
	return m_currentNbrTextAttached;
}
//===============================================
void FBO :: ForceRender(int width, int height)
{
	_GPUCV_FBO_ERROR_CHECK();
	SetRender(width, height);
	active = RENDER_FORCED;
	_GPUCV_FBO_ERROR_CHECK();
}
//===============================================
void FBO :: UnForce()
{
	_GPUCV_FBO_ERROR_CHECK();
	if (active == TEX_FORCED) UnsetTexture();
	if (active == RENDER_FORCED) UnsetRender();

	active = NO_FORCED;
	_GPUCV_FBO_ERROR_CHECK();
}
//===============================================
#if FBO_SUPPORT_DEPTH
//to be tested
bool FBO :: AddDepthFBO(GLuint depthFormat, GLuint width, GLuint height)
{//format should be GL_DEPTH_COMPONENT16 or 24
	_GPUCV_FBO_ERROR_CHECK();
	// Create depth renderbuffer
	glGenRenderbuffersEXT(1,&renderdepthbuf);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderBuffer_Depth);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, depthFormat, width,height);

	//To attach it to the currently bound FBO you call:
	//glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, renderdepthbuf);

	glBindRenderbufferEXT(GL_FRAMEBUFFER_EXT, 0)
		_GPUCV_FBO_ERROR_CHECK();
	return true;
}
#endif
//===============================================
#if FBO_SUPPORT_STENCIL
//to be tested
bool FBO :: AddStencilFBO(GLuint depthFormat, GLuint width, GLuint height)
{//format should be GL_DEPTH_COMPONENT16
	_GPUCV_FBO_ERROR_CHECK();
	// Create depth renderbuffer
	glGenRenderbuffersEXT(1,&render_Stencil);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, renderdepthbuf);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, depthFormat, width,height);

	//To attach it to the currently bound FBO you call:
	//glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, renderdepthbuf);

	_GPUCV_FBO_ERROR_CHECK();
	return true;
}
#endif
//===============================================
//from http://www.mathematik.uni-dortmund.de/~goeddeke/gpgpu/tutorial.html#error1
bool FBO ::CheckFramebufferStatus(char * File, long Line) 
{
	GLenum status;
	status=(GLenum)glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);

	if (status == GL_FRAMEBUFFER_COMPLETE_EXT)
		return true;

	GPUCV_ERROR ("\n\n!!!!!!!!!!!!! FBO ERROR !!!!!!!!!!!!!!!");
	GPUCV_ERROR ("FILE  : " << File);
	GPUCV_ERROR ("LINE  : " << Line);
	GPUCV_ERROR ("CODE  : " << status);
	GPUCV_ERROR ("MSG	: ");
	std::string msg;
	switch(status) {
		case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT_EXT:
			msg="FrameBufferObject incomplete,incomplete attachment";
			break;
		case GL_FRAMEBUFFER_UNSUPPORTED_EXT:
			msg="Unsupported FrameBufferObject format";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT_EXT:
			msg="FrameBufferObject incomplete,missing attachment";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
			msg="FrameBufferObject incomplete,attached images must have same dimensions";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_FORMATS_EXT:
			msg="FrameBufferObject incomplete,attached images must have same format";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER_EXT:
			msg="FrameBufferObject incomplete,missing draw buffer";
			break;
		case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER_EXT:
			msg="FrameBufferObject incomplete,missing read buffer";
			break;
#ifdef GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT
		case GL_FRAMEBUFFER_INCOMPLETE_DUPLICATE_ATTACHMENT_EXT:
			msg="FrameBufferObject, incomplete duplicate attachment";
			break;
#endif
#ifdef GL_FRAMEBUFFER_STATUS_ERROR_EXT
		case GL_FRAMEBUFFER_STATUS_ERROR_EXT:
			msg="FrameBufferObject, incomplete status error";
			break;
#endif
		default : 
			msg="FrameBufferObject, unknown error status.";
			break;
	}
	if (msg!= "")
		GPUCV_ERROR ("MSG	: " << msg);
	GPUCV_ERROR("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

	return false;
}
//===============================================
/*virtual*/
std::ostringstream & FBO ::operator << (std::ostringstream & _stream)const
{
	_stream << LogIndent() <<"FBO==============" << std::endl;
	LogIndentIncrease();
	_stream << LogIndent() << "bool_rb:\t"		<< bool_rb	<< std::endl;
	_stream << LogIndent() << "FBOinternalFormat:\t" << GetStrGLInternalTextureFormat(FBOinternalFormat)	<< std::endl;
	_stream << LogIndent() << "active:\t"		<< active	<< std::endl;
	_stream << LogIndent() << "framebufID:\t"	<< framebuf	<< std::endl;
	_stream << LogIndent() << "m_currentNbrTextAttached:" << m_currentNbrTextAttached	<< std::endl;
	_stream << LogIndent() << "m_maxDrawBuffer:" << m_maxDrawBuffer	<< std::endl;
	_stream << LogIndent() << "List of objects====" << std::endl;
	LogIndentIncrease();
	std::string Type;
	for(unsigned int i=0; i < m_currentNbrTextAttached;i++)
	{
		_stream << LogIndent() << "ID:\t"				<<	m_drawBuffers[i].m_id << std::endl;
		_stream << LogIndent() << "Color attachement:\t"	<<	GetStrGLColorAttachment(m_drawBuffers[i].m_colorattachment)<< std::endl;
		switch (m_drawBuffers[i].m_type)
		{
		case TYPE_renderBuffer: Type = "renderBuffer"; break;
		case TYPE_texture:		Type = "texture"; break;
		case TYPE_undefined:	Type = "undefined"; break;
		default :				Type = "Unknown"; break;
		}
		_stream << LogIndent() << "type:\t"				<<	Type << std::endl;
		_stream << LogIndent() << "texture type:\t"		<<	GetStrGLTextureType(m_drawBuffers[i].m_textureType) << std::endl;
		_stream << LogIndent() << "width:\t"				<<	m_drawBuffers[i].m_width << std::endl;
		_stream << LogIndent() << "height:\t"			<<	m_drawBuffers[i].m_height << std::endl;
	}
	LogIndentDecrease();
	_stream << LogIndent() << "===================" << std::endl;

	LogIndentDecrease();
	_stream << LogIndent() <<"FBO==============" << std::endl;
	return _stream;
}
//===============================================
FBO* FBOManager()
{
	static FBO FboFilterSingleton(_GPUCV_FRAMEBUFFER_DFLT_FORMAT/*GL_RGB*/, false);
	//static FBO FboFilterSingleton(GL_RGB, false);
	return &FboFilterSingleton;
}
//===============================================

}//namespace GCV

