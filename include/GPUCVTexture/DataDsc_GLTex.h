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
#ifndef __GPUCV_TEXTURE_DATADSC_GLTEX_H
#define __GPUCV_TEXTURE_DATADSC_GLTEX_H

#include <GPUCVTexture/DataDsc_GLBase.h>
#include <GPUCVTexture/TexCoord.h>

namespace GCV{

#define GPUCV_TEXTURE_OPTIMIZED_ALLOCATION 1
/**	\brief DataDsc_GLTex is the OpenGL implementation of DataDsc_Base class. It is used to store and manipulate OpenGL textures.
*	\sa DataDsc_GLBase
*	\author Yannick Allusse
*/
class _GPUCV_TEXTURE_EXPORT DataDsc_GLTex
	:public virtual DataDsc_GLBase
	//,virtual public DataDsc_Base
{
public:
	typedef TextCoord<double>	TextCoordType;		//!< Type of texture coordinates.

	/**
	*	\brief Defines color attachment used by FBO(Pbuffer..??) for rendering to texture.
	*/
	enum attachmentEXT{
		NO_ATTACHEMENT		= 0,
		COLOR_ATTACHMENT0_EXT = GL_COLOR_ATTACHMENT0_EXT,
		COLOR_ATTACHMENT1_EXT = GL_COLOR_ATTACHMENT1_EXT,
		COLOR_ATTACHMENT2_EXT = GL_COLOR_ATTACHMENT2_EXT,
		COLOR_ATTACHMENT3_EXT = GL_COLOR_ATTACHMENT3_EXT,
		COLOR_ATTACHMENT4_EXT = GL_COLOR_ATTACHMENT4_EXT,
		COLOR_ATTACHMENT5_EXT = GL_COLOR_ATTACHMENT5_EXT,
		COLOR_ATTACHMENT6_EXT = GL_COLOR_ATTACHMENT6_EXT,
		COLOR_ATTACHMENT7_EXT = GL_COLOR_ATTACHMENT7_EXT,
		DEPTH_ATTACHMENT_EXT  = GL_DEPTH_ATTACHMENT_EXT,
		STENCIL_ATTACHMENT_EXT= GL_STENCIL_ATTACHMENT_EXT
	};

protected:
	GLuint				m_textureId_ARB;		//!< OpenGL ARB texture ID, used for multi texturing.
	attachmentEXT		m_textureAttachedID;	//!< OpenGL attachment ID used for rendering to texture.
	GLuint				m_textureType;			//!< OpenGL texture type, GL_TEXTURE_RECTANGLE/GL_TEXTURE_2D.
	GLint				m_internalFormat;		//!< OpenGL texture internal format.
	bool				m_needReload;			//!< Set to true when some parameters of the texture have been changed and need the texture to be reloaded.
	TextCoordType		*m_texCoord;	//!< Pointer to texture coordinates.
	//bool				m_isBinded;??
	bool				m_autoMipMap;	//!< Enable texture MipMapp auto generation, create new MipMaps every time we render to this texture or load data from CPU.
	std::string			m_strTextureName;//!< Texture name in the shader file, if not specified a generic name will be used such as BaseImage/Image0...
public:
	/** \brief Default constructor. */
	__GPUCV_INLINE
		DataDsc_GLTex(void);
	/** \brief Default destructor. */
	__GPUCV_INLINE virtual
		~DataDsc_GLTex(void);

	//Redefinition of global parameters manipulation functions
	virtual void	SetFormat(const GLuint _pixelFormat,const GLuint _pixelType);

	//Redefinition of data parameters manipulation

	//Redefinition of data manipulation
	__GPUCV_INLINE virtual void Allocate(void);
	__GPUCV_INLINE virtual bool IsAllocated(void)const;
	__GPUCV_INLINE virtual void Free();

	//Redefinition of DataDsc_Base interaction with other objects
	virtual bool CopyTo(DataDsc_Base* _destination, bool _datatransfer=true);
	virtual bool CopyFrom(DataDsc_Base* _source, bool _datatransfer=true);
	virtual	DataDsc_GLTex * Clone(DataDsc_GLTex * _src, bool _datatransfer=true);
	virtual	DataDsc_Base * CloneToNew(bool _datatransfer=true);


	//===========================================
	//local functions:
	//===========================================
	/**	\brief Set the "render to texture" mechanism to start for current texture.
	*	Start the mechanism to render to current texture. Should be use when we render to only one texture.
	*	If texture is not on GPU, a texture memory is allocated.
	*	If the texture option AUTO_MIPMAP is enable, mipmaps will be generated when calling UnSetRenderToTexture().
	*	\sa UnsetRenderToTexture().
	*/
	void	SetRenderToTexture();

	/**	\brief Unset the "render to texture" mechanism to stop for current texture
	*	Stop the mechanism to render to current texture. Should be use when we render to only one texture.
	*	If the texture option AUTO_MIPMAP is enable, mipmaps will be generated.
	*	\sa SetRenderToTexture().
	*/
	void	UnsetRenderToTexture();

	
	__GPUCV_INLINE virtual void _Bind(void) const;
	__GPUCV_INLINE virtual void _UnBind(void)const;
	__GPUCV_INLINE virtual void		_BindARB();
	__GPUCV_INLINE virtual void		_UnBindARB();
	__GPUCV_INLINE virtual GLuint	_GetARBId()const;
	__GPUCV_INLINE virtual void		_SetARBId(GLuint _texARBID);
	__GPUCV_INLINE virtual void		_SetColorAttachment(attachmentEXT _clr);
	__GPUCV_INLINE virtual attachmentEXT _GetColorAttachment()const;
	__GPUCV_INLINE virtual void		_SetTexType(GLuint _TexType);
	__GPUCV_INLINE virtual GLuint 	_GetTexType()const;
	__GPUCV_INLINE virtual GLenum	_GetInternalPixelFormat()const;
	__GPUCV_INLINE virtual void		_SetInternalPixelFormat(const GLuint _internalPixelFormat);
	__GPUCV_INLINE virtual void		_GetTexParami(GLuint _ParamType, GLint *_Value);
	__GPUCV_INLINE virtual void		_SetTexParami(GLuint _ParamType, GLint _Value);
	__GPUCV_INLINE virtual void		_SetAutoMipMap(bool _val);
	__GPUCV_INLINE virtual const std::string &	_GetTextureName()const;
	__GPUCV_INLINE virtual void		_SetTextureName(const char*);



	void	_Writedata(const PIXEL_STORAGE_TYPE ** _data, const bool _dataTransfer=true);

	PIXEL_STORAGE_TYPE ** _ReadData(PIXEL_STORAGE_TYPE ** _Pixelsdata, GLuint _xmin=0, GLuint _xmax=0, GLuint _ymin=0, GLuint _ymax=0);

	__GPUCV_INLINE void _SetReloadFlag(bool _val)
	{ m_needReload = _val;}
	__GPUCV_INLINE bool _GetReloadFlag()const
	{return m_needReload;}

	__GPUCV_INLINE void	_GlMultiTexCoordARB(GLuint _coordID);

	/**	\brief Return pointer to texture coordinate object.
	*	Texture coordinates are allocated on demand.
	*	\sa TextCoordType, _SetTextCoord(), _GlMultiTexCoordARB(),_GenerateTextCoord().
	*/
	__GPUCV_INLINE
		TextCoordType * 	_GetTextCoord();

	/**	\brief Set texture coordinate object.
	*	If m_texCoord is empty, we allocate new TextCoord object.
	*	Change texture coordinates if you are not using default one:
	*   <ul><li>GL_TEXTURE_2D=> {[0,0], [1,0],[1,1],[0,1]}</li>
	*	<li>GL_TEXTURE_2D=> {[0,0], [m_width,0],[m_width,m_height],[0,m_height]}</li></ul>
	*	\param _texCoord -> new texture coordinates, giving NULL value will delete existing TextCoord object.
	*	\sa TextCoordType, _GetTextCoord(), _GlMultiTexCoordARB(),_GenerateTextCoord().
	*/
	__GPUCV_INLINE
		void	_SetTextCoord(const TextCoordType * _texCoord);

	/**	\brief generate texture coordinate for given texture.
		*	\param _texCoord -> new texture coordinates, giving NULL value will delete existing TextCoord object.
		*	\sa TextCoordType, _GetTextCoord(), _SetTextCoord(), _GlMultiTexCoordARB().
		*/
		__GPUCV_INLINE
		TextCoordType * _GenerateTextCoord(void);

	/** \brief Activate the given texture coordinate using _coordID and the function glewMultiTexCoord2dARB.
	*	\param _coordID -> index of the quad corner [0..4].
	*	\sa TextCoordType, _GetTextCoord(),_SetTextCoord().
	*/
	__GPUCV_INLINE
		void	_GlMultiTexCoordARB(GLuint _coordID)const;

	/**	\brief Update texture coordinates depending on texture type.
	*	Set the texture coordinates to Width/Height if texture is GL_TEXTURE_RECTANGLE_ARB or to 1/1 for GL_TEXTURE_2D.
	*/
	__GPUCV_INLINE
		void	_UpdateTextCoord();

	/**
	*	\brief Initialize the current view port to match texture size and ROI
	*	\sa InitGLView(int, int), InitGLView(int, int, int, int)
	*/
	virtual
		void InitGLView();
	bool	FrameBufferToGpu(bool _dataTransfer=true);

	void	DrawFullQuad(const int _width, const int _height);
	void	DrawFullQuad(float centerX, float centerY, float scaleX, float scaleY, float width, float height);

	virtual std::ostringstream & operator << (std::ostringstream & _stream)const;
#if 0//to test and add
	/** \brief Check that internal format, or format and pixel type are defined, so the texture can be attached.
	*/
	__GPUCV_INLINE
		bool	IsValidForattachment()const;

	/** \brief Force OpenGL texture ID, and delete previous one is required.
	*	\param _texID -> New openGl texure ID.
	*	\sa _GetTexID().
	*/
	__GPUCV_INLINE
		void	_ForceTexID(const GLuint _texID);

#endif
};
}//namespace GCV
#endif//__GPUCV_TEXTURE_DATADSC_GLTEX_H
