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



#ifndef __GPUCV_TEXTURE_TRB_H
#define __GPUCV_TEXTURE_TRB_H

#include <GPUCVTexture/TextureGrp.h>
#include <GPUCVTexture/DataDsc_GLTex.h>

namespace GCV{

/** @defgroup RENDERTOTEXT_GRP Rendering to textures.
	@ingroup GPUCV_SDK_GRP
*  This group describes the different tools available to render OpenGL scenes directly into texture without using the frame buffer.
*  @{
*/

/*
#define _RENDER_BUFFER_TWINFBO				1


#if !_RENDER_BUFFER_TWINFBO && _GPUCV_TEXTURE_SUPPORT_PBUFFER
#define USE_PBUFF	1
#define USE_FBO		0
#elif _GPUCV_TEXTURE_SUPPORT_FBO
#define USE_PBUFF	0
#define USE_FBO		1
#else
#pragma message("Warning, you should define _GPUCV_TEXTURE_SUPPORT_PBUFFER or _GPUCV_TEXTURE_SUPPORT_FBO");
#endif
*/

#define _GPUCV_USE_TEMP_FBO_TEX 0

/*!
*	\brief Supply a base class to create Render to texture class object like FBO or PBuffer depending on hardware compatibilities.
*	\note this class supply basic openlGL rendering to texture in case no hardware is supported...
*	\author Yannick Allusse
*/
class _GPUCV_TEXTURE_EXPORT TextureRenderBuffer 
	: public CL_Profiler
{
public:
	/*!
	*	\brief Define the Render Buffer type.
	*/
	enum RENDER_OBJ_TYPE{
		RENDER_OBJ_AUTO,			//!< Choose the best Render Buffer depending on Hardware.
		//RENDER_OBJ_NOT_SUPPORTED,	//
		RENDER_OBJ_OPENGL_BASIC,	//!< Use classic glReadPixels().
		RENDER_OBJ_FBO,				//!< Use FBO.
		RENDER_OBJ_PBUFF			//!< Use PBuffer.
	};

#if _GPUCV_USE_TEMP_FBO_TEX
	//! Define the Render Buffer internal format.
	enum RENDER_INTERNAL_OBJ{
		//RENDER_FP8_RGBA,
		RENDER_FP16_RGBA,
		RENDER_FP32_RGBA_1,
		RENDER_FP32_RGBA_2,
		RENDER_FP8_RGB,
		//	RENDER_FP16_RGB,
		//	RENDER_FP32_RGB
	};
#endif

protected:
	RENDER_OBJ_TYPE		m_type;			//!< Render Buffer Type.
	DataDsc_GLTex	*	m_textID;		//!< ID of the texture linked to the buffer.
	TextureGrp		*	m_textureGrp;	//!< Group of textue to render to.
	GLuint				m_width;		//!< Width of the buffer.
	GLuint				m_height;		//!< Height of the buffer.

public:

	/*!
	*	\brief Default constructor
	*/
	TextureRenderBuffer(bool verbose =false);
	/*!
	*	\brief Default destructor
	*/
	virtual ~TextureRenderBuffer();

	//virtual std::string LogException(void)const;
	virtual std::ostringstream & operator << (std::ostringstream & _stream)const;
	/*!
	*	\brief Initialize RenderBuffer.
	*/
	virtual bool InitRenderBuff();

	virtual void SetContext(DataDsc_GLTex * _tex);
	virtual void SetContext(DataDsc_GLTex * _tex, int _width, int _height);
	/*!
	*	\brief Force the Render Buffer to render to multiple textures using a texture group.
	*	\sa TextureGrp, DataContainer.
	*	\note Each texture contains a field m_textureAttachedID that specify the color attachment to render to.
	*/

	virtual int SetContext		(TextureGrp *_grp, int _width, int _height);


	virtual void UnSetContext	();

	//virtual void MapTexture()=0;

	/*!
	*	\brief Force the Render Buffer to render to the texture
	*	\param _tex	=> Texture to render to.
	*/
	virtual 
		void Force(DataDsc_GLTex * _tex);

	/*!
	*	\brief Force the Render Buffer to render to the texture
	*	\param _tex	=> Texture to render to.
	*	\param _width	=> width to use if different than texture width.
	*	\param _height	=> height to use if different than texture height.
	*/
	virtual 
		void Force(DataDsc_GLTex * _tex, int _width, int _height);

	/*!
	*	\brief Force the Render Buffer to render to multiple textures using a texture group.
	*	\sa TextureGrp, DataContainer.
	*	\note Each texture contains a field m_textureAttachedID that specify the color attachment to render to.
	*/
	virtual 
		int  Force(TextureGrp * _grp);
	virtual 
		int  Force(TextureGrp * _grp, int _width, int _height);

	/*!
	*	\brief Unforce the Render Buffer.
	*	\note Before calling UnForce(), you must call GetResult() to insure maximum compatibilities.
	*/
	virtual void UnForce(void);

	/*!
	*	\brief Return the forced status.
	*/
	virtual int  IsForced(void);

	/*!
	*	\brief Get the result into the texture.
	*	\note Before calling UnForce(), you must call GetResult() to insure maximum compatibilities.
	*/
	virtual void GetResult(void);

	/*!
	*	\brief Get the result into the texture.
	*	\note Before calling UnForce(), you must call GetResult() to insure maximum compatibilities.
	*/
	virtual void GetResult(DataDsc_GLTex * _tex);

	/*!
	*	\brief Return the Render Object type.
	*/
	virtual RENDER_OBJ_TYPE GetType();

	/*!
	*	\brief Return the Render Object singleton.
	*/
	static	TextureRenderBuffer * GetMainRenderer();

	/*!
	*	\brief Set the Render Object singleton.
	*/
	static	TextureRenderBuffer * SetMainRenderer(TextureRenderBuffer * _Renderer);

	bool	IsTextureBinded(DataDsc_GLTex * _tex);


#if _GPUCV_USE_TEMP_FBO_TEX
	//access built-in textures...
	virtual GPUCV_TEXT_TYPE SelectInternalTexture(RENDER_INTERNAL_OBJ _InternTextID);
#endif

protected:
	void PushAttribs();
	void PopAttribs();
	/*!
	*	\brief Affect texture ID and size to the Render Object.
	*/
	void SetValues(DataDsc_GLTex * _tex);
	void SetValues(DataDsc_GLTex * _tex, int _width, int _height);

	static	TextureRenderBuffer * MainRenderer;//!< Pointer to the singleton.
};


/** @} */ // end of RENDERTOTEXT_GRP

}//namespace GCV
#endif// _PIXEl_BUFFER_OBJ_H
