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



#ifndef __GPUCV_TEXTURE_TRB_FBO_H
#define __GPUCV_TEXTURE_TRB_FBO_H


/* Pixel Buffer Object */
#include "GPUCVTexture/TextureRenderBuffer.h"
#include "GPUCVTexture/fbo.h"

namespace GCV{

/** @addtogroup RENDERTOTEXT_GRP
*  @{
*/

/*! 
*	\brief Supply functions to use FBO.
*	\author Yannick Allusse
*	\sa TextureRenderBuffer
*/
class _GPUCV_TEXTURE_EXPORT TextureRenderBufferFBO
	:public TextureRenderBuffer
{
#if _GPUCV_USE_TEMP_FBO_TEX
protected:
	DataContainer * TexFP16;
	DataContainer * TexFP32;
	DataContainer * TexFP32_1;
	DataContainer * TexFP32_2;
	DataContainer * TexFP8_RGB;
#endif
public:

	TextureRenderBufferFBO(bool verbose =false);
	virtual ~TextureRenderBufferFBO();
	virtual std::ostringstream & operator << (std::ostringstream & _stream)const;
	virtual bool InitRenderBuff();
	virtual void SetContext		(DataDsc_GLTex * _tex, int _width, int _height);
	virtual int  SetContext		(TextureGrp * _grp, int _width, int _height);
	virtual void UnSetContext	();
	virtual int  Force(TextureGrp * _grp, int _width, int _height);
	virtual void Force(DataDsc_GLTex * _tex, int _width, int _height);
	virtual void UnForce(void);
	virtual int  IsForced(void);
	virtual void GetResult(void){};//noting to do here...
	virtual const std::string GetValStr()const{return "NOID";}

#if _GPUCV_USE_TEMP_FBO_TEX
	virtual GPUCV_TEXT_TYPE SelectInternalTexture(RENDER_INTERNAL_OBJ _InternTextID);
#endif
protected:
	virtual void ClearRenderBuff(){};
};

/** @} */ // end of RENDERTOTEXT_GRP
}//namespace GCV

#endif
