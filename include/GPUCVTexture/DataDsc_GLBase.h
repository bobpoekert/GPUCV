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
#ifndef __GPUCV_TEXTURE_DATADSC_GLBASE_H
#define __GPUCV_TEXTURE_DATADSC_GLBASE_H

#include <GPUCVTexture/DataDsc_base.h>

namespace GCV{

/**	\brief DataDsc_GLBase is the OpenGL base class implementation of DataDsc_Base class.
*	\sa DataDsc_Base
*	\author Yannick Allusse
*/
class _GPUCV_TEXTURE_EXPORT DataDsc_GLBase
	:public virtual DataDsc_Base
{
protected:
	//! OpenGL object ID (texture/buffer ID).
	_DECLARE_MEMBER_CONST(GLuint, GLId);
public:
	/** \brief Default constructor. */
	__GPUCV_INLINE
		DataDsc_GLBase(void);
	/** \brief Default destructor. */
	__GPUCV_INLINE virtual
		~DataDsc_GLBase(void);

	//Redefinition of global parameters manipulation functions
	virtual void Flush(void);


	//Redefinition of data parameters manipulation
	virtual void	ConvertPixelFormat_GLToLocal(const GLuint _pixelFormat);
	virtual GLuint	ConvertPixelFormat_LocalToGL(void);
	virtual void	ConvertPixelType_GLToLocal(const GLuint _pixelType);
	virtual GLuint	ConvertPixelType_LocalToGL(void);
	//Redefinition of data manipulation

	//Redefinition of DataDsc_Base interaction with other objects
	virtual DataDsc_Base * CloneToNew(bool _datatransfer=true)=0;

	
	//===========================================
	//local functions:
	//===========================================

	__GPUCV_INLINE virtual void _Bind(void) const=0;
	__GPUCV_INLINE virtual void _UnBind(void)const=0;

	virtual std::ostringstream & operator << (std::ostringstream & _stream)const;
};
}//namespace GCV
#endif//__GPUCV_TEXTURE_DATADSC_GLTEX_H
