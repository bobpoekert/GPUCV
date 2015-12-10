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
#ifndef __GPUCV_TEXTURE_TEMP_H
#define __GPUCV_TEXTURE_TEMP_H
#include <GPUCVTexture/DataContainer.h>


namespace GCV{
/** \brief TextureTemp class is a texture object linked to another texture. Is is used to be manipulated(Write) when the original texture can't be manipulated.
*	\sa DataContainer.
*/
class _GPUCV_TEXTURE_EXPORT TextureTemp
	:public DataContainer
{
public:

	__GPUCV_INLINE
		explicit TextureTemp(DataContainer * _tex);

	__GPUCV_INLINE
		~TextureTemp();

	void SetSourceTexture(DataContainer * _tex);
	DataContainer * GetSourceTexture()const;
private:
	DataContainer * m_sourceTexture;
};
}//namespace GCV
#endif
