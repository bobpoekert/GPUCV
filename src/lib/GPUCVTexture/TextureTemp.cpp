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
#include <GPUCVTexture/TextureTemp.h>

namespace GCV{
//=====================================================
/*explicit*/
TextureTemp::TextureTemp(DataContainer * _tex)
:DataContainer()
,m_sourceTexture(NULL)
{
	SetSourceTexture(_tex);
}
//=====================================================
TextureTemp::~TextureTemp()
{
}
//=====================================================
void TextureTemp::SetSourceTexture(DataContainer * _tex)
{
	m_sourceTexture = _tex;
	if(!m_sourceTexture)
		return;

	this->_CopyProperties(*_tex);
	//we copy the last data descriptor to have at least the last data format and size(without data)
	if(_tex->GetLastDataDsc())
		this->AddNewDataDsc(_tex->GetLastDataDsc()->CloneToNew(false));

}	
//=====================================================
DataContainer * TextureTemp::GetSourceTexture()const
{
	return m_sourceTexture;
}

}//namespace GCV
