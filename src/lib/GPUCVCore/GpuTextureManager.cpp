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
#include <GPUCVCore/GpuTextureManager.h>
#include <GPUCVCore/coretools.h>

namespace GCV{

//Initialize singleton
//template <> TextureManager * CL_Singleton<TextureManager>::m_registeredSingleton = NULL;

//=======================================================
TextureManager :: TextureManager()
	:SGE::CL_TEMPLATE_OBJECT_MANAGER/*CL_TplObjManager*/<DataContainer, TypeIDPtr>(NULL)
	,CL_Singleton<TextureManager>()
{
}
//=======================================================
TextureManager :: ~TextureManager()
{
	GPUCV_DEBUG("TextureManager Destructor");
	PrintAllObjects();
	TplManager::iterator itTex;
	for ( itTex = GetFirstIter( ) ; itTex!= GetLastIter( ) ;itTex++ )
	{
		delete((*itTex).second);
	}
	DeleteAllLocal();
}
//=======================================================
DataContainer* TextureManager ::Find(TypeIDPtr _ID)
{
	SG_Assert(_ID, "TextureManager ::Find()=> _ID obj is null");
	DataContainer* temp= TplManager::Get(_ID);
	return temp;
}
//=======================================================
/*virtual*/
void TextureManager :: PrintAllObjects()
{
#if 1
	//show all objects labels
	TplManager::PrintAllObjects();
	//add some memory informations
	GPUCV_NOTICE("Memory informations:" << GetCount());
	GetGpuCVSettings()->PushSetOptions(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG, true);
	TplManager::iterator itTex=GetFirstIter();
	for (; itTex!= GetLastIter( ) ;itTex++ )
	{
//		(*itTex).second->PushSetOptions(CL_Options::LCL_OPT_DEBUG, true);
//		(*itTex).second->SetOption(DataContainer::DBG_IMG_MEMORY, true);
		GPUCV_NOTICE("___________________");
		GPUCV_NOTICE("ID: " << (*itTex).second->GetIDStr() << " => " << (*itTex).second->GetValStr());
//		(*itTex).second->PrintMemoryInformation("");
//		(*itTex).second->PopOptions();
	}
	GetGpuCVSettings()->PopOptions();
#endif
}

//=======================================================
}//namespace GCV
