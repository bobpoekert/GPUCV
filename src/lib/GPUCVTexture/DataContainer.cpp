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
#include <GPUCVTexture/DataContainer.h>
#include <GPUCVTexture/TextureRenderManager.h>
#include <GPUCVTexture/TextureRecycler.h>
#include <GPUCVTexture/DataDsc_CPU.h>
//#include <GPUCVTexture/DataDsc_GLTex.h>

/*#if USE_PBOjects
#define BUFFER_OFFSET(i) ((char *)NULL + (i))
#endif
*/
namespace GCV{
//===============================
//===============================
//	Textures
//===============================
//===============================

//=========================================
DataContainer::DataContainer(DataContainer &_Copy, bool _dataOnly)
:
CL_Profiler("DataContainer")
,SGE::CL_BASE_OBJ<TplIDType>(NULL)
,m_textureLastLocations(NULL)
,m_textureLocationsArrLastID(0)
#if _GPUCV_DEPRECATED
,m_nChannel(0)
,m_nChannel_force(false)
#endif
//	,m_location(LOC_NO_LOCATION)
,m_needReload(true)
,m_SwitchLockImplementationID(GPUCV_IMPL_AUTO)
{
	CLASS_FCT_SET_NAME("DataContainer");
	CLASS_DEBUG("");

	for (int i = 0; i < _GPUCV_MAX_TEXTURE_DSC; i++)
	{
		m_textureLocationsArr[i] = NULL;
	}

	//we do not set default option here, cause we will use _Copy object.
	_CopyProperties(_Copy);
	_CopyAllDataDsc(_Copy, _dataOnly);
}

//=================================================
DataContainer::DataContainer(const TplIDType _ID)
:
CL_Profiler("DataContainer")
,SGE::CL_BASE_OBJ<TplIDType>(NULL)
,m_textureLastLocations(NULL)
,m_textureLocationsArrLastID(0)
#if _GPUCV_DEPRECATED
	,m_nChannel(0)
	,m_nChannel_force(false)
#endif
//	m_location(LOC_NO_LOCATION)
,m_needReload(true)
,m_SwitchLockImplementationID(GPUCV_IMPL_AUTO)
{
	CLASS_FCT_SET_NAME("DataContainer");
	CLASS_DEBUG("");

	for (int i = 0; i < _GPUCV_MAX_TEXTURE_DSC; i++)
	{
		m_textureLocationsArr[i] = NULL;
	}
	SGE::CL_BASE_OBJ<TplIDType>::SetID(_ID);
}
//=================================================
DataContainer::~DataContainer()
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("~DataContainer");
	//no Bench and log in Destructor//	CLASS_DEBUG("");
	RemoveAllTextureDesc();
}
//=================================================

void DataContainer::_CopyProperties(DataContainer &_src)
{
	CLASS_FCT_SET_NAME("_CopyProperties");
	CLASS_DEBUG("");
	//set options
	SetOption(_src.GetOption(0xFFFFF), true);
	m_label = _src.m_label;
	m_label += "_clone";

	//texture parameters...???
}
//=================================================
/*virtual*/
void DataContainer::_CopyActiveDataDsc(DataContainer &_src, bool _dataOnly)
{
	CLASS_FCT_SET_NAME("_CopyActiveDataDsc");
	CLASS_DEBUG("");

	//copy last location from source to destination
	//remove other destination instances...execpt CPU one that represent the link with current app
	RemoveAllDataDscExcept<DataDsc_CPU>();

	//copy new one
	if(_src.m_textureLastLocations)
	{//if last location have data, we copy only last location
		if(_src.m_textureLastLocations->HaveData())
		{
			AddNewDataDsc( _src.m_textureLastLocations->CloneToNew(_dataOnly));
			return;
		}
		else
		{
			if(_dataOnly)
			{
				CLASS_WARNING("A copy is requested with a source having no DATA, performing allocation instead of copy!");
			}
			AddNewDataDsc( _src.m_textureLastLocations->CloneToNew(false));
		}
	}
	else
	{
		CLASS_ASSERT(0, "A copy is requested with a source having no DataDsc to copy!");
	}
	return;
}

//=================================================
/*virtual*/
void DataContainer::_CopyAllDataDsc(DataContainer &_src, bool _dataOnly)
{
	CLASS_FCT_SET_NAME("_CopyAllDataDsc");
	CLASS_DEBUG("");

	//remove other destination instances...execpt CPU one that represent the link with current app
	RemoveAllDataDscExcept<DataDsc_CPU>();

	//copy new one
	if(_dataOnly && _src.m_textureLastLocations)
	{//if last location have data, we copy only last location
		if(_src.m_textureLastLocations->HaveData())
		{
			AddNewDataDsc( _src.m_textureLastLocations->CloneToNew(true));
			return;
		}
	}
	else
	{//we try to copy all locations
		for (int i = 0; i < _src.m_textureLocationsArrLastID; i++)
		{
			if(_src.m_textureLocationsArr[i]->HaveData())
				//this->m_textureLocationsArr[i] = _src.m_textureLocationsArr[i]->CloneToNew(true);
				AddNewDataDsc(_src.m_textureLocationsArr[i]->CloneToNew(true));
			else if (!_dataOnly)
				//this->m_textureLocationsArr[i] = _src.m_textureLocationsArr[i]->CloneToNew(false);
				AddNewDataDsc(_src.m_textureLocationsArr[i]->CloneToNew(false));
			/*
			this->m_textureLocationsArr[i]->SetParent(this);
			//change parent ID cause it might have changed...
			if(this->m_textureLocationsArr[i]->GetNewParentID())
			this->SetID(this->m_textureLocationsArr[i]->GetNewParentID());
			//changing the ID here is not sufficent, we need to change it in the manager to???
			m_textureLocationsArrLastID++;
			*/
		}
	}
}
//=================================================
void	DataContainer::SetLabel(const std::string _lab)
{
	m_label = _lab;
}
//=================================================
const std::string & DataContainer::GetLabel()const
{
	return m_label;
}
//=================================================
void DataContainer::SetOption(CL_Options::OPTION_TYPE _opt, bool val)
{
	CLASS_FCT_SET_NAME("SetOption");
	CLASS_DEBUG("opt:" << _opt << "\tval:" << val);
	CL_Options::SetOption(_opt, val);
}
//=================================================
#if _GPUCV_DEPRECATED
/*virtual*/
void DataContainer::GenerateMipMaps(GLuint _textureMinFilter/* = GL_NEAREST_MIPMAP_NEAREST*/)
{
	if(_IsLocation(LOC_GPU))
	{//texture already exists and contains the data
		//we need to use
		_Bind();//bind it.
		glGenerateMipmapEXT(m_textureType);
		_SetTexParami(GL_TEXTURE_MIN_FILTER, _textureMinFilter);//set filter settings
	}
}

//=================================================
/**
\todo Add dynamic Max texture size using glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTexSize);
*/

void DataContainer::_SetSize(const unsigned int width, const unsigned int height)
{
	if (m_width != int(width) || m_height != int(height))
	{
		_SetReloadFlag(true);//must be reloaded.
		//if (m_pixels)
		//	delete m_pixels;//...is it safe...????
	}
	else
		return;

	SG_Assert(width	<= _GPUCV_TEXTURE_MAX_SIZE_X, "DataContainer::SetSize()=> Texture width exceed Max size of " << _GPUCV_TEXTURE_MAX_SIZE_X);
	SG_Assert(heigh	<= _GPUCV_TEXTURE_MAX_SIZE_Y, "DataContainer::SetSize()=> Texture heigh exceed Max size of " << _GPUCV_TEXTURE_MAX_SIZE_Y);

	TextSize<GLsizei>::_SetSize(width,heigh);
#if !_GPUCV_USE_DATA_DSC
	_UpdateTextCoord();
#endif
}
#endif
//=================================================
PIXEL_STORAGE_TYPE **
DataContainer::_GetPixelsData()
{
	CLASS_FCT_SET_NAME("_GetPixelsData");
	CLASS_DEBUG("");
	char CpuTexId = FindDataDscID<DataDsc_CPU>();
	if(CpuTexId != -1)
	{
		return GetDataDsc<DataDsc_CPU>()->_GetPixelsData();
	}
	return NULL;
}
//=================================================
#if _GPUCV_DEPRECATED
void
DataContainer::_SetPixelsData(PIXEL_STORAGE_TYPE ** _pix)
{
	if(_pix==NULL)
	{
		GetDataDsc<DataDsc_CPU>()->Free();
	}
	else
	{
		GetDataDsc<DataDsc_CPU>()->_SetPixelsData(_pix);
		_SetReloadFlag(true);
	}
	//DataDsc_CPU* TexCpu = FindTex
	//m_pixels = _pix;
}
#endif
//=================================================
/*virtual*/
const std::string
DataContainer::GetValStr() const
{
	if(GetLabel() != "")
		return "ID:'" + GetLabel() +"'";//--Object memory allocated:" + SGE::ToCharStr(GetMemoryAllocated());
	else
		return "ID: noname.";
}

//=================================================
void DataContainer::Print() const
{
	GPUCV_LOCAL_DEBUG("=========================================================");
	if(m_label != "")
	{
		GPUCV_LOCAL_DEBUG("\tTexture Label:			"	<< m_label);
	}
	GPUCV_LOCAL_DEBUG("\tDescription:			" << GetValStr());
	GPUCV_LOCAL_DEBUG("=========================================================");
	GPUCV_LOCAL_DEBUG("\tShowing all locations informations:");
	for (int i = 0; i < m_textureLocationsArrLastID; i++)
	{
		GPUCV_LOCAL_DEBUG(*m_textureLocationsArr[i]);
	}
	GPUCV_LOCAL_DEBUG("=========================================================");
}
//=================================================
/*virtual*/
std::string DataContainer::PrintMemoryInformation(std::string text)const
{
	GPUCV_LOCAL_DEBUG("=========================================================");
	GPUCV_LOCAL_DEBUG("Memory information:			" << GetValStr());
	GPUCV_LOCAL_DEBUG("---------------------------------------------------------");
	GPUCV_LOCAL_DEBUG("Locations informations:");
	for (int i = 0; i < m_textureLocationsArrLastID; i++)
	{
		GPUCV_LOCAL_DEBUG(m_textureLocationsArr[i]->PrintMemoryInformation(""));
	}
	GPUCV_LOCAL_DEBUG("=========================================================");
	return "";//..?why returning a string here?
}
//=================================================
void DataContainer::ForceRenderToTexture()
{
	GetDataDsc<DataDsc_GLTex>()->SetRenderToTexture();
}
//=================================================
void	DataContainer::UnForceRenderToTexture()
{
	GetDataDsc<DataDsc_GLTex>()->UnsetRenderToTexture();
}
//=================================================
void	DataContainer::SetRenderToTexture()
{
	DataContainer::ForceRenderToTexture();
}
//=================================================
void DataContainer::UnsetRenderToTexture()
{
	DataContainer::UnForceRenderToTexture();
}
//=================================================
//=========================================================
// Simple openGL wrapping function, that are not calling
// any other member function and not performing any tests
//=========================================================
//=========================================================
//=========================================================
char DataContainer::RemoveAllTextureDesc()
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("RemoveAllTextureDesc");
	//no Bench and log in Destructor//	CLASS_DEBUG("");
	int i=0;
	for (int i = 0; i < m_textureLocationsArrLastID; i++)
	{
		if( m_textureLocationsArr[i])
			if(!m_textureLocationsArr[i]->IsLocked())
			{
				delete m_textureLocationsArr[i];
				m_textureLocationsArr[i] = NULL;
			}
	}
	//perform it twice cause some objects can be locked by other
	for (int i = 0; i < m_textureLocationsArrLastID; i++)
	{//no delete every thing...
		if( m_textureLocationsArr[i])
		{
			delete m_textureLocationsArr[i];
			m_textureLocationsArr[i] = NULL;
		}
	}
	m_textureLocationsArrLastID = 0;
	return i;
}
//========================================================
GLuint GetGLDepth(DataContainer * tex)
{
	SG_Assert(tex, "Texture empty");
	return GetGLDepth(tex->GetLastDataDsc());
}
GLuint GetGLDepth(DataDsc_Base * tex)
{
	SG_Assert(tex, "Texture empty");
	return tex->GetPixelType();
}
//========================================================
GLuint GetWidth(DataContainer * tex)
{
	SG_Assert(tex, "Texture empty");
	return GetWidth(tex->GetLastDataDsc());
}
GLuint GetWidth(DataDsc_Base * tex)
{
	SG_Assert(tex, "Texture empty");
	return tex->_GetWidth();
}
//========================================================
GLuint GetHeight(DataContainer * tex)
{
	SG_Assert(tex, "Texture empty");
	return GetHeight(tex->GetLastDataDsc());
}
GLuint GetHeight(DataDsc_Base * tex)
{
	SG_Assert(tex, "Texture empty");
	return tex->_GetHeight();
}
//========================================================
GLuint GetnChannels(DataContainer * tex)
{
	SG_Assert(tex, "Texture empty");
	return GetnChannels(tex->GetLastDataDsc());
}
GLuint GetnChannels(DataDsc_Base * tex)
{
	SG_Assert(tex, "Texture empty");
	return tex->GetnChannels();
}
//========================================================

}//namespace GCV

