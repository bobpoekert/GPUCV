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
#include <GPUCVTexture/TextureRecycler.h>
#include <GPUCVTexture/DataContainer.h>
#include <GPUCVTexture/DataDsc_CPU.h>
namespace GCV{

//==================================================
DataDsc_CPU::DataDsc_CPU()
: DataDsc_Base("DataDsc_CPU")
,m_pixels(NULL)
{
}
//==================================================
DataDsc_CPU::DataDsc_CPU(const std::string _ClassName)
: DataDsc_Base(_ClassName)
,m_pixels(NULL)
{
}
//==================================================
DataDsc_CPU::~DataDsc_CPU(void)
{
	Free();
}
//==================================================
/*virtual*/
DataDsc_Base * DataDsc_CPU::CloneToNew(bool _datatransfer/*=true*/)
{
	DataDsc_CPU * TempTex = new DataDsc_CPU();
	return TempTex->Clone(this,_datatransfer);
}
//==================================================
/*virtual*/
DataDsc_CPU*	DataDsc_CPU::Clone(DataDsc_CPU* _source, bool _datatransfer/*=true*/)
{
	DataDsc_Base::Clone(_source, _datatransfer);
	Allocate();
	if(_datatransfer && _source->HaveData())
	{
		memcpy(*m_pixels, *_source->m_pixels, GetMemSize());
		SetDataFlag(true);
	}
	return this;
}
//==================================================
/*virtual*/
bool DataDsc_CPU::CopyTo(DataDsc_Base* _destination, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_destination->GetDscType(),"CopyTo");
	TEXTUREDESC_COPY_START(_destination,_datatransfer);

	//copy to CPU => clone
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_destination);
	if(TempCPU)
	{
		UnsetReshape();
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CPU", "DD_CPU", this, TempCPU,_datatransfer);
		return TempCPU->Clone(this, _datatransfer)?true:false;;
	}
	//======================
	return false;
}
//==================================================
/*virtual*/
bool DataDsc_CPU::CopyFrom(DataDsc_Base* _source, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_source->GetDscType(),"CopyFrom");
	TEXTUREDESC_COPY_START(_source,_datatransfer);


	//copy from CPU => clone
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_source);
	if(TempCPU)
	{
		_source->UnsetReshape();
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CPU", "DD_CPU", TempCPU, this,_datatransfer);
		return Clone(TempCPU, _datatransfer)?true:false;;
	}
	//======================
	return false;
}
//==================================================
/*virtual*/
//! \todo Check if data size has changed before freeing and reallocating memory.
void DataDsc_CPU::Allocate(void)
{
	CLASS_FCT_SET_NAME("Allocate");
	CLASS_FCT_PROF_CREATE_START();

	if(IsAllocated())
	{
		Free();
	}
	DataDsc_Base::Allocate();
	CLASS_DEBUG("Allocate buffer of size " << m_memSize);
	//CLASS_ASSERT(m_pixels, "DataDsc_CPU::Allocate()=> Target buffer has no adress");
	*m_pixels = new unsigned char [m_memSize];
	Log_DataAlloc(m_memSize);
	CLASS_ASSERT(*m_pixels, "DataDsc_CPU::Allocate()=> Allocation failed");
}
//==================================================
/*virtual*/
void DataDsc_CPU::Free()
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("Free");
	if(IsAllocated())
	{
		
		//no Bench and log in Destructor//		CLASS_FCT_PROF_CREATE_START();
		delete (char *) (*m_pixels);
		*m_pixels = NULL;
		Log_DataFree(m_memSize);
		//no Bench and log in Destructor//		CLASS_DEBUG("Free()");
		DataDsc_Base::Free();
	}
}
//==================================================
/*virtual*/
bool DataDsc_CPU::IsAllocated()const
{
	if(!m_pixels)
		return false;
	return (*m_pixels)? true:false;
}
//==================================================

//==================================================
//==================================================
//Local functions
//==================================================
//==================================================
/*virtual*/
PIXEL_STORAGE_TYPE** DataDsc_CPU::_GetPixelsData()
{
	return m_pixels;
}
/*virtual*/
void DataDsc_CPU::_SetPixelsData(PIXEL_STORAGE_TYPE** _data)
{
	m_pixels = _data;
}

//==================================================
std::ostringstream & DataDsc_CPU::operator << (std::ostringstream & _stream)const
{
	_stream << std::endl;
	_stream << std::endl;
	DataDsc_Base::operator <<(_stream);
	_stream << LogIndent() <<"DataDsc_CPU==============" << std::endl;
	LogIndentIncrease();
	_stream << LogIndent() <<"m_pixels: \t\t\t\t"		<< m_pixels << std::endl;
	LogIndentDecrease();
	_stream << LogIndent() <<"DataDsc_CPU==============" << std::endl;
	return _stream;
}
}//namespace GCV

