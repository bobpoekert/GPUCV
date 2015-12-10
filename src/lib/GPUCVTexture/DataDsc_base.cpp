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
#include <GPUCVTexture/DataDsc_base.h>


namespace GCV{

long												DataDsc_Base::ms_totalMemoryAllocated	= 0;
#if _GPUCV_PROFILE
SG_TRC::CL_CLASS_TRACER<CL_Profiler::TracerType>*	DataDsc_Base::m_TransferClassTracer		= NULL;
#endif

//-------------------------------------
//Constructors/Destructors
//-------------------------------------
//==================================================
DataDsc_Base::DataDsc_Base()
:CL_Profiler("DataDsc_Base")
,TextSize<GLsizei>()
,m_dscType("DataDsc_Base")
,m_glPixelFormat(0)
,m_glPixelType(0)
,m_nChannels(0)
,m_nChannelsBackup(0)
,m_parent(NULL)
,m_haveData(false)
,m_lockedBy(NULL)
,m_lockedObj(NULL)
,m_dataSize(0)
,m_memSize(0)
,m_localMemoryAllocated(0)
{
#if _GPUCV_PROFILE
	if(m_TransferClassTracer==NULL)
	{//first DD created, ask for transfer bench class
		GET_CLASSMNGR_MUTEX(MutexClassManager);
		m_TransferClassTracer = MutexClassManager->Add("transfers");
		m_TransferClassTracer->SetParentAppliTracer(&GetTimeTracer());
	}
#endif
}
//==================================================
DataDsc_Base::DataDsc_Base(const std::string _className)
:CL_Profiler(_className)
,TextSize<GLsizei>()
,m_dscType(_className)
,m_glPixelFormat(0)
,m_glPixelType(0)
,m_nChannels(0)
,m_nChannelsBackup(0)
,m_parent(NULL)
,m_haveData(false)
,m_lockedBy(NULL)
,m_lockedObj(NULL)
,m_dataSize(0)
,m_memSize(0)
,m_localMemoryAllocated(0)
{
#if _GPUCV_PROFILE
	if(m_TransferClassTracer==NULL)
	{//first DD created, ask for transfer bench class
		GET_CLASSMNGR_MUTEX(MutexClassManager);
		m_TransferClassTracer = MutexClassManager->Add("transfers");
		m_TransferClassTracer->SetParentAppliTracer(&GetTimeTracer());
	}
#endif
}
//==================================================
DataDsc_Base::~DataDsc_Base(void)
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("~DataDsc_Base");
	//no Bench and log in Destructor//	CLASS_DEBUG("");
	if(IsAllocated())
	{
		Free();
	}
}
//==================================================
//-------------------------------------
//Integration with framework
//-------------------------------------
/*virtual*/
std::ostringstream & DataDsc_Base::operator << (std::ostringstream & _stream)const
{
	_stream << LogIndent() <<"======================================" << std::endl;
	_stream << LogIndent() <<"DataDsc_Base==============" << std::endl;
	LogIndentIncrease();
		if(m_parent)
			_stream << LogIndent() <<"m_parent => \t\t\t\t"		<< m_parent->GetValStr() << std::endl;
		//_stream << "Parent ID:" << GetValStr() << std::endl;
		//_stream << "DataDsc_Base::m_dscType => "		<< m_dscType << std::endl;
		_stream << LogIndent() <<"m_haveData: \t\t\t\t"	<< m_haveData << std::endl;
		_stream << LogIndent() <<"Pixel format/Type: \t\t\t\t"	<< GetStrGLTextureFormat(m_glPixelFormat) << " / " << GetStrGLTexturePixelType(m_glPixelType) << std::endl;
		_stream << LogIndent() <<"nChannels / nChannelsBackup: \t\t"	<< m_nChannels <<  " / " << m_nChannelsBackup << std::endl;
		_stream << LogIndent() <<"Image Size: \t\t\t\t"			<< m_width << "/" << m_height << std::endl;
		_stream << LogIndent() <<"Data / Memory size: \t\t\t"		<< m_dataSize << " / " << m_memSize << std::endl;
		_stream << LogIndent() <<"Local/Total memory allocated in all DD*: \t"	<< m_localMemoryAllocated <<  " / " << ms_totalMemoryAllocated << std::endl;
	LogIndentDecrease();
	_stream << LogIndent() <<"DataDsc_Base==============" << std::endl;
	return _stream;
}
//==================================================
/*virtual*/
void	DataDsc_Base::Allocate(void)
{
}
//==================================================
void	DataDsc_Base::Free()
{
	//if(IsAllocated())
	{
		SetDataFlag(false);
	}
}
//==================================================
void DataDsc_Base::Log_DataAlloc(unsigned int _uiSize)
{
	ms_totalMemoryAllocated	+=_uiSize;
	m_localMemoryAllocated	+=_uiSize;
}
//==================================================
void DataDsc_Base::Log_DataFree(unsigned int _uiSize)
{
	ms_totalMemoryAllocated	-=_uiSize;
	m_localMemoryAllocated	-=_uiSize;
}
//==================================================
std::string DataDsc_Base::PrintMemoryInformation(std::string text)const
{
	if(GetParent() && GetParent()->GetOption(DataContainer::DBG_IMG_MEMORY))
	{
		std::string Msg;
		if(text!="") Msg += "\n" + text;
		Msg += "\nLocal data size: \t\t\t"	+	SGE::ToCharStr(m_dataSize);
		Msg += "\nLocal memory size: \t\t\t"+	SGE::ToCharStr(m_memSize);
		Msg += "\nTotal memory allocated in all DD*: \t"+	SGE::ToCharStr(ms_totalMemoryAllocated);
		Msg += "\nLocal memory allocated in this DD: \t"+	SGE::ToCharStr(m_localMemoryAllocated);
		return Msg;
	}
	else
		return "";
}

//-------------------------------------
// General parameters manipulation
//-------------------------------------
/*virtual*/
const std::string DataDsc_Base::GetValStr()const
{
	if(GetParent())
		return GetParent()->GetValStr();
	else
		return "NOID";
}
//==================================================
DataDsc_Base* DataDsc_Base::Lock(DataDsc_Base* _lockerObj)
{
	if(m_lockedBy==NULL)
	{
		m_lockedBy = _lockerObj;
		return m_lockedBy;
	}
	else if(m_lockedBy==_lockerObj)
		return m_lockedBy;
	else
		return NULL;//can not lock obj...
}
//==================================================
bool DataDsc_Base::IsLocked()const
{
	return m_lockedBy?true:false;
}
const DataDsc_Base* DataDsc_Base::LockedBy()const
{
	return m_lockedBy;
}
//==================================================
DataDsc_Base* DataDsc_Base::UnLock(DataDsc_Base* _lockerObj)
{
	if(m_lockedBy==_lockerObj)
	{
		m_lockedBy = NULL;//release locker
		return _lockerObj;//valid the unlocking
	}
	else
		return NULL;
}
//==================================================
DataDsc_Base* DataDsc_Base::GetLockedObj()
{
	return m_lockedObj;
}
//==================================================
const DataDsc_Base* DataDsc_Base::GetLockedObj()const
{
	return m_lockedObj;
}
//==================================================
const DataDsc_Base* DataDsc_Base::SetLockedObj(DataDsc_Base* _obj)
{
	return m_lockedObj = _obj;
}
//==================================================
const std::string & DataDsc_Base::GetDscType(void)const
{
	return m_dscType;
}
//==================================================
/*virtual*/
bool	DataDsc_Base::HaveData()const
{
	return m_haveData;
}
//==================================================
/*virtual*/
void  DataDsc_Base::SetDataFlag(bool _val)
{
	CLASS_FCT_SET_NAME("SetDataFlag");
	if(m_haveData != _val)
	{
		CLASS_DEBUG(((_val)?"obtained data":"lost data"));
	}
	m_haveData = _val;
	if(IsLocked())
	{
		CLASS_DEBUG("Send data flag to locker object");
		m_lockedBy->m_haveData=_val;
	}
	if(GetLockedObj())
	{
		CLASS_DEBUG("Send data flag to locked object");
		m_lockedObj->m_haveData=_val;
	}
}
//==================================================
/*virtual*/
DataDsc_Base * DataDsc_Base::Clone(DataDsc_Base * _src, bool _datatransfer/*=true*/)
{
	TransferFormatFrom(_src);
	m_dscType = _src->m_dscType;
	SetParent(_src->GetParent());
	SetDataFlag(_src->HaveData()&&_datatransfer);
	//GetParent()->SetDataFlag<DataDsc_CvMat>(true,false);??
	return this;
}
//==================================================
/*virtual*/
DataDsc_Base * DataDsc_Base::CloneToNew(bool _datatransfer/*=true*/)
{
	DataDsc_Base * TempTex = new DataDsc_Base();
	return TempTex->Clone(this, _datatransfer);
}
//==================================================
/*virtual*/
void DataDsc_Base::SetFormat(const GLuint _pixelFormat,const GLuint _pixelType)
{
	m_glPixelFormat = _pixelFormat;
	m_glPixelType	= _pixelType;
	m_nChannels		= GetGLNbrComponent(_pixelFormat);

	ConvertPixelFormat_GLToLocal(_pixelFormat);
	ConvertPixelType_GLToLocal(_pixelType);
}
//==================================================
GLuint DataDsc_Base::GetPixelFormat()const
{
	return m_glPixelFormat;
}
//==================================================
GLuint DataDsc_Base::GetPixelType()const
{
	return m_glPixelType;
}
//==================================================
GLuint DataDsc_Base::GetnChannels()const
{
	return m_nChannels;
}
//==================================================
/*virtual*/
void	DataDsc_Base::TransferFormatFrom(const DataDsc_Base * _src)
{
	SetFormat(_src->GetPixelFormat(), _src->GetPixelType());
	m_width = _src->m_width;
	m_height = _src->m_height;
	//this transfers also the reshape parameters.....
	//reshape must be desactivated when transfering to CPU data...
	m_nChannelsBackup = _src->m_nChannelsBackup;
	m_nChannels = _src->m_nChannels;
	//==============================================
	m_dataSize	= GetDataSize();
	m_memSize	= GetMemSize();
}
//==================================================
void DataDsc_Base::SetParent(DataContainer* _parent)
{
	m_parent = _parent;
}
//==================================================
DataContainer *  DataDsc_Base::GetParent()const
{
	return m_parent;
}
//==================================================
CL_Options::OPTION_TYPE DataDsc_Base::GetParentOption(CL_Options::OPTION_TYPE _opt)const
{
	if(m_parent)
		return m_parent->GetOption(_opt);
	return 0;
}
//==================================================
/*virtual*/
void *	DataDsc_Base::GetNewParentID()const
{
	return NULL;
}
//==================================================
#if _GPUCV_DEPRECATED
DataDsc_Base::operator DataContainer*()
{
	return GetParent();
}
//==================================================
DataDsc_Base::operator DataContainer&()
{
	return *GetParent();
}
//==================================================
#endif
/*virtual*/
void	DataDsc_Base::ConvertPixelFormat_GLToLocal(const GLuint _pixelFormat)
{
	//m_glPixelFormat = _pixelFormat;
	if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
	{
		CLASS_FCT_SET_NAME("ConvertPixelFormat_GLToLocal");
		CLASS_DEBUG("No format conversion");
	}
}
//==================================================
/*virtual*/
GLuint	DataDsc_Base::ConvertPixelFormat_LocalToGL(void)
{
	if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
	{
		CLASS_FCT_SET_NAME("ConvertPixelFormat_LocalToGL");
		CLASS_DEBUG("No format conversion");
	}
	return m_glPixelFormat;
}
//==================================================
/*virtual*/
void	DataDsc_Base::ConvertPixelType_GLToLocal(const GLuint _pixelType)
{
	if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
	{
		CLASS_FCT_SET_NAME("ConvertPixelType_GLToLocal");
		CLASS_WARNING("No conversion");
	}
}
//==================================================
/*virtual*/
GLuint	DataDsc_Base::ConvertPixelType_LocalToGL()
{
	if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
	{
		CLASS_FCT_SET_NAME("ConvertPixelType_LocalToGL");
		CLASS_WARNING("No conversion");
	}
	return m_glPixelType;
}

long DataDsc_Base::GetLocalMemoryAllocated()const
{
	return m_localMemoryAllocated;
}
//static
long DataDsc_Base::GetTotalMemoryAllocated()
{
	return ms_totalMemoryAllocated;
}
//==================================================
/*GLuint	DataDsc_Base::_GetWidth()
{
return _GetWidth();
}
//==================================================
GLuint	DataDsc_Base::_GetHeight()
{
return _GetHeight();
}*/
void DataDsc_Base::_SetSize(const unsigned int width, const unsigned int heigh)
{
	if (m_width != int(width) || m_height != int(heigh))
	{//do not free image if this is the first time and size is 0
		if(m_width * m_height > 0)
		{
#if GPUCV_TEXTURE_OPTIMIZED_ALLOCATION
		_SetReloadFlag(true);//must be reloaded.
#else
		Free();
#endif
		}
		CLASS_ASSERT(width	<= _GPUCV_TEXTURE_MAX_SIZE_X, "DataContainer::SetSize()=> Texture width exceed Max size of " << _GPUCV_TEXTURE_MAX_SIZE_X);
		CLASS_ASSERT(heigh	<= _GPUCV_TEXTURE_MAX_SIZE_Y, "DataContainer::SetSize()=> Texture heigh exceed Max size of " << _GPUCV_TEXTURE_MAX_SIZE_Y);

		TextSize<GLsizei>::_SetSize(width,heigh);
	#if !_GPUCV_USE_DATA_DSC
		_UpdateTextCoord();
	#endif
	}
	else
		return;
}
//==================================================
void DataDsc_Base::SetReshape(int new_cn, int new_rows/*=0*/)
{
	CLASS_FCT_SET_NAME("SetReshape");
	m_nChannelsBackup = m_nChannels;
	m_nChannels = new_cn;
	if(new_rows==0)
	{
		if(IS_MULTIPLE_OF(m_width  * m_nChannelsBackup, new_cn))
			m_width =  m_width  * m_nChannelsBackup / new_cn;
		else
		{
			CLASS_WARNING("Reshape don't fit => doing nothing!");
		}
	}
}
//==================================================
void DataDsc_Base::UnsetReshape()
{
	if(m_nChannelsBackup!=0)
	{
		m_width = m_width / m_nChannelsBackup * m_nChannels;
		m_nChannels = m_nChannelsBackup;
		m_nChannelsBackup = 0;
	}
}
//==================================================
GLuint	DataDsc_Base::_GetNChannels()const
{
	return m_nChannels;
}
//==================================================
void DataDsc_Base::_SetNChannels(GLuint _channels)
{
	m_nChannels = _channels;
}
//==================================================
GLuint	DataDsc_Base::GetDataSize(void)const
{
	return _GetWidth()*_GetHeight()* _GetNChannels();
}
//==================================================
GLuint	DataDsc_Base::GetMemSize(void)const
{
	return _GetWidth()*_GetHeight()*GetGLTypeSize(GetPixelType())*GetnChannels();
}
//==================================================

std::ostringstream & operator << (std::ostringstream & _stream, const DataDsc_Base & TexDsc)
{
	return TexDsc.operator << (_stream);
}

#if _GPUCV_PROFILE
//==================================================
long int DataDsc_Base::CopyToEstimation(DataDsc_Base* _destination, bool _datatransfer/*=true*/, SG_TRC::CL_TRACE_BASE_PARAMS*_params/*=NULL*/)
{
	std::string FctName = GetClassName()+"-"+_destination->GetClassName();
	SG_TRC::CL_FUNCT_TRC<SG_TRC::SG_TRC_Default_Trc_Type> *CurFct = NULL;
	SG_TRC::CL_TRACE_BASE_PARAMS *_PROFILE_PARAMS = NULL;
	if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_PROFILING_TRANSFER
		&& m_TransferClassTracer!=NULL))
	{
		CurFct=m_TransferClassTracer->AddFunct(FctName);
		CurFct->Process();
		//if(CurFct->TracerRecorded->GetCount()<2)
		//	return -1;
		SG_TRC::CL_TRACER_RECORD<CL_Profiler::TracerType> * Record = NULL;
		if(_params)
			Record = CurFct->TracerRecorded->Find(_params->GetParamsValue());
		else
			Record = CurFct->TracerRecorded->Find("");

		if (Record==NULL)
			return -1;
		else if(Record->GetStats().GetTotalRecords() > 1)
		{//calculate and return avg, or deviation?
			CL_Profiler::TracerType & TotalTime = *Record->GetStats().GetTotalTime();
			CL_Profiler::TracerType & MaxTime	= *Record->GetStats().GetMaxTime();
			CL_Profiler::TracerType AvgTime = (TotalTime-MaxTime)/(Record->GetStats().GetTotalRecords()-1);
			return AvgTime.GetValue();
		}
	}
	return -1;
}
//==================================================
long int DataDsc_Base::CopyFromEstimation(DataDsc_Base* _source, bool _datatransfer/*=true*/, SG_TRC::CL_TRACE_BASE_PARAMS*_params/*=NULL*/)
{
	return _source->CopyToEstimation(this, _datatransfer, _params);
}
#endif
//==================================================
void  DataDsc_Base::Flush(void)
{
	//do nothing by default...
}
//==================================================

}//namespace GCV
