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
#include <GPUCV/DataDsc_CvMat.h>
#include <GPUCV/cxtypesg.h>
#include <GPUCVTexture/DataContainer.h>


namespace GCV{
//==================================================
DataDsc_CvMat::DataDsc_CvMat()
: DataDsc_CPU("DataDsc_CvMat")
,m_CvArr(NULL)
,m_CvMat(NULL)
{

}
//==================================================
DataDsc_CvMat::~DataDsc_CvMat(void)
{
	Free();
}
//==================================================
/*virtual*/
DataDsc_Base * DataDsc_CvMat::CloneToNew(bool _datatransfer/*=true*/)
{
	DataDsc_CvMat * TempTex = new DataDsc_CvMat();
	return TempTex->Clone(this,_datatransfer);
}
//==================================================
/*virtual*/
DataDsc_CvMat*	DataDsc_CvMat::Clone(DataDsc_CvMat* _source, bool _datatransfer/*=true*/)
{
	DataDsc_Base::Clone(_source, _datatransfer);
	DataDsc_CvMat* newDD = CloneCvMat(_source->_GetCvMat(), _datatransfer);
	//CvMat are used as DataContainerID, so we need to update it.
	//we can't do ti here cause the object don't know its new parent yet
	//this is done in DataContainer::_CopyAllDataDsc
	//newDD->GetParent()->SetID(newDD->_GetCvMat());
	return newDD;
}
//==================================================
/*virtual*/
bool DataDsc_CvMat::CopyTo(DataDsc_Base* _destination, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_destination->GetDscType(),"CopyTo");
	TEXTUREDESC_COPY_START(_destination,_datatransfer);

	//copy to CvMat => clone
	DataDsc_CvMat * TempCvMat = dynamic_cast<DataDsc_CvMat *>(_destination);
	if(TempCvMat)
	{
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CVMAT", "DD_CVMAT", this, TempCvMat,_datatransfer);
		return TempCvMat->Clone(this, _datatransfer)?true:false;
	}
	//======================

	//copy to Cpu =>
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_destination);
	if(TempCPU)
	{
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CVMAT", "DD_CPU", this, TempCPU,_datatransfer);
		UnsetReshape();
		TempCPU->TransferFormatFrom(this);
		TempCPU->Allocate();
		PIXEL_STORAGE_TYPE * tmpdata = *TempCPU->_GetPixelsData();
		memcpy(tmpdata, _GetPixelsData(), GetDataSize());
		return true;
	}
	//======================
	return false;
}
//==================================================
/*virtual*/
bool DataDsc_CvMat::CopyFrom(DataDsc_Base* _source, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_source->GetDscType(),"CopyFrom");
	TEXTUREDESC_COPY_START(_source,_datatransfer);

	//copy from Iplimage => clone
	DataDsc_CvMat * TempCvMat = dynamic_cast<DataDsc_CvMat *>(_source);
	if(TempCvMat)
	{
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CVMAT", "DD_CVMAT", TempCvMat, this,_datatransfer);
		return Clone(TempCvMat, _datatransfer)?true:false;;
	}
	//======================

	//copy from Cpu =>
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_source);
	if(TempCPU)
	{
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CPU", "DD_CVMAT", TempCPU, this,_datatransfer);
		UnsetReshape();
		TransferFormatFrom(TempCPU);
		Allocate();
		PIXEL_STORAGE_TYPE * tmpdata = _GetPixelsData();
		memcpy(tmpdata, TempCPU->_GetPixelsData(), GetDataSize());
		return true;
	}
	//======================
	return false;
}
//==================================================
/*virtual*/
void DataDsc_CvMat::Allocate(void)
{
	CLASS_FCT_SET_NAME("Allocate");
	if(!IsAllocated())
	{
		DataDsc_Base::Allocate();
		CLASS_DEBUG("");
		CLASS_FCT_PROF_CREATE_START();
		CLASS_ASSERT(m_CvMat, "DataDsc_CvMat::Allocate()=> not CvMat, can't allocate data");
		cvCreateData(m_CvMat);
		Log_DataAlloc(m_memSize);
		CLASS_ASSERT(m_CvMat->data.ptr, "DataDsc_CvMat::Allocate()=> Allocation failed");
		_SetPixelsData((PIXEL_STORAGE_TYPE **)&m_CvMat->data.ptr);
	}
}
//==================================================
/*virtual*/
void DataDsc_CvMat::Free()
{
//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("Free");
	if(IsAllocated())
	{
		//no Bench and log in Destructor//		CLASS_DEBUG("");
		//no Bench and log in Destructor//CLASS_FCT_PROF_CREATE_START();
#if 0 //????? not working..????
		if(m_CvMat->data.ptr!=NULL)
			cvReleaseMatND Data(m_CvMat);
#else
		if(m_CvMat->data.ptr!=NULL)
			m_CvMat->data.ptr=NULL;
#endif
		_SetPixelsData((PIXEL_STORAGE_TYPE**)&m_CvMat->data.ptr);
		Log_DataFree(m_memSize);
		DataDsc_Base::Free();
	}
	if(m_CvMat)
		_SetPixelsData((PIXEL_STORAGE_TYPE**)&m_CvMat->data.ptr);
}
//==================================================
/*virtual*/
bool DataDsc_CvMat::IsAllocated()const
{
	if(m_pixels)
		return (*m_pixels)? true:false;
	return false;
}
//==================================================
/*virtual*/
void *	DataDsc_CvMat::GetNewParentID()const
{
	return _GetCvMat();
}

//==================================================
//==================================================
//Local functions
//==================================================
//==================================================
void DataDsc_CvMat::_SetCvMat(CvMat ** _mat)
{
	CLASS_FCT_SET_NAME("_SetCvMat");

	m_CvArr = (CvArr**)_mat;
	m_CvMat = *_mat;

	CLASS_DEBUG("");

	if(m_CvMat->data.ptr)
		SetDataFlag(true);
	else
		SetDataFlag(false);

	if(GetParent())
		GetParent()->RemoveAllDataDscExcept<DataDsc_CvMat>();
	_SetSize(m_CvMat->width, m_CvMat->height);
	_SetPixelsData((PIXEL_STORAGE_TYPE**) &m_CvMat->data.ptr);
	//here, we do it manually without calling SetFormat
	m_glPixelType = ConvertPixelType_LocalToGL();
	m_glPixelFormat = ConvertPixelFormat_LocalToGL();
	m_nChannels = GetGLNbrComponent(m_glPixelFormat);

	//check the number of channels...
#if 0//must be tested further...
	GLuint  NewWidth=0;
	GLenum	NewFormat=0;
	GLenum	NewType=0;
	GLenum	NewChannel=0;
	switch(m_nChannels)
	{
	case 1: if (IS_MULTIPLE_OF(m_CvMat->width, 4))
			{
				NewWidth = m_CvMat->width/4;
				NewFormat = GL_BGRA;
			}
			else if (IS_MULTIPLE_OF(m_CvMat->width, 3))
			{
				NewWidth = m_CvMat->width/3;
				NewFormat = GL_BGR;// BGRA;
				//we force the number of channel here
				//cause some hardware does not support CL_LUMINANCE_APLHA
				NewChannel = 3;
			}
			else if (IS_MULTIPLE_OF(m_CvMat->width, 2))
			{
				GPUCV_PERF_WARNING("Matrix processing using multiple of 2 is not working yet due to some GL_LUMINANCE_ALPHA format compatibility. Downgrading to multiple of one that could reduce performances.");
#if 0
				NewWidth = m_CvMat->width/2;
				NewFormat = GL_BGR;// BGRA;
				//cause some hardware does not support CL_LUMINANCE_APLHA
				NewChannel = 2;
#endif
			}
			break;
	case 2:
		if (IS_MULTIPLE_OF(m_CvMat->width, 4))
		{
			NewWidth = m_CvMat->width/2;
			NewFormat = GL_BGRA;
		}
		break;
	case 3://nothing to do
	case 4://nothing to do

		break;
	}

#if _GPUCV_SUPPORT_CVMAT
	if(NewWidth)
	{//set new properties
		_SetSize(NewWidth,  m_CvMat->height);
		if(NewChannel)
			_SetNChannels(NewChannel);//_ForceNChannels(NewChannel);
		SetFormat(NewFormat,m_glPixelType);
	}
#endif
#endif
	if(m_CvMat->data.ptr)
	{
		//update image data size and mem here, cause data have already been allocated.
		m_dataSize	= GetDataSize();
		m_memSize	= GetMemSize();
		ms_totalMemoryAllocated	+=m_memSize;
		m_localMemoryAllocated	+=m_memSize;
		CLASS_MEMORY_LOG("new TT data allocated:" << DataDsc_Base::ms_totalMemoryAllocated << "\tLocal:"<< m_localMemoryAllocated);
	}
}
//==================================================
CvMat* DataDsc_CvMat::_GetCvMat()const
{
	return m_CvMat;
}
//==================================================
CvMat** DataDsc_CvMat::_GetCvMatPtr()const
{
	return (CvMat**)m_CvArr;
}
//==================================================
DataDsc_CvMat*	DataDsc_CvMat::CloneCvMat(const CvMat* _source, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME("CloneCvMat");
	CLASS_DEBUG("");
	CV_FUNCNAME("DataDsc_CvMat::CloneCvMat()");
	CLASS_ASSERT(_source, "DataDsc_CvMat::CloneCvMat(CvMat)=>Empty source!");
	__BEGIN__

		if(_source == m_CvMat)
		{
			CLASS_WARNING("Cloning A with A..??");
			return NULL;
		}
#if _GPUCV_SUPPORT_CVMAT
		CvMat * dst=NULL;

		if(_datatransfer)
		{
			cvReleaseMat(&m_CvMat);
			m_CvMat = cvCloneMat(_source);
			m_CvArr = (void**)&m_CvMat;
			//GetParent()->SetDataFlag<DataDsc_CvMat>(true,false);
		}
		else
		{
			if( !CV_IS_MAT_HDR( _source ))
				CV_ERROR( CV_StsBadArg, "Bad CvMat header" );
			if(m_CvMat==NULL)
			{
				CV_CALL( dst = cvCreateMatHeader( _source->rows, _source->cols, _source->type ));
			}
			else
			{
				dst = m_CvMat;
			}

			if( _source->data.ptr )
			{
				CV_CALL( cvCreateData( dst ));
				CV_CALL( cvCopy( _source, dst ));
			}
			//==========================
			m_CvMat = dst;
			m_CvArr = (void**)&m_CvMat;
		}
#endif
		__END__

			return this;
}
//==================================================
/*virtual*/
void	DataDsc_CvMat::ConvertPixelFormat_GLToLocal(const GLuint _pixelFormat)
{
	if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
	{
		CLASS_FCT_SET_NAME("ConvertPixelFormat_GLToLocal");
		CLASS_DEBUG("No format conversion");
	}
	//if(m_CvMat)
	//	m_CvMat->nChannels = cvgConvertGLTexFormatToCV(_pixelFormat, m_CvMat->channelSeq);
}
//==================================================
/*virtual*/
GLuint	DataDsc_CvMat::ConvertPixelFormat_LocalToGL(void)
{
	GLuint InternalFormat=0;
	GLuint Format=0;
	GLuint PixType=0;

	if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
	{
		CLASS_FCT_SET_NAME("ConvertPixelFormat_LocalToGL");
		if(m_CvMat)
		{	CLASS_DEBUG("Convert from " << ".."<< "to "<< GetStrGLTextureFormat(Format));}
		else
		{	CLASS_DEBUG("No conversion");}
	}

	if(m_CvMat)
	{
		cvgConvertCVMatrixFormatToGL(m_CvMat, InternalFormat, Format, PixType);
		return Format;
	}
	else
	{
		return m_glPixelFormat;
	}
}
//==================================================
/*virtual*/
void	DataDsc_CvMat::ConvertPixelType_GLToLocal(const GLuint _pixelType)
{
#if _GPUCV_SUPPORT_CVMAT
	if(m_CvMat)
	{
		m_CvMat->type = cvgConvertGLFormattoCVMatrix(GetPixelType(), GetnChannels());
	}
	else if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
	{
		CLASS_FCT_SET_NAME("ConvertPixelFormat_LocalToGL");
		CLASS_WARNING("No conversion");
	}
	//m_glPixelType = _pixelType;
	//if(m_CvMat)
	//	m_CvMat->type = cvgConvertGLPixTypeToCV(_pixelType);
#endif
}
//==================================================
/*virtual*/
GLuint	DataDsc_CvMat::ConvertPixelType_LocalToGL(void)
{
#if _GPUCV_SUPPORT_CVMAT
	GLuint InternalFormat=0;
	GLuint Format=0;
	GLuint PixType=0;
	if(m_CvMat)
	{
		cvgConvertCVMatrixFormatToGL(m_CvMat, InternalFormat, Format, PixType);
		return PixType;
		//cvgConvertCVPixTypeToGL(m_CvMat);
	}
	else
#endif
		return m_glPixelType;
}
//==================================================
std::ostringstream & DataDsc_CvMat::operator << (std::ostringstream & _stream)const
{
	_stream << std::endl;
	_stream << std::endl;
	DataDsc_CPU::operator <<(_stream);
	_stream << "DataDsc_CvMat==============" << std::endl;
	LogIndentIncrease();
	_stream << "m_CvMat: \t"			<<	m_CvMat << std::endl;
	_stream << "m_CvMat->type: \t"		<< m_CvMat->type << std::endl;
	_stream << "m_CvMat->step: \t"		<< m_CvMat->step << std::endl;
	_stream << "m_CvMat->rows: \t"		<< m_CvMat->rows << std::endl;
	_stream << "m_CvMat->cols: \t"		<< m_CvMat->cols << std::endl;
	LogIndentDecrease();
	_stream << "DataDsc_CvMat==============" << std::endl;
	return _stream;
}
}//namespace GCV
