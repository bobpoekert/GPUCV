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
#include <GPUCV/DataDsc_IplImage.h>
#include <GPUCVTexture/DataContainer.h>
#include <GPUCV/toolscvg.h>
#include <GPUCV/cxtypesg.h>
//#include <highguig/highguig.h>
//#include <cxcoreg/cxtypesg.h>

namespace GCV{

//==================================================
DataDsc_IplImage::DataDsc_IplImage()
: DataDsc_CPU("DataDsc_IplImage")
,m_CvArr(NULL)
,m_IplImage(NULL)
{

}
//==================================================
DataDsc_IplImage::~DataDsc_IplImage(void)
{
	//Free();
	if(m_IplImage)
	{
		DataDsc_Base::Free();
		//we do an opencv free here cause the DataDsc_IplImage::free() will
		//only release the data and not the images...
		cvReleaseImage(&m_IplImage);
		Log_DataFree(m_memSize);
		m_pixels	=NULL;
		m_IplImage	=NULL;
		m_CvArr		=NULL;
	}
}
//==================================================
/*virtual*/
DataDsc_Base * DataDsc_IplImage::CloneToNew(bool _datatransfer/*=true*/)
{
	DataDsc_IplImage * TempTex = new DataDsc_IplImage();
	return TempTex->Clone(this,_datatransfer);
}
//==================================================
/*virtual*/
DataDsc_IplImage*	DataDsc_IplImage::Clone(DataDsc_IplImage* _source, bool _datatransfer/*=true*/)
{
	DataDsc_Base::Clone(_source, _datatransfer);
	DataDsc_IplImage* newDD = CloneIpl(_source->_GetIplImage(), _datatransfer);
	//IplImage are used as DataContainerID, so we need to update it.
	//we can't do ti here cause the object don't know its new parent yet
	//this is done in DataContainer::_CopyAllDataDsc
	return newDD;
}
//==================================================
/*virtual*/
bool DataDsc_IplImage::CopyTo(DataDsc_Base* _destination, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_destination->GetDscType(),"CopyTo");
	TEXTUREDESC_COPY_START(_destination,_datatransfer);

	//copy to IplImage => clone
	DataDsc_IplImage * TempIpl = dynamic_cast<DataDsc_IplImage *>(_destination);
	if(TempIpl)
	{
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_IPL", "DD_IPL", this, TempIpl,_datatransfer);
		return TempIpl->Clone(this, _datatransfer)?true:false;;
	}
	//======================

	//copy to CPU =>
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_destination);
	if(TempCPU)
	{
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_IPL", "DD_CPU", this, TempCPU,_datatransfer);
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
bool DataDsc_IplImage::CopyFrom(DataDsc_Base* _source, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_source->GetDscType(),"CopyFrom");
	TEXTUREDESC_COPY_START(_source,_datatransfer);

	//copy from IplImage => clone
	DataDsc_IplImage * TempIpl = dynamic_cast<DataDsc_IplImage *>(_source);
	if(TempIpl)
	{
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_IPL", "DD_IPL", TempIpl, this, _datatransfer);
		return Clone(TempIpl, _datatransfer)?true:false;;
	}
	//======================

	//copy from CPU =>
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_source);
	if(TempCPU)
	{
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_IPL", "DD_CPU", TempCPU, this,_datatransfer);
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

void DataDsc_IplImage::Allocate(void)
{
	CLASS_FCT_SET_NAME("Allocate");
	if(!IsAllocated())
	{
		DataDsc_Base::Allocate();
		CLASS_DEBUG("");
		CLASS_FCT_PROF_CREATE_START();
		CLASS_ASSERT(m_IplImage, "DataDsc_IplImage::Allocate()=> not IplImage, can't allocate data");
		//update image size..just in case:
		//	m_IplImage->imageSize = GetMemSize();
		cvCreateData(m_IplImage);
		Log_DataAlloc(m_memSize);
		CLASS_ASSERT(m_IplImage->imageData, "DataDsc_IplImage::Allocate()=> Allocation failed");
		_SetPixelsData((PIXEL_STORAGE_TYPE **)&m_IplImage->imageData);
	}
}

//==================================================
/*virtual*/
void DataDsc_IplImage::Free()
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("Free");
	if(IsAllocated())
	{
		//no Bench and log in Destructor//		CLASS_DEBUG("");
		if(m_memSize!=m_IplImage->imageSize)
		{
			//CLASS_ASSERT(m_memSize==m_IplImage->imageSize,"Memory size corrupted");
			GPUCV_WARNING("m_memSize!=m_IplImage->imageSize, Memory size corrupted");
		}
		/*
		if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG))
		{
		cvNamedWindow("DataDsc_IplImage::Free()", 1);
		cvShowImage("DataDsc_IplImage::Free()", m_IplImage);
		cvWaitKey(0);
		cvDestroyWindow("DataDsc_IplImage::Free()");
		}
		*/
		//CLASS_FCT_PROF_CREATE_START();
		if(m_IplImage->imageData)
		{
			cvReleaseData(m_IplImage);
			Log_DataFree(m_memSize);
			m_IplImage->imageData=NULL;
		}
		DataDsc_Base::Free();
	}
	//make sure data pointer is sync(supposed to be NULL here)
	if(m_IplImage)
		_SetPixelsData((PIXEL_STORAGE_TYPE**)&m_IplImage->imageData);
}
//==================================================
/*virtual*/
bool DataDsc_IplImage::IsAllocated()const
{
	if(m_pixels)
		return (*m_pixels)? true:false;
	return false;
}
//==================================================
/*virtual*/
void *	DataDsc_IplImage::GetNewParentID()const
{
	return _GetIplImage();
}
//==================================================
//==================================================
//Local functions
//==================================================
//==================================================
void DataDsc_IplImage::_SetIplImage(IplImage ** _IplImage)
{
	CLASS_FCT_SET_NAME("_SetIplImage");

	m_CvArr = (CvArr**)_IplImage;
	m_IplImage = *_IplImage;

	if(m_IplImage->imageData)
	{
		SetDataFlag(true);
	}
	else
	{
		SetDataFlag(false);
	}

	if(GetParent())
		GetParent()->RemoveAllDataDscExcept<DataDsc_IplImage>();
	_SetSize(m_IplImage->width, m_IplImage->height);
	_SetPixelsData((PIXEL_STORAGE_TYPE**) &m_IplImage->imageData);
	m_glPixelType = ConvertPixelType_LocalToGL();
	m_glPixelFormat = ConvertPixelFormat_LocalToGL();
	m_nChannels = GetGLNbrComponent(m_glPixelFormat);

	if(m_IplImage->imageData)
	{
		//update image data size and mem here, cause data have already been allocated.
		m_dataSize	= GetDataSize();
		m_memSize	= GetMemSize();
		Log_DataAlloc(m_memSize);
		CLASS_MEMORY_LOG("new TT data allocated:" << DataDsc_Base::ms_totalMemoryAllocated << "\tLocal:"<< m_localMemoryAllocated);
	}
	CLASS_DEBUG("");
}
//==================================================
IplImage* DataDsc_IplImage::_GetIplImage()const
{
	return m_IplImage;
}
//==================================================
IplImage** DataDsc_IplImage::_GetIplImagePtr()const
{
	return (IplImage**)m_CvArr;
}
//==================================================
DataDsc_IplImage*	DataDsc_IplImage::CloneIpl(const IplImage* _source, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME("CloneIpl");
	CLASS_DEBUG("");
	CV_FUNCNAME("DataDsc_IplImage::CloneIpl()");
	CLASS_ASSERT(_source, "DataDsc_IplImage::CloneIpl(IplImage)=>Empty source!");
	__BEGIN__

		if(_source == m_IplImage)
		{
			CLASS_WARNING("Cloning A with A..??");
			return NULL;
		}
		IplImage * dst=NULL;

		if(_datatransfer)
		{
			cvReleaseImage(&m_IplImage);
			m_IplImage = cvCloneImage(_source);
			m_CvArr = (void**)&m_IplImage;
			//GetParent()->SetDataFlag<DataDsc_IplImage>(true,false);
		}
		else
		{
			if(m_IplImage==NULL)
			{
				CV_CALL( dst = (IplImage*)cvAlloc( sizeof(*dst)));
			}
			else
			{
				dst = m_IplImage;
			}
			memcpy( dst, _source, sizeof(*_source));
			dst->imageData = dst->imageDataOrigin = 0;
			dst->roi = 0;
			if( _source->roi )
			{
				CV_CALL( dst->roi = (_IplROI*)cvAlloc( sizeof(*dst->roi)));
				memcpy( dst->roi, _source->roi, sizeof(*_source->roi));
			}
			//data is done somewhere else..???

			dst->imageData=NULL;
			_SetPixelsData((PIXEL_STORAGE_TYPE **)&dst->imageData);
			//==========================
			m_IplImage = dst;
			m_CvArr = (void**)&m_IplImage;
		}
		__END__

			return this;
}
//==================================================
/*virtual*/
void	DataDsc_IplImage::ConvertPixelFormat_GLToLocal(const GLuint _pixelFormat)
{
	//m_glPixelFormat = _pixelFormat;//_pixelFormat;
	if(m_IplImage)
	{
		m_IplImage->nChannels = cvgConvertGLTexFormatToCV(_pixelFormat, m_IplImage->channelSeq);
		if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
		{
			CLASS_FCT_SET_NAME("ConvertPixelFormat_GLToLocal");
			CLASS_DEBUG("Convert from " << GetStrGLTextureFormat(_pixelFormat) << " to " << GetStrCVTextureFormat(m_IplImage));
		}
	}
}
//==================================================
/*virtual*/
GLuint	DataDsc_IplImage::ConvertPixelFormat_LocalToGL(void)
{
	if(m_IplImage)
	{
		GLuint Format = cvgConvertCVTexFormatToGL(m_IplImage);
		if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
		{
			CLASS_FCT_SET_NAME("ConvertPixelFormat_GLToLocal");
			CLASS_DEBUG("Convert from " << GetStrCVTextureFormat(m_IplImage) << " to " << GetStrGLTextureFormat(Format));
		}
		return Format;
	}
	else
		return m_glPixelFormat;
}
//==================================================
/*virtual*/
void	DataDsc_IplImage::ConvertPixelType_GLToLocal(const GLuint _pixelType)
{
	//m_glPixelType = _pixelType;
	if(m_IplImage)
	{
		m_IplImage->depth = cvgConvertGLPixTypeToCV(_pixelType);
		if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
		{
			CLASS_FCT_SET_NAME("ConvertPixelType_GLToLocal");
			CLASS_DEBUG("Convert from " << GetStrGLTexturePixelType(_pixelType) << " to " << GetStrCVPixelType(m_IplImage));
		}
	}
}
//==================================================
/*virtual*/
GLuint	DataDsc_IplImage::ConvertPixelType_LocalToGL(void)
{
	if(m_IplImage)
	{
		GLuint Type = cvgConvertCVPixTypeToGL(m_IplImage);

		if(GetParentOption(DataContainer::DBG_IMG_FORMAT))
		{
			CLASS_FCT_SET_NAME("ConvertPixelType_LocalToGL");
			CLASS_DEBUG("Convert from " << GetStrCVPixelType(m_IplImage) << " to " << GetStrGLTexturePixelType(Type));
		}
		return Type;
	}
	else
		return m_glPixelType;
}
//==================================================
std::ostringstream & DataDsc_IplImage::operator << (std::ostringstream & _stream)const
{
	_stream << std::endl;
	_stream << std::endl;
	DataDsc_CPU::operator <<(_stream);
	_stream << "DataDsc_IplImage==============" << std::endl;
	LogIndentIncrease();
		_stream << "m_IplImage: \t"		<< m_IplImage << std::endl;
		_stream << "m_IplImage->depth: \t"		<< m_IplImage->depth << std::endl;
		_stream << "m_IplImage->nChannels: \t"		<< m_IplImage->nChannels << std::endl;
		_stream << "m_IplImage->width: \t"		<< m_IplImage->width << std::endl;
		_stream << "m_IplImage->height: \t"		<< m_IplImage->height << std::endl;
		_stream << "m_IplImage->imageSize: \t"		<< m_IplImage->imageSize << std::endl;
		_stream << "m_IplImage->channelSeq: \t"		<< m_IplImage->channelSeq << std::endl;
	LogIndentDecrease();
	_stream << "DataDsc_IplImage==============" << std::endl;
	return _stream;
}
}//namespace GCV
