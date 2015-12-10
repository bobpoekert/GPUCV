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
#include "GPUCV/cvgArr.h"
#include <GPUCV/toolscvg.h>
#include <GPUCV/DataDsc_IplImage.h>

namespace GCV{

//=======================================================
CvgArr :: CvgArr()
:DataContainer()
,m_texIplImage(NULL)
#if _GPUCV_SUPPORT_CVMAT
,m_texCvMat(NULL)
#endif
{
}
//=======================================================
CvgArr :: CvgArr(CvgArr::TInput ** _origin)
:DataContainer((CvgArr::TInput *)*_origin)
,m_texIplImage(NULL)
#if _GPUCV_SUPPORT_CVMAT
,m_texCvMat(NULL)
#endif
{
	SetCvArr(_origin);
}
//=======================================================
#if _GPUCV_SUPPORT_CVMAT
CvgArr :: CvgArr(CvMat	** _origin)
:DataContainer((CvgArr::TInput *)*_origin)
,m_texIplImage(NULL)
,m_texCvMat(NULL)
{
	SetCvArr((TInput **)_origin);
	//SetCvMat(_origin);
	//SetIplImage(NULL);
}
#endif
//=======================================================
CvgArr :: CvgArr(IplImage ** _origin)
:DataContainer((CvgArr::TInput *)*_origin)
,m_texIplImage(NULL)
#if _GPUCV_SUPPORT_CVMAT
,m_texCvMat(NULL)
#endif
{
	SetCvArr((TInput **)_origin);
	//SetCvMat(NULL);
	//SetIplImage(_origin);
}
//=======================================================
CvgArr :: CvgArr( CvgArr & _copy)
:DataContainer(_copy, true)//we ask to copy only the descriptor the last descriptor that have data
,m_texIplImage(NULL)
#if _GPUCV_SUPPORT_CVMAT
,m_texCvMat(NULL)
#endif
{//copy constructor

	//Copy all locations done with DataContainer(_copy)

	//here we just get ID of the cloned IplImage or Cvmat
	//if the DataContainer copy constroctor is called with (_copy, true), we might check that the IplImage
	//has been copied properly.
	int ID=-1;
	if(_copy.m_texIplImage)
	{
		ID=FindDataDscID<DataDsc_IplImage>();
		if(ID==-1)
		{//manual copy required
			m_texIplImage = (DataDsc_IplImage*)this->AddNewDataDsc(_copy.GetDataDsc<DataDsc_IplImage>()->CloneToNew(false));
		}
		else
			m_texIplImage = GetDataDsc<DataDsc_IplImage>();
	}
	else if(_copy.m_texCvMat)
	{//check for matrix
		ID=FindDataDscID<DataDsc_CvMat>();
		if(ID==-1)
		{//manual copy required
			m_texCvMat = (DataDsc_CvMat*)this->AddNewDataDsc(_copy.GetDataDsc<DataDsc_CvMat>()->CloneToNew(false));
		}
		else
			m_texCvMat = GetDataDsc<DataDsc_CvMat>();
	}
}
//=======================================================
CvgArr :: ~CvgArr()
{
	//	freeTexture();
	//Yann.A 20/12/2005 => this line cause error
	//if (m_IplImage_sav) m_IplImage_sav->alphaChannel = 0;

	// Free CPU memory
	//if (m_IplImage && m_IplImage->imageData) cvReleaseImage(&m_IplImage);
}

//==========================================
void CvgArr :: SetCvMat(CvMat  **_mat)
{
	SetCvArr((TInput **)_mat);
}
//==========================================
void CvgArr :: SetIplImage(IplImage **_img)
{
	SetCvArr((TInput **)_img);
}
//==========================================
IplImage	* CvgArr ::GetIplImage()const
{
	return (m_texIplImage)?
		m_texIplImage->_GetIplImage():
		NULL;
}
//==========================================
CvMat		* CvgArr::GetCvMat()const
{
	return (m_texCvMat)?
		m_texCvMat->_GetCvMat():
		NULL;
}
//==========================================
CvgArr::TInput** CvgArr ::GetCvArr()const
{
	if(m_texIplImage)
	{
		return (CvArr**)m_texIplImage->_GetIplImagePtr();
	}
	else if (m_texCvMat)
		return (CvArr**)m_texCvMat->_GetCvMatPtr();
	else
		return NULL;
}
//==========================================
bool CvgArr :: SetCvArr(CvArr **_arr)
{
	SG_Assert(!CV_IS_SEQ(*_arr), "Warning, GPUCV operators can't manipulate CvSeq objects yet.");

	CvArr ** ArrBackup = (CvArr **)((m_texIplImage)? m_texIplImage->_GetIplImagePtr(): NULL);

	if(_arr && * _arr)
	{//we remove all other locations
		//DataContainer::RemoveAllTextureDesc();
		//load default options... from DataContainer
		CL_Options::OPTION_TYPE LocalOption=0;
		if(GetGpuCVSettings()->GetDefaultOption("DataContainer", LocalOption))
			ForceAllOptions(LocalOption);
		//SetOption(DataContainer::UBIQUITY, false);
		//SetOption(DataContainer::CPU_RETURN, true);

		if(CV_IS_IMAGE_HDR(*_arr))
		{ // and add new location for IPLIMAGE
			//RemoveDataDsc<DataDsc_CvMat>();
			m_texIplImage = GetDataDsc<DataDsc_IplImage>();
			m_texIplImage->_SetIplImage((IplImage**)_arr);
			//_AddLocation(DataContainer::LOC_CPU);

		}
		else if(CV_IS_MAT(*_arr))
		{
			RemoveDataDsc<DataDsc_IplImage>();
			m_texCvMat = GetDataDsc<DataDsc_CvMat>();
			m_texCvMat->_SetCvMat((CvMat**)_arr);
		}
		else
		{
			SG_Assert(0, "CvgArr :: SetCvArr()=> input parameter is not an Image or Matrix");
		}
	}
	else
		return false;

	if (ArrBackup && ArrBackup!=_arr)
	{//change reference in the manager if required
		GetTextureManager()->UpdateID(*ArrBackup, *_arr);
	}
	return true;
}

//==========================================
CvgArr::TInput* CvgArr :: GetCvArrObject()
{
	if(m_texIplImage)
	{
		SetLocation<DataDsc_IplImage>();
		return m_texIplImage->_GetIplImage();
	}
	else if (m_texCvMat)
	{
		SetLocation<DataDsc_CvMat>();
		return m_texCvMat->_GetCvMat();
	}
	else
	{
		GPUCV_WARNING("CvgArr :: GetCvArrObject() => No CvArr to return!");
		return NULL;
	}
	//	GPUCV_WARNING("CvgArr :: GetIplImage() => Critical : CvgArr conversion from GPU to CPU failed.\n");
}
//=======================================================
GPUCV_TEXT_TYPE   CvgArr :: GetGpuImage()
{
	if(SetLocation<DataDsc_GLTex>())
		return this;
	else
		return NULL;
}
//=======================================================
#if _GPUCV_DEVELOP_BETA
void  CvgArr :: GrabGLBuffer(GLenum format, GLenum pixType, bool totexture)
{
	if (totexture)
	{//!!!
		SG_Assert(false, "Check this part of code???");
		/* if (!IsGPUTEXT(image_gpu))
		{

		image_gpu->SetPixelFormat(format, pixType);
		_GPUCV_CLASS_GL_ERROR_TEST();
		image_gpu->SetData(NULL);
		image_gpu->ReadFromFrameBuffer();
		}
		*/
		/*
		if (format != GL_RGB)
		{
		if (format != GL_RED)   glPixelTransferf(GL_RED_SCALE, -1.);
		if (format != GL_GREEN) glPixelTransferf(GL_GREEN_SCALE, -1.);
		if (format != GL_BLUE)  glPixelTransferf(GL_BLUE_SCALE, -1.);
		}
		*/

#if 0
		glBindTexture(GetHardProfile()->GetTextType(), image_gpu);
		glCopyTexSubImage2D(GetHardProfile()->GetTextType(), 0, 0, 0, 0, 0,m_IplImage_sav->width,m_IplImage_sav->height);
		glBindTexture(GetHardProfile()->GetTextType(), 0);
#endif
		//m_IplImage = NULL;
	}
	else
	{
		SetLocation(DataContainer::LOC_CPU, false);

		int allocsize;
		unsigned char * temp_image;
		//if (m_IplImage->origin == IPL_ORIGIN_TL)
		{
			//      allocsize = m_IplImage_sav->nChannels*m_IplImage->height*m_IplImage_sav->width;
			allocsize = m_IplImage->nChannels*m_IplImage->height*m_IplImage->width*(m_IplImage->depth/8);
			temp_image = new unsigned char[allocsize];
			if (!temp_image)
			{
				GPUCV_ERROR("Critical : memory allocation error");
				return;
			}
		}

#if 0
		cvgReadPixels(0,0,m_IplImage_sav->width,image_cpu_sav->height,
			cvgConvertCVTexFormatToGL(image_cpu_sav),
			cvgConvertCVPixTypeToGL(image_cpu_sav),
			(m_IplImage->origin == IPL_ORIGIN_TL)?temp_image:(unsigned char*)m_IplImage->imageData);
#endif

		if (m_IplImage->origin == IPL_ORIGIN_TL)// or IPL_ORIGIN_BL
		{
			FrameBufferToCpu(true);
			temp_image = (unsigned char *)_GetPixelsData();
			for(int i=0; i<allocsize; i++)
				m_IplImage->imageData[i] = temp_image[allocsize-i-1];
			//delete [] temp_image;
		}
		else
		{
			FrameBufferToCpu(true);
			m_IplImage->imageData = (char *)_GetPixelsData();
		}
		delete [] temp_image;
	}
}
#endif
//=======================================================
/*virtual*/
const std::string
CvgArr::GetValStr() const
{
	if (GetIplImage()!=NULL)
		return DataContainer::GetValStr() + " | IPL:" + SGE::ToCharStr(GetIplImage());
	else if(GetCvMat()!=NULL)
		return DataContainer::GetValStr() + " | MAT:" + SGE::ToCharStr(GetCvMat());
#if 0//not working, get a crash with calling cvgCuaCopy()????
	else if(GetCvArr()!=NULL)
	{
		if(*GetCvArr()!=NULL)
			return DataContainer::GetValStr() + " | ARR:" + SGE::ToCharStr(*GetCvArr());
		else
			return DataContainer::GetValStr() + " | ARR:EMPTY";
	}
#endif
	else
		return DataContainer::GetValStr() + " | ARR:EMPTY";
}
//=======================================================
void CvgArr::_CopyProperties(DataContainer &_src)
{
	CvgArr * _cvgSrc = dynamic_cast<CvgArr *> (&_src);
#if _GPUCV_SUPPORT_CVMAT
	SG_Assert(!(_cvgSrc->m_texCvMat!=NULL && this->m_texIplImage!=NULL),
		"CvgArr::_CopyProperties(DataContainer)=> Trying to copy a CvMat into an existing IlImage!");


	SG_Assert(!(_cvgSrc->m_texIplImage!=NULL && this->m_texCvMat!=NULL),
		"CvgArr::_CopyProperties(DataContainer)=> Trying to copy a IlImage into an existing CvMat!");
#endif

	if(_cvgSrc)
	{//we have a CvgArr

		if(_cvgSrc->GetIplImage())
		{//with IplImage
			if(_cvgSrc->_IsLocation<DataDsc_CPU>())
			{//clone...easiest way..?

			}
			//else
			{
				_CopyProperties(_cvgSrc->GetIplImage());
			}
		}
#if _GPUCV_SUPPORT_CVMAT
		if(_cvgSrc->m_texCvMat)
		{//with IplImage
			if(_cvgSrc->_IsLocation<DataDsc_CPU>())
			{//clone...easiest way..?

			}
			//else
			{
				_CopyProperties(_cvgSrc->GetCvMat());
			}

		}
#endif
	}//else nothing

	DataContainer::_CopyProperties(_src);
}
//=======================================================

void CvgArr::_CopyProperties(const IplImage *src)
{
	GetDataDsc<DataDsc_IplImage>()->CloneIpl(src, false);
}

//=======================================================
#if _GPUCV_SUPPORT_CVMAT
void CvgArr::_CopyProperties(const CvMat *src)
{
	GetDataDsc<DataDsc_CvMat>()->CloneCvMat(src, false);
}

#endif
//=======================================================
/*
void CvgArr::CopyTextCPUData(DataContainer &_src)
{
//image data

if (_src._GetPixelsData())
{
if(_GetPixelsData())
{
_RemoveLocation(LOC_CPU);//desallocate buffer
}
_CreateLocation(LOC_CPU);//reallocate buffer
//copy it
memcpy(m_pixels, _src._GetPixelsData(), m_width*m_height*GetGLPixelSize(m_format, m_type));
}
}

void CvgArr::CopyTextGPUData(DataContainer &_src)
{
if (_src._GetTextID())
{
SetRenderToTexture();
InitGLView(_src.GetWidth(), _src.GetHeight());
DrawFullQuad(_src.GetWidth(), _src.GetHeight(), *_src);
UnsetRenderToTexture();
}
}
*/
//=======================================================
}//namespace GCV
