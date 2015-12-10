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
#include "GPUCVTexture/TextureRenderManager.h"

#include <GPUCVTexture/DataDsc_GLTex.h>
#include <GPUCVTexture/DataDsc_GLBuff.h>
#include <GPUCVTexture/DataDsc_CPU.h>

namespace GCV{
//==================================================
DataDsc_GLTex::DataDsc_GLTex()
: DataDsc_GLBase()
, DataDsc_Base("DataDsc_GLTex")
,m_textureId_ARB(0)
,m_textureAttachedID(NO_ATTACHEMENT)
,m_textureType(GetHardProfile()->GetTextType())
,m_internalFormat(-1)
,m_needReload(true)
,m_texCoord(NULL)
,m_autoMipMap(false)
{
}
//==================================================
DataDsc_GLTex::~DataDsc_GLTex(void)
{
	Free();
	if(m_texCoord)
		delete m_texCoord;
}
//==================================================
/*virtual*/
DataDsc_Base * DataDsc_GLTex::CloneToNew(bool _datatransfer/*=true*/)
{
	DataDsc_GLTex * TempTex = new DataDsc_GLTex();
	return TempTex->Clone(this,_datatransfer);
}
//==================================================
/*virtual*/
DataDsc_GLTex*	DataDsc_GLTex::Clone(DataDsc_GLTex* _source, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME("Clone");
	CLASS_FCT_PROF_CREATE_START();
	CLASS_DEBUG("");
	_GPUCV_CLASS_GL_ERROR_TEST();

	DataDsc_Base::Clone(_source, _datatransfer);
	//don't copy texture ID, we create a new one.
	m_textureId_ARB = _source->m_textureId_ARB;
	m_textureAttachedID = _source->m_textureAttachedID;
	m_textureType = _source->m_textureType;
	m_internalFormat = _source->m_internalFormat;
	m_autoMipMap = _source->m_autoMipMap;
	_SetTextCoord(_source->_GetTextCoord());

	//
	Free();

	if(_datatransfer)
	{
		Allocate();
		if(1)
		{
			SetRenderToTexture();
			InitGLView();
			_source->DrawFullQuad(_source->_GetWidth(), _source->_GetHeight());
			UnsetRenderToTexture();
			_GPUCV_CLASS_GL_ERROR_TEST();
			SetDataFlag(true);
		}
		else
		{
		}
	}

	return this;
}
//==================================================
/*virtual*/
bool DataDsc_GLTex::CopyTo(DataDsc_Base* _destination, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_destination->GetDscType(),"CopyTo");
	TEXTUREDESC_COPY_START(_destination,_datatransfer);
	_GPUCV_CLASS_GL_ERROR_TEST();

	//copy to texture
	DataDsc_GLTex * TempGL = dynamic_cast<DataDsc_GLTex *>(_destination);
	if(TempGL)
	{//clone texture
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_GLTex", "DD_GLTex", this, TempGL,_datatransfer);
		return TempGL->Clone(this, _datatransfer)?true:false;
	}
	//====================

	//copy to a GL_BUFFER
	DataDsc_GLBuff * TempGLbuff = dynamic_cast<DataDsc_GLBuff *>(_destination);
	if(TempGLbuff)
	{//
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_GLTex", "DD_GLBuff", this, TempGLbuff,_datatransfer);
		CLASS_DEBUG("");
		TempGLbuff->TransferFormatFrom(this);
		TempGLbuff->_SetType(DataDsc_GLBuff::PIXEL_PACK_BUFFER);
		//buffer will be used inside OpenGL
		TempGLbuff->_SetTransferMode(DataDsc_GLBuff::DYNAMIC_COPY/*STREAM_DRAW*/);
		TempGLbuff->Allocate();
		TempGLbuff->_Bind();
		TempGLbuff->_Writedata(NULL, GetMemSize(), false);


		if(_datatransfer && this->HaveData())
		{
			_GPUCV_CLASS_GL_ERROR_TEST();
			_Bind();
			glGetTexImage(_GetTexType(), 0, GetPixelFormat(), GetPixelType(), NULL);
			_UnBind();
			_GPUCV_CLASS_GL_ERROR_TEST();
		}

		TempGLbuff->_UnBind();
		TempGLbuff->SetDataFlag(true);
		return true;
	}
	//====================

	//copy to CPU
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_destination);
	if(TempCPU)
	{//read data back to CPU
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_GLTex", "DD_CPU", this, TempCPU,_datatransfer);
		UnsetReshape();
		TempCPU->TransferFormatFrom(this);
		TempCPU->Allocate();
		if(HaveData() && _datatransfer)
		{
			CLASS_DEBUG("");
			_ReadData(TempCPU->_GetPixelsData());
			TempCPU->SetDataFlag(true);
			return true;
		}
	}
	//====================


	//don't know how to copy...must be done in another object

	return false;
}
//==================================================
/*virtual*/
bool DataDsc_GLTex::CopyFrom(DataDsc_Base* _source, bool _datatransfer/*=true*/)
{
	CLASS_FCT_SET_NAME_TPL_STR(_source->GetDscType(),"CopyFrom");
	TEXTUREDESC_COPY_START(_source,_datatransfer);
	_GPUCV_CLASS_GL_ERROR_TEST();

	//copy from texture
	DataDsc_GLTex * TempGL = dynamic_cast<DataDsc_GLTex *>(_source);
	if(TempGL)
	{//clone texture
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_GLTex", "DD_GLTex", TempGL, this,_datatransfer);
		//Allocate();
		return Clone(TempGL, _datatransfer)?true:false;
	}
	//====================

	//copy from a GL_BUFFER
	DataDsc_GLBuff * TempGLbuff = dynamic_cast<DataDsc_GLBuff *>(_source);
	if(TempGLbuff)
	{//
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_GLBuff", "DD_GLTex", TempGLbuff, this,_datatransfer);
		CLASS_DEBUG("");
		UnsetReshape();
		TransferFormatFrom(TempGLbuff);
		Allocate();
		if(_datatransfer && TempGLbuff->HaveData())
		{
			_GPUCV_CLASS_GL_ERROR_TEST();
#if 1
			//YCK test: get about	380MB/s
			TempGLbuff->_SetTransferMode(DataDsc_GLBuff::DYNAMIC_COPY/*STREAM_DRAW*/);
			this->_Bind();
			TempGLbuff->_SetType(DataDsc_GLBuff::PIXEL_UNPACK_BUFFER);
			TempGLbuff->_Bind();
			glTexImage2D(_GetTexType(), 0, _GetInternalPixelFormat(), _GetWidth(), _GetHeight(),0,GetPixelFormat(), GetPixelType(), NULL);
			TempGLbuff->_UnBind();
			this->_UnBind();
#elif 0
			//YCK test: get about	MB/s
			SetRenderToTexture();
			InitGLView();
			//glClearColor(1.,0.,1.,0.);
			//glClear(GL_COLOR_BUFFER_BIT);
			//glPolygonMode( GL_FRONT_AND_BACK, GL_FILL);
			//glDisable( GL_DEPTH_TEST);
			TempGLbuff->_Bind();
			glCopyPixels(0,0, _GetWidth(), _GetHeight(), GL_COLOR);
			//glReadPixels(0,0, _GetWidth(), _GetHeight(), GetPixelFormat(), GetPixelType(), NULL);
			TempGLbuff->_UnBind();
			_GPUCV_CLASS_GL_ERROR_TEST();
			UnsetRenderToTexture();
#else
			InitGLView();
			TempGLbuff->_Bind();
			glCopyTexImage2D(_GetTexType(),
				0,
				GetPixelFormat(),
				0, 0,
				_GetWidth(),
				_GetHeight(),
				0);
			TempGLbuff->_UnBind();

#endif
			SetDataFlag(true);
		}
		return true;
	}
	//====================

	//copy from CPU => load data into texture
	DataDsc_CPU * TempCPU = dynamic_cast<DataDsc_CPU *>(_source);
	if(TempCPU)
	{
		GPUCV_DD_TRANSFER_BENCH("transfer", "DD_CPU", "DD_GLTex", TempCPU, this,_datatransfer);
#if 0//to test
		if( GetParent())
		{
			DataDsc_GLBuff * TempGLBuffer = GetParent()->GetDataDsc<DataDsc_GLBuff>();
			if(TempGLBuffer->CopyFrom(TempCPU,_datatransfer))
				return this->CopyFrom(TempGLBuffer, _datatransfer);
			else
				return false;
		}
#else
		UnsetReshape();
		TransferFormatFrom(_source);
		//here, allocation can be avoided cause it is done in _WRITE_DATA
		Allocate();
		CLASS_DEBUG("");

		_Writedata((const PIXEL_STORAGE_TYPE **)TempCPU->_GetPixelsData(), _datatransfer);
#endif
		return true;
	}
	//====================


	//don't know how to copy...must be done in another object
	return false;
}
//==================================================
/*virtual*/
void DataDsc_GLTex::Allocate(void)
{
	CLASS_FCT_SET_NAME("Allocate");
	CLASS_FCT_PROF_CREATE_START();

#if GPUCV_TEXTURE_OPTIMIZED_ALLOCATION
	if(!IsAllocated())
#else
	if(_GetReloadFlag() || !IsAllocated())
#endif
	{
		//texture generation is done here but not texture allocation
		//we set the flag (_SetReloadFlag) to true so we will allocate it when required.
		DataDsc_Base::Allocate();
		CLASS_DEBUG("");
		_GPUCV_CLASS_GL_ERROR_TEST();
		SetGLId(TextureRecycler::GetNewTexture());
		CLASS_ASSERT(m_GLId, "DataDsc_GLTex::Allocate()=> Allocation failed");
		//if(GetParent()->GetOption(DataContainer::DEST_IMG))
		//allocate with empty data

		//else
		//	_Writedata(_data, true);
#if GPUCV_TEXTURE_OPTIMIZED_ALLOCATION
		_SetReloadFlag(true);
#else
		_Writedata(NULL, false);
		//_SetReloadFlag(false);//done in _Writedata
#endif
		_Bind();
		_SetTexParami(GL_TEXTURE_MIN_FILTER,GL_NEAREST);
		_SetTexParami(GL_TEXTURE_MAG_FILTER,GL_NEAREST);
		_SetTexParami( GL_TEXTURE_WRAP_S,GL_CLAMP);
		_SetTexParami( GL_TEXTURE_WRAP_T,GL_CLAMP);
		Log_DataAlloc(m_memSize);
		_GPUCV_CLASS_GL_ERROR_TEST();

		//GPUCVTEXTURE_DEBUG(this, GetValStr() << "::DataDsc_GLTex::_CreateLocation("<< GetTextureLocationStr(LOC_GPU)<<")");
	}
}

//==================================================
/*virtual*/
void DataDsc_GLTex::Free()
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("Free");
	//no Bench and log in Destructor//	CLASS_FCT_PROF_CREATE_START();
	_SetReloadFlag(true);//must be reloaded if we want to use it
	if(IsAllocated())
	{
		_GPUCV_CLASS_GL_ERROR_TEST();
		//no Bench and log in Destructor//		CLASS_DEBUG("");
		TextureRecycler::AddFreeTexture(GetGLId());
		SetGLId(0);
		Log_DataFree(m_memSize);
		DataDsc_Base::Free();
		_GPUCV_CLASS_GL_ERROR_TEST();
	}
}
//==================================================
/*virtual*/
bool DataDsc_GLTex::IsAllocated()const
{
#if GPUCV_TEXTURE_OPTIMIZED_ALLOCATION
	return (!_GetReloadFlag() || GetGLId());
#else
	return (_GetId());
#endif
}
//==================================================
/*virtual*/
void DataDsc_GLTex::SetFormat(const GLuint _pixelFormat,const GLuint _pixelType)
{
	//format is different, we need reload
	if((m_glPixelFormat	!= _pixelFormat)
		|| (m_glPixelType	!= _pixelType))
		_SetReloadFlag(true);


	DataDsc_Base::SetFormat(_pixelFormat,_pixelType);
	_SetInternalPixelFormat(ProcessingGPU()->ConvertGLFormatToInternalFormat(_pixelFormat, _pixelType));
}
//==================================================
//==================================================
//==================================================
//Local functions
//==================================================
//==================================================
/*virtual*/
void DataDsc_GLTex::_Bind(void)const
{
	_GPUCV_CLASS_GL_ERROR_TEST();
	if(m_needReload)//need to be reloaded
	{
		//	CLASS_WARNING(this, GetValStr() + std::string("DataDsc_GLTex::Bind() => m_needReload to 1, need to be reloaded before binding"));
	}
	if(!GetGLId())
	{
		//	CLASS_WARNING(this, GetValStr() + "DataDsc_GLTex::Bind() => no texture ID");
	}
	glEnable(m_textureType);
	glBindTexture(m_textureType, GetGLId());
	_GPUCV_CLASS_GL_ERROR_TEST();
}
//==================================================
/*virtual*/
void DataDsc_GLTex::_UnBind(void)const
{
	glBindTexture(m_textureType, 0);
}
//========================================================
void DataDsc_GLTex::_BindARB()
{
	//_GPUCVTEXTURE_GL_ERROR_TEST(this);
	if(!_GetARBId())//need to be reloaded
	{
		//CLASS_WARNING("DataDsc_GLTex::BindARB() => no ARB texture ID");
	}
	_GPUCV_CLASS_GL_ERROR_TEST();
	glActiveTextureARB(_GetARBId());
	_Bind();
	_GPUCV_CLASS_GL_ERROR_TEST();
}
//========================================================
void DataDsc_GLTex::_UnBindARB()
{
	glActiveTextureARB(_GetARBId());
	_UnBind();
}

//==================================================
void	DataDsc_GLTex::_SetInternalPixelFormat(const GLuint _internalPixelFormat)
{
	//format is different, we need reload
	if(m_internalFormat != _internalPixelFormat)
		_SetReloadFlag(true);
	m_internalFormat = _internalPixelFormat;
}
//==================================================
GLuint	DataDsc_GLTex::_GetInternalPixelFormat()const
{
	return m_internalFormat;
}
//==================================================
void DataDsc_GLTex::_SetColorAttachment(attachmentEXT _clr)
{
	m_textureAttachedID = _clr;
}
//========================================================
DataDsc_GLTex::attachmentEXT DataDsc_GLTex::_GetColorAttachment()const
{
	return m_textureAttachedID;
}
GLuint  DataDsc_GLTex::_GetARBId()const
{
	return m_textureId_ARB;
}
//========================================================
void DataDsc_GLTex::_SetARBId(GLuint _texARBID)
{
	m_textureId_ARB = _texARBID;
}
//========================================================
GLuint DataDsc_GLTex::_GetTexType()const
{
	return m_textureType;
}
//========================================================
void DataDsc_GLTex::_SetTexType(GLuint _TexType)
{
	m_textureType = _TexType;
}
//========================================================
void DataDsc_GLTex::_SetTexParami(GLuint _ParamType, GLint _Value)
{
#if _GPUCV_DEBUG_MODE
	CLASS_FCT_SET_NAME("_SetTexParami");
	//test value...
	switch(_ParamType)
	{
	case GL_TEXTURE_MIN_FILTER:
	case GL_TEXTURE_MAG_FILTER:
	case GL_TEXTURE_WRAP_S:
	case GL_TEXTURE_WRAP_T:
	case GL_TEXTURE_BORDER_COLOR:
	case GL_TEXTURE_PRIORITY:
		break;
	default:
		CLASS_DEBUG("Unknown param");
		return;
	}
#endif
	glTexParameteri(m_textureType, _ParamType , _Value);
}
//========================================================
void DataDsc_GLTex::_GetTexParami(GLuint _ParamType, GLint *_Value)
{
#if _GPUCV_DEBUG_MODE
	CLASS_FCT_SET_NAME("_GetTexParami");
	switch(_ParamType)
	{
	case GL_TEXTURE_MIN_FILTER:
	case GL_TEXTURE_MAG_FILTER:
	case GL_TEXTURE_WRAP_S:
	case GL_TEXTURE_WRAP_T:
	case GL_TEXTURE_BORDER_COLOR:
	case GL_TEXTURE_PRIORITY:
		break;
	default:
		CLASS_DEBUG("Unknown param");
		return;
	}
#endif
	glGetTexParameteriv(m_textureType, _ParamType, _Value);
}
//========================================================
void DataDsc_GLTex::_Writedata(const PIXEL_STORAGE_TYPE ** _data, const bool _dataTransfer/*=true*/)
{
	CLASS_FCT_SET_NAME("_Writedata");
	CLASS_FCT_PROF_CREATE_START();

	_GPUCV_CLASS_GL_ERROR_TEST();

	if(_dataTransfer && !_data)
	{
		CLASS_WARNING("No texture data to transfer");
	}
	//TextSize<GLsizei> * TempSize = dynamic_cast<TextSize<GLsizei> *>(GetParent());

	//check properties
	CLASS_ASSERT(m_textureType>0, "Unknown texture type");
	CLASS_ASSERT(m_internalFormat>0, "Unknown internal format");
	CLASS_ASSERT(_GetHeight()*_GetWidth()>0, "Null size");
	CLASS_ASSERT(m_glPixelFormat>0, "Unknown pixel format");
	CLASS_ASSERT(m_glPixelType>0, "Unknown pixel type");


	_GPUCV_CLASS_GL_ERROR_TEST();
	m_needReload = false;//no need anymore to reload
	_Bind();

	const PIXEL_STORAGE_TYPE * CurrData = (_dataTransfer)? *_data:NULL;
#if _GPUCV_GL_USE_MIPMAPING
	if(GetParent())
		if(GetParent()->GetOption(DataContainer::DEST_IMG))
			CurrData = NULL;//no data transfer if destination image.

	if(m_textureType == GL_TEXTURE_2D_ARRAY_EXT)
	{
		glTexImage3D(GL_TEXTURE_2D_ARRAY_EXT, 0, m_internalFormat, _GetWidth(),_GetHeight(), 2, 0, m_glPixelFormat,m_glPixelType, CurrData);
	}
	else if(m_autoMipMap)
	{//generate mipmaps
		if(MainGPU()->m_glExtension.GetSGISTextureLOD())
		{//hardware compatible : http://www.nvidia.com/dev_content/nvopenglspecs/GL_SGIS_generate_mipmap.txt
			_SetTexParami(GL_GENERATE_MIPMAP_SGIS, true);
			glTexImage2D (m_textureType, 0,
				m_internalFormat,
				_GetWidth(),_GetHeight(),
				0,
				m_glPixelFormat,m_glPixelType, CurrData);
		}
		else
		{
			if (gluBuild2DMipmaps(m_textureType,
				_GetInternalPixelFormat(),
				_GetWidth(),_GetHeight(),
				m_glPixelFormat,
				m_glPixelType,
				CurrData) !=0)//build it
			{
				GPUCV_ERROR("Error while generating MipMap");
			}
			glGenerateMipmapEXT(m_textureType);
		}
		_SetTexParami(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);//set filter settings
	}
	else
#endif
	{
		CLASS_DEBUG(*this);
		glTexImage2D (m_textureType, 0,
			m_internalFormat,
			_GetWidth(),_GetHeight(),
			0,
			m_glPixelFormat,m_glPixelType, CurrData);
	}

	_UnBind();
	m_needReload =false;//must be set again to false. because of _SetPixelsData that set it true.
	_GPUCV_CLASS_GL_ERROR_TEST();
}
//=================================================================
PIXEL_STORAGE_TYPE **
DataDsc_GLTex::_ReadData(PIXEL_STORAGE_TYPE **_Pixelsdata, GLuint _xmin/*=0*/, GLuint _xmax/*=0*/, GLuint _ymin/*=0*/, GLuint _ymax/*=0*/)
{
	CLASS_FCT_SET_NAME("_ReadData");
	CLASS_FCT_PROF_CREATE_START();
	CLASS_ASSERT(_Pixelsdata, "No memory allocated to read pixels");
	CLASS_ASSERT(*_Pixelsdata, "No memory allocated to read pixels");

	_GPUCV_CLASS_GL_ERROR_TEST();
	//GLuint xmin = 0;//_xmin;
	GLuint xmax = (_xmax)?_xmax:_GetWidth();
	//GLuint ymin = 0;//_ymin
	GLuint ymax = (_ymax)?_ymax:_GetHeight();

	//_GPUCVTEXTURE_GL_ERROR_TEST(this);

	bool RenderBufferBinded = true;//RenderBufferManager()->IsTextureBinded(this);

	if(!RenderBufferBinded)
	{
		SetRenderToTexture();// RenderBufferManager()->SetContext(this);
		// if FBO are compatible, the filter result is set in the texture as well as in the FBO
		// then we only need to read those informations
		// else we have to show this image on a quad to be able to read its data

		if (RenderBufferManager()->GetType()== TextureRenderBuffer::RENDER_OBJ_PBUFF)
		{
			//TextSize<GLsizei> textureSize(xmax, ymax);
			InitGLView();
			DrawFullQuad(xmax, ymax);
		}
	}
	else
	{
		_Bind();
	}
	CLASS_DEBUG(*this);
	glGetTexImage(_GetTexType(), 0, GetPixelFormat(), GetPixelType(), *_Pixelsdata);
	_UnBind();

	if(!RenderBufferBinded)
		UnsetRenderToTexture();//RenderBufferManager()->UnSetContext();
	//else


	//_GPUCVTEXTURE_GL_ERROR_TEST(this);
	_GPUCV_CLASS_GL_ERROR_TEST();
	return _Pixelsdata;
}
//=================================================================
void DataDsc_GLTex::_GlMultiTexCoordARB(GLuint _coordID)
{
	CLASS_ASSERT(_GetTextCoord(), "DataContainer::_GlMultiTexCoordARB()=> Trying to use multitexture coordinate without local texture coordonnate");
	_GetTextCoord()->glMultiTexCoordARB(m_textureId_ARB, _coordID);
}

//=================================================================
void DataDsc_GLTex::DrawFullQuad(const int _width, const int _height)
{
	CLASS_FCT_SET_NAME("DrawFullQuad");
	//_GPUCVTEXTURE_GL_ERROR_TEST(_Tex);
	glPushMatrix();
	//glColor4f(1., 1., 1., 1.);
	if(HaveData())
		_Bind();
	else
	{
		CLASS_ERROR("Drawing a quad with empty data?");
	}

	glBegin(GL_QUADS);
	{
		//the difference between GL_TEXTURE_2D and GL_TEXTURE_RECTANGLE_ARB has been managed into the texture coordinate system of the texture.
		if (_GetTexType() ==  GL_TEXTURE_2D || _GetTexType() ==  GL_TEXTURE_RECTANGLE_ARB || GL_TEXTURE_2D_ARRAY_EXT)//GL_TEXTURE_2D_ARRAY_EXT is experimental
		{
			TextCoordType LocalCoord;
			TextCoordType * Coord = _GetTextCoord();
			if(!Coord)
			{
				Coord = &LocalCoord;
				if(_GetTexType() ==  GL_TEXTURE_RECTANGLE_ARB)
				{
					TextCoord<double>::TplVector2D TempScale(_GetWidth(), _GetHeight());
					Coord->Scale(TempScale);
				}
			}
			Coord->glTexCoord(0); glVertex3fv(&GPUCVDftQuadVertexes[0]);
			Coord->glTexCoord(1); glVertex3fv(&GPUCVDftQuadVertexes[3]);
			Coord->glTexCoord(2); glVertex3fv(&GPUCVDftQuadVertexes[6]);
			Coord->glTexCoord(3); glVertex3fv(&GPUCVDftQuadVertexes[9]);
		}
		else
		{
			CLASS_ASSERT(0, "Unknown texture type:");
			//Print();
		}
	}
	glEnd();
	glPopMatrix();
	if(HaveData())
		_UnBind();
	//_GPUCVTEXTURE_GL_ERROR_TEST(_Tex);
}
//=================================================================
void DataDsc_GLTex::DrawFullQuad(float centerX, float centerY, float scaleX, float scaleY, float width, float height)
{
	glPushMatrix();
	glTranslatef(centerX, centerY, 0.);
	glScalef(scaleX, scaleY, 1.);
	DrawFullQuad((int) width, (int) height);
	glPopMatrix();
}
//=================================================================
void DataDsc_GLTex::InitGLView()
{
	TextCoordType * CurTexCoord = _GetTextCoord();
	//Is specific Text coordinates set?
	if(CurTexCoord)
	{

		if(m_textureType == GL_TEXTURE_2D)
			//InitGLView(CurTexCoord->operator [](0).x*Width, CurTexCoord->operator [](1).x*Width, CurTexCoord->operator [](0).y*Height, CurTexCoord->operator [](2).y*Height);
			GCV::InitGLView((int)*CurTexCoord->operator [](0)*_GetWidth(), (int)*CurTexCoord->operator [](2)*_GetWidth(), (int)*CurTexCoord->operator [](1)*_GetHeight(), (int)*CurTexCoord->operator [](5)*_GetHeight());
		else if (m_textureType == GL_TEXTURE_RECTANGLE_ARB)
			//InitGLView(CurTexCoord->operator [](0).x, CurTexCoord->operator [](1).x, CurTexCoord->operator [](0).y, CurTexCoord->operator [](2).y);
			GCV::InitGLView((int)*CurTexCoord->operator [](0), (int)*CurTexCoord->operator [](2), (int)*CurTexCoord->operator [](1),(int) *CurTexCoord->operator [](5));

	}
	else
		GCV::InitGLView(0, _GetWidth(), 0, _GetHeight());
}
//=================================================================
void		DataDsc_GLTex::SetRenderToTexture()
{
	CLASS_FCT_SET_NAME("SetRenderToTexture");
	CLASS_DEBUG("===============");\
		LogIndentIncrease();\

	if(GetParent())
	{
		GetParent()->PushSetOptions(DataContainer::UBIQUITY, false);
		GetParent()->SetOption(DataContainer::DEST_IMG, true);
	}
	if(!GetGLId())
		Allocate();

	if(_GetReloadFlag())
		_Writedata(NULL, false);
	//		_SetReloadFlag(false);
	//	}
	RenderBufferManager()->SetContext(this);
	SetDataFlag(true);
}
//=================================================
void DataDsc_GLTex::UnsetRenderToTexture()
{
	CLASS_FCT_SET_NAME("UnsetRenderToTexture");
	RenderBufferManager()->GetResult();
	RenderBufferManager()->UnSetContext();
#if _GPUCV_GL_USE_MIPMAPING
	if(m_autoMipMap)//generate LODs when we finish the render to texture.
		glGenerateMipmapEXT(m_textureType);
#endif
	if(GetParent())
		GetParent()->PopOptions();
	LogIndentDecrease();\
	CLASS_DEBUG("===============");\
}

//=================================================
DataDsc_GLTex::TextCoordType * DataDsc_GLTex::_GetTextCoord()
{
	return m_texCoord;
}
//=================================================
void DataDsc_GLTex::_SetTextCoord(const DataDsc_GLTex::TextCoordType * _texCoord)
{
	if(!_texCoord)
	{//disable text coord
		delete m_texCoord;
		m_texCoord = NULL;
		return;
	}

	//enable it
	_GenerateTextCoord();
	*m_texCoord = *_texCoord;
}
//=================================================
DataDsc_GLTex::TextCoordType * DataDsc_GLTex::_GenerateTextCoord(void)
{
	if(!m_texCoord)
	{
		m_texCoord  = new TextCoordType;
		_UpdateTextCoord();
	}
	return m_texCoord;
}
//=================================================
void	DataDsc_GLTex::_UpdateTextCoord()
{
	if(!m_texCoord)
		return;//nothing to update..
#if !_GPUCV_USE_DATA_DSC
	if(m_textureType == GL_TEXTURE_2D)
		m_texCoord->Clear();
	else if(m_textureType == GL_TEXTURE_RECTANGLE_ARB)
		m_texCoord->SetCoord(0.,m_width,0.,m_height);
#endif
}
//==================================================
void DataDsc_GLTex::_SetAutoMipMap(bool _val)
{
#if _GPUCV_GL_USE_MIPMAPING
	//Manage MipMap generation...
	if(GetGLId())
		if(MainGPU()->m_glExtension.GetSGISTextureLOD())
		{//hardware compatible : http://www.nvidia.com/dev_content/nvopenglspecs/GL_SGIS_generate_mipmap.txt
			_Bind();
			_SetTexParami(GL_GENERATE_MIPMAP_SGIS, _val);
			_UnBind();
		}
		else
		{
			if (_val)
			{
				glGenerateMipmapEXT(m_textureType);
			}
		}
		m_autoMipMap = _val;
#endif
}
//==================================================
const std::string &	DataDsc_GLTex::_GetTextureName()const
{
	return m_strTextureName;
}
//==================================================
void DataDsc_GLTex::_SetTextureName(const char* _name)
{
	m_strTextureName = _name;
}
//==================================================
std::ostringstream & DataDsc_GLTex::operator << (std::ostringstream & _stream)const
{
	_stream << std::endl;
	_stream << std::endl;
	DataDsc_GLBase::operator <<(_stream);
	_stream << LogIndent() <<"DataDsc_GLTex==============" << std::endl;
	LogIndentIncrease();
	_stream << LogIndent() <<"m_textureId_ARB: \t\t\t\t"		<< m_textureId_ARB << std::endl;
	_stream << LogIndent() <<"m_textureAttachedID: \t\t\t"	<< m_textureAttachedID << std::endl;
	_stream << LogIndent() <<"m_textureType: \t\t\t\t"			<< GetStrGLTextureType(m_textureType)<< std::endl;
	_stream << LogIndent() <<"m_internalFormat: \t\t\t\t"		<< GetStrGLInternalTextureFormat(m_internalFormat) << std::endl;
	_stream << LogIndent() <<"m_needReload: \t\t\t\t"			<< m_needReload << std::endl;
	//_Bind();
	//int iResident;
	//glGetTexParameteriv(m_textureType, GL_TEXTURE_RESIDENT, &iResident);
	//_UnBind();
	// see page 338 of openGL SuperBible

	//_stream << LogIndent() <<"Is texture resident?\t"	<< ((iResident)?"true": "false")<< std::endl;
	LogIndentDecrease();
	_stream << LogIndent() <<"DataDsc_GLTex==============" << std::endl;
	return _stream;
}
}//namespace GCV
