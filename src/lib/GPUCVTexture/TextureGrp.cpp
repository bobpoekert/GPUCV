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
#include <GPUCVHardware/ToolsGL.h>
#include <GPUCVTexture/TextureGrp.h>
#include <GPUCVTexture/TextureRenderManager.h>
#include <GPUCVTexture/DataDsc_GLTex.h>


namespace GCV{

//=================================================
TextureGrp::TextureGrp():
CL_Profiler("TextureGrp"),
//control flags and members
m_controlFlag(TEXTGRP_NO_CONTROL),
m_textureType(0),
m_groupType(TEXTGRP_INPUT),

#if !TEXTUREGRP_USE_VECTOR
m_TextArray(NULL),
m_TextNbr(0),
m_TextMaxNbr(0),
#endif
//	m_ctrl_width(0),
//	m_ctrl_height(0),
m_ctrl_internalFormat(0),
m_ctrl_pixelType(0)
,m_mainAttachement(0)
//==========================
{
	CLASS_FCT_SET_NAME("TextureGrp");
	CLASS_FCT_PROF_CREATE_START();
	//m_TextVect = new std::vector<DataContainer*>();
	m_TextMaxNbr = MainGPU()->m_glExtension.m_MAX_TEXTURE_UNITS_ARB;
	m_TextArray = new DataContainer* [m_TextMaxNbr];
}
//=================================================
TextureGrp::~TextureGrp()
{
	//no Bench and log in Destructor//	CLASS_FCT_SET_NAME("~TextureGrp");
	//no Bench and log in Destructor//	CLASS_FCT_PROF_CREATE_START();
	//delete m_TextVect;
	if(m_TextArray)
		delete []m_TextArray;
}
//=================================================
/*virtual*/
std::ostringstream & TextureGrp::operator << (std::ostringstream & _stream)const
{
	_stream << LogIndent() <<"======================================" << std::endl;
	_stream << LogIndent() <<"TextureGrp==============" << std::endl;
	LogIndentIncrease();
	std::string Type;
	switch(m_groupType)
	{
	case TEXTGRP_INPUT:		Type = "TEXTGRP_INPUT";	break;
	case TEXTGRP_OUTPUT:	Type = "TEXTGRP_OUTPUT";break;
	}
	_stream << LogIndent() <<"Type:" << Type << std::endl;
	_stream << LogIndent() <<"m_controlFlag: \t"	<< m_controlFlag << std::endl;
	_stream << LogIndent() <<"m_textureType: \t"	<< m_textureType << std::endl;

#if !TEXTUREGRP_USE_VECTOR
	_stream << LogIndent() <<"m_TextNbr: \t"		<< m_TextNbr << std::endl;
	_stream << LogIndent() <<"m_TextMaxNbr: \t"	<< m_TextMaxNbr << std::endl;
	_stream << LogIndent() <<"m_TextArray: \t"	<< "???"<< std::endl;
#endif
	//	_stream << LogIndent() <<"m_ctrl_size: \t"			<< m_ctrl_size << std::endl;
	_stream << LogIndent() <<"m_ctrl_internalFormat: \t"	<< GetStrGLInternalTextureFormat(m_ctrl_internalFormat) << std::endl;
	_stream << LogIndent() <<"m_ctrl_pixelType: \t"		<< GetStrGLTexturePixelType(m_ctrl_pixelType) << std::endl;
	_stream << LogIndent() <<"m_mainAttachement: \t"		<< GetStrGLColorAttachment(m_mainAttachement) << std::endl;
	LogIndentDecrease();
	_stream << LogIndent() <<"TextureGrp==============" << std::endl;
	return _stream;
}
//=================================================
void TextureGrp::SetGrpType(TextGrpType type)
{
	m_groupType = type;
	if (type == TEXTGRP_INPUT)
	{
		m_TextMaxNbr = MainGPU()->m_glExtension.m_MAX_TEXTURE_UNITS_ARB;
	}
	else if (type == TEXTGRP_OUTPUT)
	{
		if(MainGPU()->m_glExtension.m_multipleRenderTarget)
			m_TextMaxNbr = MainGPU()->m_glExtension.m_multipleRenderTarget_max_nbr;
		else
			m_TextMaxNbr = 1;
	}
}
//=================================================
TextureGrp::TextGrpType TextureGrp::GetGrpType()
{
	return m_groupType;
}
//=================================================
/**\todo move extension check to TextureGrp
*/
bool TextureGrp::AddTextures(DataContainer ** _tex, unsigned int _nbr/*=1*/)
{
	CLASS_FCT_SET_NAME("AddTextures");
	CLASS_FCT_PROF_CREATE_START();
	if(m_textureType == 0 && _tex!=NULL)
	{//get texture type of the first texture
		m_textureType = _tex[0]->GetDataDsc<DataDsc_GLTex>()->_GetTexType();
	}
	for(unsigned int i = 0; i< _nbr; i++)
	{
		if(!_tex[i])
			return false;
		//check compatibilities with hardware
		if(m_groupType == TEXTGRP_INPUT)
		{
			if(m_TextMaxNbr < m_TextNbr )
			{
				GPUCV_WARNING("Current hardware can not support so many input textures. Limiting actual number of texture in current TextureGrp.");
				return false;
			}
			_tex[i]->SetOption(DataContainer::DEST_IMG, false);
		}
		else if (m_groupType == TEXTGRP_OUTPUT)
		{
			if(m_TextMaxNbr < m_TextNbr)
			{
				GPUCV_WARNING("Current hardware can not support so many multi Render target. Skiping new targets...");
				return false;
			}
			_tex[i]->SetOption(DataContainer::DEST_IMG, true);
		}
		//===================================

#if TEXTUREGRP_USE_VECTOR
		m_TextVect.push_back(_tex[i]);
#else
		m_TextArray[m_TextNbr++]=_tex[i];
#endif
	}
	return true;
}
//=================================================
//! \todo Do we destroy texture/release, optional parameters??
void TextureGrp::Clear()
{
#if TEXTUREGRP_USE_VECTOR
	m_TextVect.clear();
#else
	m_TextNbr = 0;
#endif
}
//=================================================
void TextureGrp::SetControlFlag(TextureGrp_CheckFlag _flag)
{	m_controlFlag = _flag;}
//=================================================
#if 0
void TextureGrp::BindAll()
{
	CLASS_FCT_PROF_CREATE_START("BindAll");
	_GPUCV_CLASS_GL_ERROR_TEST();
	if(m_TextNbr==1)
	{
		operator[](0)->_Bind();
	}
	else
	{
		TEXTURE_GRP_INTERNE_DO_FOR_ALL(
			TEXT,
			TEXT->_SetARBId(GL_TEXTURE0+iTexID);
		TEXT->_BindARB();
		);
	}
	//;

	_GPUCV_CLASS_GL_ERROR_TEST();
}
//=================================================
void TextureGrp::UnBindAll()
{
	CLASS_FCT_PROF_CREATE_START("UnBindAll");
	_GPUCV_CLASS_GL_ERROR_TEST();

	TEXTURE_GRP_INTERNE_DO_FOR_ALL(
		TEXT,
		TEXT->_UnBindARB();
	);
	_GPUCV_CLASS_GL_ERROR_TEST();
}

//=================================================
/*virtual*/
void TextureGrp::_SetTexParami(GLuint _ParamType, GLint _Value)
{
	TEXTURE_GRP_INTERNE_DO_FOR_ALL(
		TEXT,
		TEXT->_SetTexParami(_ParamType, _Value)
		);
}
#endif
//=================================================
#if 0
int	TextureGrp::GetCPUReturnNbr()const
{
	itObj itText;
	int i=0;
	for( itText = m_TextVect.begin(); itText != m_TextVect.end(); 1)
	{
		if((*itText)->GetCpuReturn())
			i++;
		itText++;
	}
	return i;
}
#endif
//=================================================
void TextureGrp::DrawMultiTextQuad(int x/*=0*/, int y/*=0*/)
{
	CLASS_FCT_SET_NAME("DrawMultiTextQuad");
	CLASS_FCT_PROF_CREATE_START();
	//_BENCH_GL("drawMultiTextQuad",

	_GPUCV_CLASS_GL_ERROR_TEST();

	//SG_Assert(_textures,	"drawMultiTextQuad() => No input textures.");
	//SG_Assert(_TextNbr,		"drawMultiTextQuad() => Textures nbr is 0.");


	//define main texCoord.
	SG_Assert(m_TextArray[0], "Empty texture group");

	TextCoord<> *GlobTexCoord=NULL;
	if(m_TextNbr==1)
	{
		m_TextArray[0]->GetDataDsc<DataDsc_GLTex>()->DrawFullQuad(x, y);

	}
	else
	{
		if(m_mainAttachement == DataDsc_GLTex::NO_ATTACHEMENT)
		{
			if((m_textureType == GL_TEXTURE_RECTANGLE_ARB))
			{
				if(!GlobTexCoord)
				{
					GlobTexCoord = new TextCoord<>(0.,m_ctrl_size._GetWidth(),0., m_ctrl_size._GetHeight());
				}
			}
		}

		glPushMatrix();
		glColor4f(1., 1., 1.,1.);
		glBegin(GL_QUADS);
		{
			if (m_textureType ==  GL_TEXTURE_2D || m_textureType == GL_TEXTURE_RECTANGLE_ARB)
			{
				if(GlobTexCoord)
				{
					for(int i = 0; i<4; i++)
					{
						glTexCoord2dv(GlobTexCoord->operator [](i*2));
						glVertex3fv(&GPUCVDftQuadVertexes[i*3]);
						//glVertex3f(Vertexes[i].x,Vertexes[i].y,Vertexes[i].z);
					}
				}
				else if(m_mainAttachement == DataDsc_GLTex::NO_ATTACHEMENT)
				{
					for(int i = 0; i<4; i++)
					{
						glTexCoord2dv(&GPUCVDftTextCoord[i*2]);
						glVertex3fv(&GPUCVDftQuadVertexes[i*3]);
						//glVertex3f(Vertexes[i].x,Vertexes[i].y,Vertexes[i].z);
					}
				}
				else
				{

					for(int i = 0; i<4; i++)
					{
						DataDsc_GLTex * CurTexGL =NULL;
						TEXTURE_GRP_INTERNE_DO_FOR_ALL(
							TEXT,
							CurTexGL = TEXT->GetDataDsc<DataDsc_GLTex>();
						CurTexGL->_GlMultiTexCoordARB(i);
						);
					}
				}
			}
			else
			{
				SG_Assert(0, "Unknown texture type, ID: " << m_textureType);
			}
			/*else if (m_textureType ==  GL_TEXTURE_RECTANGLE_ARB)
			{
			itObj itText;
			for(int i = 0; i<4; i++)
			{
			for( itText = m_TextVect.begin(); itText != m_TextVect.end(); itText++)
			(*itText)->GlMultiTexCoordARB(i);
			glVertex3f(Vertexes[i].x,Vertexes[i].y,Vertexes[i].z);
			}


			if (x!=0 && y!=0)
			{
			glTexCoord2f(0, 0); glVertex3f(-1, -1, -0.5f);
			glTexCoord2f(x, 0); glVertex3f( 1, -1, -0.5f);
			glTexCoord2f(x, y); glVertex3f( 1,  1, -0.5f);
			glTexCoord2f(0, y); glVertex3f(-1,  1, -0.5f);
			}
			else
			{   string Msg = "Critical : drawQuad()=> Using GL_TEXTURE_RECTANGLE_ARB and image size is equal to O\n";
			GPUCV_ERROR(Msg.data());
			}

			}
			*/
		}
		glEnd();
		glPopMatrix();
	}
	//	, "", 0,0);//_BENCH_GL
	_GPUCV_CLASS_GL_ERROR_TEST();
}
//=================================================
bool TextureGrp::CheckControlFlag()
{

#if !_GPUCV_USE_DATA_DSC
	//Control multi texture ARB mode
#if !_GPUCV_USE_DATA_DSC
	m_mainAttachement = DataContainer::NO_ATTACHEMENT;
	TEXTURE_GRP_INTERNE_DO_FOR_ALL(
		TEXT,
		if(TEXT->_GetTextCoord())//local texture coordinates defines
		{
			if(TEXT->_GetColorAttachment()!=DataContainer::NO_ATTACHEMENT)
			{
				if(m_mainAttachement==DataContainer::NO_ATTACHEMENT)
					m_mainAttachement = TEXT->m_textureAttachedID;
				//else if(m_mainAttachement == TEXT->_GetColorAttachment())
				//	SG_Assert(!m_mainAttachement == TEXT->_GetColorAttachment(), "Two textures have same color attachement");
			}
		}
		);
#endif
		//first texture
		//itObj itText = m_TextVect.begin();
		DataContainer* FirstText= operator[](0);
		SG_Assert(FirstText, "Could not retrieve first texture from texture group");
		m_ctrl_size.m_width = FirstText->m_width;
		m_ctrl_size.m_height = FirstText->m_height;
		m_ctrl_internalFormat = FirstText->m_internalFormat;
		m_ctrl_pixelType = FirstText->m_type;

		if(!GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_CHECK_IMAGE_ATTRIBS))
			return true;//we don't perform image attribs checking.

		CLASS_FCT_PROF_CREATE_START("CheckControlFlag");
		if(m_controlFlag == TEXTGRP_NO_CONTROL)//Nothing to Check
			return true;
		if(m_TextNbr <= 1)//not need to check
			return true;

		/*	//reset global flags
		m_ctrl_size._SetSize(0, 0);
		m_ctrl_internalFormat	=	0;
		m_ctrl_pixelType	= 0;

		//check width
		if(m_controlFlag & TEXTGRP_SAME_WIDTH)
		m_ctrl_size._SetWidth(FirstText->_GetWidth());

		//check height
		if(m_controlFlag & TEXTGRP_SAME_HEIGHT)
		m_ctrl_size._SetWidth(FirstText->_GetHeight());

		//check internal format
		if(m_controlFlag & TEXTGRP_SAME_INTERNAL_FORMAT)
		m_ctrl_internalFormat = FirstText->_GetInternalPixelFormat();

		//check pixel type
		if(m_controlFlag & TEXTGRP_SAME_PIXEL_TYPE)
		m_ctrl_pixelType = FirstText->_GetPixelType();

		*/
		//! \todo Proceed to texture group flag check.
		TEXTURE_GRP_INTERNE_DO_FOR_ALL(
			TEXT,
			//width
			if((m_controlFlag & TEXTGRP_SAME_WIDTH) && (m_ctrl_size.m_width !=TEXT->m_width))
			{
				GPUCV_WARNING("TextureGrp::CheckControlFlag() => Width does not match");
			}
			//height
			if((m_controlFlag & TEXTGRP_SAME_HEIGHT) && (m_ctrl_size.m_height !=TEXT->m_height))
			{
				GPUCV_WARNING("TextureGrp::CheckControlFlag() => Height does not match");
			}
			//internal Format
			if((m_controlFlag & TEXTGRP_SAME_INTERNAL_FORMAT) && (m_ctrl_internalFormat != TEXT->m_internalFormat))
			{
				GPUCV_WARNING("TextureGrp::CheckControlFlag() => Internal Format does not match");
			}

			//internal Format
			if((m_controlFlag & TEXTGRP_SAME_PIXEL_TYPE) && (m_ctrl_pixelType != TEXT->m_type))
			{
				GPUCV_WARNING("TextureGrp::CheckControlFlag() => PixelType does not match");
			}
			);
#endif
			return true;
}
//=================================================
//=================================================
/*virtual*/
#if _GPUCV_DEPRECATED
bool TextureGrp::ProcessCPUReturn(bool _dataTransfer)
{
	CLASS_FCT_PROF_CREATE_START("ProcessCPUReturn");
	int errorNbr=0;
	int RenderBufferType = RenderBufferManager()->GetType();
	bool CpuReturnFlag = 0;
	TEXTURE_GRP_INTERNE_DO_FOR_ALL(
		TEXT,
		CpuReturnFlag = TEXT->GetOption(DataContainer::CPU_RETURN) != 0;

	if(RenderBufferType == TextureRenderBuffer::RENDER_OBJ_PBUFF)
	{
		//we force the dst image to be on FRAME_BUFFER, so the image will be read back to GPU
		//cause PBUFFER need the texture to be written using FrameBuffer content
		TEXT->_ForceLocation(DataContainer::LOC_FRAME_BUFFER);
		if(!CpuReturnFlag)
		{//NO CPU RETURN on PBUFFER
			if(!TEXT->SetLocation(DataContainer::LOC_GPU, _dataTransfer))
				errorNbr++;
		}else
		{//CPU RETURN on PBUFFER
			if(!TEXT->SetLocation(DataContainer::LOC_CPU, _dataTransfer))
				errorNbr++;
		}
	}
	else if (CpuReturnFlag)
	{// CPU RETURN move texture to CPU
		if(!TEXT->SetLocation(DataContainer::LOC_CPU, _dataTransfer))
			errorNbr++;
	}
	);

	if (errorNbr)
		return false;
	else
		return true;
}
#endif
//=================================================
bool TextureGrp::IsTextureInGroup(GPUCV_TEXT_TYPE _tex)
{
	TEXTURE_GRP_INTERNE_DO_FOR_ALL(Tex,
		if(Tex == _tex)
			return true;);
	return false;
}
//=================================================
/*virtual*/
bool TextureGrp::ReplaceTexture(GPUCV_TEXT_TYPE _old, GPUCV_TEXT_TYPE _new)
{
	SG_Assert(_old, "Empty source texture");
	SG_Assert(_new, "Empty new texture");

	TEXTURE_GRP_INTERNE_DO_FOR_ALL(iTex,
		if(iTex == _old)
		{
			iTex = _new;
			return true;
		}
		);
		return false;
}
//=================================================
/*virtual*/
void TextureGrp::PushOptions()
{
	TEXTURE_GRP_INTERNE_DO_FOR_ALL(Tex, Tex->PushOptions());
}
//=================================================
/*virtual*/
void TextureGrp::PopOptions()
{
	TEXTURE_GRP_INTERNE_DO_FOR_ALL(Tex, Tex->PopOptions());
}
//=================================================
/*virtual*/
void TextureGrp::SetOption(CL_Options::OPTION_TYPE _opt, bool val)
{
	TEXTURE_GRP_INTERNE_DO_FOR_ALL(Tex, Tex->SetOption(_opt, val));
}
//=================================================
/*virtual*/
CL_Options::OPTION_TYPE TextureGrp::GetOption(CL_Options::OPTION_TYPE _opt)const
{
	GPUCV_ERROR("GetOption() can not be used for a TextureGrp");
	return 0;
}
//=================================================
/*virtual*/
void TextureGrp::PushSetOptions(CL_Options::OPTION_TYPE _opt, bool val)
{
	TEXTURE_GRP_INTERNE_DO_FOR_ALL(Tex, Tex->PushSetOptions(_opt, val));
}
//===========================================================
void ViewerDisplayTextureGroup(TextureGrp * _grp, SGE::CL_Vector2Df & _pos, SGE::CL_Vector2Df & _size)
{
	//	glDrawBuffer(GL_BACK);
	//    InitGLView(0,Windows_size[0], 0, Windows_size[1]);
	//	glDisable(GL_LIGHTING);

	DataContainer * CurTex = NULL;
	size_t TextureNbr = _grp->GetTextNbr();
	if(TextureNbr==0)
		return;
	float Pos2D[2]={
					(float)0.,
					(TextureNbr==1)? (float)0.: (float)(-1*2./TextureNbr + 1./TextureNbr)
					};

	float Scale2D[2] = {
		(float)1./TextureNbr,
		(float)1./TextureNbr
	};

	glPushMatrix();
	glTranslatef(_pos.x, _pos.y, 0.);
	glScalef(_size.x, _size.y, 0.);

	DataDsc_GLTex * texGL = NULL;
	TEXTURE_GRP_EXTERNE_DO_FOR_ALL(_grp,iText,
		texGL = iText->GetDataDsc<DataDsc_GLTex>();
	texGL->DrawFullQuad(Pos2D[0], Pos2D[1], Scale2D[0], Scale2D[1], (float)GetGpuCVSettings()->GetWindowSize()[0], (float)GetGpuCVSettings()->GetWindowSize()[1]);
	Pos2D[1] += (float)1./TextureNbr*2;
	);
	glPopMatrix();
}
//============================================
std::ostringstream & operator << (std::ostringstream & _stream, const TextureGrp& TexDsc)
{
	return TexDsc.operator << (_stream);
}
//============================================
}//namespace GCV

