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
#ifndef __GPUCV_TEXTURE_GROUP_H
#define __GPUCV_TEXTURE_GROUP_H

#include <GPUCVTexture/DataContainer.h>

namespace GCV{

typedef DataContainer* GPUCV_TEXT_TYPE;

#define TEXTUREGRP_USE_VECTOR 0


/**
\brief A texture group contains a group of textures that can be used as input or output textures.
Texture groups are used to manipulate several textures of a same type (input or output), it is used to defines multi-texturing input or output for filters.
Some control flags can be used to check the all the texture from the group shares the same characteristics.
*\todo TextureGrp should be transformed into DataDscGrp and manipulate any DataDsc_*
*/
class _GPUCV_TEXTURE_EXPORT TextureGrp
	: public CL_Profiler
{
public:

	typedef GPUCV_TEXT_TYPE *					Obj;
#if TEXTUREGRP_USE_VECTOR
	typedef std::vector<Obj>::iterator itObj;
#endif
	/**
	*\brief TextureGrp type : Input/Output.
	*/
	enum TextGrpType{
		TEXTGRP_INPUT,
		TEXTGRP_OUTPUT
	};

	/**
	*	\brief This enum is used to define control flags for TextureGrp objects.
	*	\sa SetControlFlag(), CheckControlFlag().
	*/
	enum TextureGrp_CheckFlag
	{
		TEXTGRP_NO_CONTROL		= 0x00000,		//!< No control done.
		TEXTGRP_SAME_WIDTH		= 0x00001,		//!< Identical width required.
		TEXTGRP_SAME_HEIGHT		= 0x00002,		//!< Identical height required.
		TEXTGRP_SAME_SIZE		= TEXTGRP_SAME_WIDTH + TEXTGRP_SAME_HEIGHT, 
		TEXTGRP_SAME_FORMAT		= 0x00004, 		//!< Identical format format required.
		TEXTGRP_SAME_INTERNAL_FORMAT	= 0x00008,		//!< Identical internal format required.
		TEXTGRP_SAME_PIXEL_TYPE	= 0X00010,		//!< Identical pixel type format required.
		TEXTGRP_SAME_ALL_FORMAT	= TEXTGRP_SAME_FORMAT + TEXTGRP_SAME_INTERNAL_FORMAT + TEXTGRP_SAME_PIXEL_TYPE, 
		TEXTGRP_SAME_ALL		= TEXTGRP_SAME_ALL_FORMAT + TEXTGRP_SAME_SIZE
	};


public:
#if TEXTUREGRP_USE_VECTOR
	std::vector<DataContainer*> m_TextVect;
#endif
protected:
	GLuint					m_controlFlag;		//!< Control flag to check if image must share some parameters like size, format...
	GLuint					m_textureType;		//!< Is set when the first texture is added, we consider that all the texture must be of the same type.
	TextGrpType				m_groupType;		//!< Define the type of the group, input or output.

#if !TEXTUREGRP_USE_VECTOR
	DataContainer **		m_TextArray;
	size_t					m_TextNbr;
	size_t					m_TextMaxNbr;
#endif

	//flag used for m_controlFlag and CheckControlFlag()
	TextSize<GLsizei>		m_ctrl_size;			//!< Control flag storing common size.
	GLuint					m_ctrl_internalFormat;	//!< Control flag storing common internalFormat.
	GLuint					m_ctrl_pixelType;		//!< Control flag storing common pixel Type.
	GLuint					m_mainAttachement;
public:
	/**
	*\brief Default constructor
	*/
	__GPUCV_INLINE
		TextureGrp();

	/**
	*\brief Destructor
	*/
	__GPUCV_INLINE
		~TextureGrp();

	virtual const std::string GetValStr()const{return "NOID";}

	virtual std::ostringstream & operator << (std::ostringstream & _stream)const;

	/**
	*	\brief Set the control flag for the group.
	*	\param _flag => see TextureGrp_CheckFlag.
	*	\sa CheckControlFlag, TextureGrp_CheckFlag. 
	*/
	__GPUCV_INLINE
		void SetControlFlag(TextureGrp_CheckFlag _flag);

	/**
	*	\brief Check that all the textures from the group match the control flag.
	*	\sa SetControlFlag, TextureGrp_CheckFlag.
	*/
	bool CheckControlFlag();

	/**
	*	\brief Add texture(s) to the group of texture.
	*	Check Hardware compatibilities before adding new textures.
	*	\param _tex => Pointer to texture(s) to add
	*	\param _nbr => Number of texture to add.
	*	\return True if success, false if any hardware compatibilities issues has been found.
	*/
	bool AddTextures(DataContainer ** _tex, unsigned int nbr=1);

	/**
	*	\brief Add texture to the group of texture.
	*	Check Hardware compatibilities before adding new textures.
	*	\param _tex => Pointer to texture to add
	*	\return True if success, false if any hardware compatibilities issues has been found.
	*/
	__GPUCV_INLINE
		bool AddTexture(DataContainer * _tex)
	{
		return AddTextures(&_tex, 1);
	}
	void SetGrpType(TextGrpType type);
	TextGrpType GetGrpType();
	/**
	*	\brief Clear the group.
	*/
	__GPUCV_INLINE
		void Clear();


#if _GPUCV_DEPRECATED
	/**
	*	\brief Bind all the textures using multi-texturing ARB extensions.
	*/
	void BindAll();
	void UnBindAll();

	/**
	* \sa http://www.opengl.org/documentation/specs/man_pages/hardcopy/GL/html/gl/texparameter.html
	* \deprecated
	*/
	virtual 
		void _SetTexParami(GLuint _ParamType, GLint _Value);

	int	GetCPUReturnNbr()const;
#endif

#if TEXTUREGRP_USE_VECTOR
	DataContainer* operator[](const unsigned int _i)
	{
		SG_Assert(_i < m_TextNbr, "Out of bound:"<<_i);
		return m_TextVect[_i];
	}
#else
	DataContainer* operator[](const size_t _i)
	{
		SG_Assert(_i < m_TextNbr, "Out of bound:"<<_i);
		return m_TextArray[_i];

		/*return NULL;RICHARD??*/
	}

#endif

	DataContainer* GetTex(const size_t _i)
	{
		SG_Assert(_i < m_TextNbr, "Out of bound:"<<_i);
		return m_TextArray[_i];
	}



	/**
	*	\brief Draw a quad using multi textures and their own texCoord.
	*	\todo check that it works with texture rectangle
	*	\note Right now, we assume that the texture ARB ID start from ARB0 and is ++
	*   \todo If all texture have same ARB id and same textures coordinates, use normal glTexCoord()...=>should be faster!!
	*/
	void DrawMultiTextQuad(int x, int y);

	/**
	*	\brief Push options for all Textures of the Group.
	*/
	virtual void PushOptions();

	/**
	*	\brief Pop options for all Textures of the Group.
	*/
	virtual void PopOptions();

	/**
	*	\brief Set option for all Textures of the Group.
	*	\param _opt => Option Type to be set.
	*	\param val => Value.
	*/
	virtual void SetOption(CL_Options::OPTION_TYPE _opt, bool val);

	/**
	*	\brief Get option can not be used, it is declared for compatibility only.
	*/
	virtual OPTION_TYPE GetOption(CL_Options::OPTION_TYPE _opt)const;

	/**
	*	\brief Push and set option for all Textures of the Group.
	*	\param _opt => Option Type to be set.
	*	\param val => Value.
	*/
	virtual void PushSetOptions(CL_Options::OPTION_TYPE _opt, bool val);

	/**
	*	\brief Check that given texture is present in the texture group.
	*	\param _tex => texture to look for.
	*	\return True if texture is found.
	*/
	virtual bool IsTextureInGroup(GPUCV_TEXT_TYPE _tex);

	/**
	*	\brief Replace one texture from the group by another one.
	*	\param _old => texture to look for and replace.
	*	\param _new => new texture.
	*	\return True if texture is found.
	*/
	virtual bool ReplaceTexture(GPUCV_TEXT_TYPE  _old, GPUCV_TEXT_TYPE _new);

	size_t GetTextNbr()const{return m_TextNbr;}
	size_t GetTextMaxNbr()const{return m_TextMaxNbr;}

	/** \brief Call a piece of code represented by function on all the textures of the group.
	*	\param TEXT => Name of the texture variable that will be used inside the macro.
	*	\param FCT  => Piece of code to call.
	*	\note This macro is designed to be called internally in the TextureGrp object.
	*	\sa TEXTURE_GRP_EXTERNE_DO_FOR_ALL.
	*/
#if TEXTUREGRP_USE_VECTOR
#define TEXTURE_GRP_INTERNE_DO_FOR_ALL(TEXT,FCT)\
	{\
	itObj itText = m_TextVect.begin();\
	DataContainer* TEXT=NULL;\
	for( itText = m_TextVect.begin(); itText != m_TextVect.end(); itText++)\
	{\
	TEXT= (*itText);\
	if(!TEXT)\
	continue;\
	FCT;\
	}\
	}
#else
#define TEXTURE_GRP_INTERNE_DO_FOR_ALL(TEXT,FCT)\
	{\
	DataContainer* TEXT=NULL;\
	size_t iTexID=0;\
	if(m_TextNbr ==1 )\
	{\
	TEXT = m_TextArray[0];\
	if(TEXT)\
	FCT;\
	}\
		else if(m_TextNbr>1)\
	{\
	for(iTexID=0; iTexID < m_TextNbr; iTexID++)\
	{\
	TEXT= m_TextArray[iTexID];\
	if(!TEXT)\
	continue;\
	FCT;\
	}\
	}\
	}
#endif

	/** \brief Call a piece of code represented by function on all the textures of the group.
	*	\param GRP => Texture group object to parse.
	*	\param TEXT => Name of the texture variable that will be used inside the macro.
	*	\param FCT  => Piece of code to call.
	*	\note This macro is designed to be called externally in the TextureGrp object.
	*	\sa TEXTURE_GRP_EXTERNE_DO_FOR_ALL.
	*/
#if TEXTUREGRP_USE_VECTOR
#define TEXTURE_GRP_EXTERNE_DO_FOR_ALL(GRP,TEXT,FCT)\
	{\
	TextureGrp::itObj itText = GRP->m_TextVect.begin();\
	DataContainer* TEXT=NULL;\
	for( itText = GRP->m_TextVect.begin(); itText != GRP->m_TextVect.end(); itText++)\
	{\
	TEXT= (*itText);\
	if(!TEXT)\
	continue;\
	FCT;\
	}\
	}
#else
#define TEXTURE_GRP_EXTERNE_DO_FOR_ALL(GRP,TEXT,FCT)\
	{\
	DataContainer* TEXT=NULL;\
	size_t TexGrpNbr = GRP->GetTextNbr();\
	size_t iTexID=0;\
	if(TexGrpNbr ==1)\
	{\
	TEXT = GRP->operator[](0);\
	if(TEXT)\
	FCT;\
	}\
		else if(TexGrpNbr>1)\
	{\
	for(iTexID =0; iTexID < TexGrpNbr; iTexID++)\
	{\
	TEXT= GRP->operator[](iTexID);\
	if(!TEXT)\
	continue;\
	FCT;\
	}\
	}\
	}
#endif

	/**
	*	\brief Transfer images to the corresponding target
	*	\sa DataContainer::SetLocation().
	*/
	template <typename TType>
	bool SetLocation(bool _dataTransfer=true)
	{
		CLASS_FCT_SET_NAME_TPL(TType,"SetLocation");
		CLASS_FCT_PROF_CREATE_START();

		int errorNbr=0;
		TEXTURE_GRP_INTERNE_DO_FOR_ALL(
			TEXT,
			if(!TEXT->SetLocation<TType>(_dataTransfer))
				errorNbr++;
		);
		if (errorNbr)
			return false;
		else
			return true;
	}
};

#if _GPUCV_GL_USE_GLUT
/** \brief Draw the given texture group into an OpenGL windows. Use for debugging with GpuCVSettings::EnableGlutDebug().
*/
_GPUCV_TEXTURE_EXPORT 
void ViewerDisplayTextureGroup(TextureGrp * _grp, SGE::CL_Vector2Df & _pos, SGE::CL_Vector2Df & _size);
#endif
}//namespace GCV
#endif
