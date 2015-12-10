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



#ifndef __GPUCV_CORE_FILTER_H
#define __GPUCV_CORE_FILTER_H

#ifdef _MACOS
	#include <sys/malloc.h>
#else
	#include <malloc.h>
#endif
#include <GPUCVCore/GpuShaderManager.h>
#include <GPUCVCore/GpuTextureManager.h>

namespace GCV{
#define FCT_PTR_DRAW_TEXGRP(NAME)void(*NAME)(GCV::TextureGrp*,GCV::TextureGrp*, GLuint, GLuint)
#define FCT_PTR_DRAW_TEX(NAME)void(*NAME)(GCV::DataContainer**, GLuint, GCV::DataContainer**, GLuint, GLuint, GLuint)

/** @addtogroup GPUCV_SHADER_GRP
*  @{
*/

/**
*	\brief Manage a context (drawing function and parameter) for a specific filter
*	 \author Yannick Allusse
*	\author Erwan Guehenneux
*	\author Jean-Philippe Farrugia
*
*  This class centralize all informations between CPU and GPU and centralized 
*  them to simply apply on filter on a image.
*/
class _GPUCV_CORE_EXPORT GpuFilter
	: public CL_Profiler
	, public SGE::CL_BASE_OBJ<std::string>
{
public:
	typedef	TextSize<GLsizei>	FilterSize;
private : 
	//string		m_shader_name;   //!< filter Id
	int			m_display_list;  	//!< object where shader should be apply
	float*		m_params;        //!< Parameters for the filter
	int			m_nb_params;     //!< Number of parameters, excluding texture handle
	ShaderManager *m_ShaderManager; //!< link to one unique cvgShaderManager
protected:
	//	TextureGrp *m_inputTextureGrp;
	//	TextureGrp *m_outputTextureGrp;

public : 
	/*!
	*	\brief default constructor
	*/
	GpuFilter();

	/*!
	*	\brief Constructor
	*/
	GpuFilter(std::string _name);

	/*!
	*	\brief Destructor
	*/
	~GpuFilter();

	virtual 
		std::ostringstream & operator << (std::ostringstream & _stream)const;

	/**
	*	\brief link the cvgFilter to a ShaderManager
	*	\param *sm -> pointer to a ShaderManager
	*/
	__GPUCV_INLINE
		void  SetShaderManager(ShaderManager *sm);

	/**
	*	\brief associate an OpenGL Display list to the GpuFilter
	*	\param list -> corresponding display list
	*/
	__GPUCV_INLINE
		void SetDisplayList(GLuint list);

	// Accessors and mutators
	/*!
	*	\brief define the corresponding source code of this filter, load and compile
	*	\param _fragmentShaderFilename -> fragment shader source code file
	*	\param _vertexShaderFilename -> vertex shader source code file
	*	\return string -> filter id or "" if an error occured
	*/
#if _GPUCV_SUPPORT_GS
	string SetShadersFiles(ShaderProgramNames & _Names);
#else
	string SetShadersFiles(string _fragmentShaderFilename, string _vertexShaderFilename);
#endif

	/*!
	*	\brief Get the corresponding filenames for the shader
	*	\param filename1 -> vertex or fragment shader source code file
	*	\param filename2 -> vertex or fragment shader source code file
	*/
	__GPUCV_INLINE
		void   GetShadersFiles(string &filename1, string &filename2);

	// parameter management
	/*!
	*	\brief initialize the parameter array for the cvgFilter
	*/
	__GPUCV_INLINE
		void   ClearParams();

	/*!
	*	\brief set a specific parameter
	*	\param i -> array index (number) of the parameter to set
	*	\param new_param -> new parameter value
	*	\return int exiting status
	*/
	void    SetParamI(unsigned int i,float new_param);

	/*!
	*	\brief add a parameter for this filter
	*	\param new_param -> value of the new parameter
	*	\return int exiting status
	*/
	__GPUCV_INLINE
		void    AddParam(float new_param);

	/*!
	*	\brief get the parameter array
	*	\return float * -> pointer to the parameter array
	*/
	__GPUCV_INLINE
		float*      GetParams();
	/*!
	*	\brief get the number of parameters set in the parameter array
	*	\return int -> number of parameters
	*/
	__GPUCV_INLINE
		int     GetNbParams();

	/*!
	*	\brief update a float array of parameters 
	*	For each parameters of the array, test if it already exists and updates it, else add a new parameter.
	*	\param Params -> float array of parameters to update
	*	\param ParamNbr -> number of parameters
	*/
	void SetParams(const float * Params , int ParamNbr);

	/*!
	*	\brief get the number of parameters set in the parameter array
	*	\return int -> number of parameters
	*/
	__GPUCV_INLINE
		const std::string & GetName()const;
protected:
	/** \brief Set the name(ID) of the filter
	*	\param _name -> Name to affect to the filter.
	*/
	__GPUCV_INLINE
		void      SetName(string _name);	
public:

	/*!
	*	\brief get the OpenGL shader Handle of this filter
	*	\return GLhandleARB handle of the shader program (or 0 if an error occurred)
	*/
	__GPUCV_INLINE
		GLhandleARB GetProgram();

	/*! 
	*	\brief apply a filter on a TextureGrp and store result in another TextureGrp.
	*	\param InputGrp -> source TextureGroup.
	*	\param OutputGrp -> destination TextureGroup.
	*	\param _size -> filter rendering size.
	*	\param _DrawFct -> Customized drawing function[Optional].
	*	\return int -> status of execution
	*/
	int Apply(TextureGrp * InputGrp, TextureGrp * OutputGrp, 
		FilterSize * _size, 
		FCT_PTR_DRAW_TEXGRP(_DrawFct)=NULL);
	/*! 
	*	\brief Applies a filter on some input Texture and stores result in another texture.
	*	\param th_s -> source OpenGL texture id
	*	\param th_d -> destination OpenGL texture id
	*	\param _size -> image rendering size.
	*	\param more_tex -> array of textures id used if filter need more than one texture source
	*	\param nb_tex -> number of textures set in more_tex array
	*	\param _DrawFct -> Customized drawing function[Optional].
	*	\return int -> status of execution
	*/

	int Apply(GPUCV_TEXT_TYPE th_s,GPUCV_TEXT_TYPE th_d,
		FilterSize * _size, 
		GPUCV_TEXT_TYPE * more_tex=NULL, int nb_tex=0, 
		FCT_PTR_DRAW_TEXGRP(_DrawFct)=NULL);

	void PreProcessDataTransfer(TextureGrp * _Grp, TextureGrp * _OptControlGrp=NULL);

	void PostProcessDataTransfer(TextureGrp * _Grp);

	// TextureGrp * GetInputGrp(){return m_inputTextureGrp;}
	// TextureGrp * GetOutput(){return m_outputTextureGrp;}
private:
	/*!
	*	\brief get location of one parameter in a compile shader
	*	\param prog -> program to scan
	*	\param name -> parameter to find
	*	\return GLuint -> location of the parameter
	*/
	GLuint FindParam(GLhandleARB prog, const char* name);
};

_GPUCV_CORE_EXPORT void ViewerDisplayFilter(GpuFilter *, TextureGrp * _ingrp, TextureGrp * _outgrp);

/** @}*///GPUCV_SHADER_GRP
}//namespace GCV
#endif
