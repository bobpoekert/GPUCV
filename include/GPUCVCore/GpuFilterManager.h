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
#ifndef __GPUCV_CORE_FILTER_MANAGER_H
#define __GPUCV_CORE_FILTER_MANAGER_H

#include <GPUCVCore/GpuFilter.h>

namespace GCV{
/** @addtogroup GPUCV_SHADER_GRP
*  @{
*/

/**
*	\brief Manage cvgFilter objects
*	\author Yannick Allusse
*	\author Erwan Guehenneux
*	\author Jean-Philippe Farrugia
*
*  This class store all GpuFilter object.
*	\sa cvgFilter
*/

class _GPUCV_CORE_EXPORT GpuFilterManager
	//: public SGE::CL_TplObjManager<GpuFilter, std::string>
	: public SGE::CL_TEMPLATE_OBJECT_MANAGER<GpuFilter, std::string>
	, public CL_Singleton<GpuFilterManager>
{
public:
	//typedef		SGE::CL_TplObjManager<GpuFilter, std::string> TplManager;
	typedef		SGE::CL_TEMPLATE_OBJECT_MANAGER<GpuFilter, std::string> TplManager;
	typedef		std::string		TypeID;	//! < Type of the identifier.
	typedef		std::string*	TypeIDPtr;	//! < Type of the identifier.
private : 
	ShaderManager m_ShaderManager;	//!< Storage of shader_objects (GLSL shader programs)
public : 

	/*!\brief Constructor.*/
	GpuFilterManager();

	/*! \brief Destructor.
	*	Destroy all the GpuFilter from GpuFilterManager::manager.
	*	\sa GpuFilter::~GpuFilter()
	*/
	~GpuFilterManager();

	/*! 
	*	\brief Get a cvgFilter name by giving its source code filenames. If needed, it will be creating and loaded.
	*	\param _fragmentShaderFilename -> fragment source code filename.
	*	\param _vertexShaderFilename -> vertex source code filename.
	*	\return If succes return GpuFilter name.
	*	\return else return empty string "".
	*	\sa GpuFilterManager::RemoveCvgFilter()
	*	\bug No error message are raised when shader file is not found..??
	*/
#if _GPUCV_DEPRECATED
	string GetFilterName(string _fragmentShaderFilename, string _vertexShaderFilename="");
#endif
#if _GPUCV_SUPPORT_GS
	GpuFilter * GetFilterByName(ShaderProgramNames & _ShaderNames);
	GpuFilter * GetFilterByName(std::string & file1, std::string file2="",std::string file3="");
#else
	GpuFilter * GetFilterByName(string _fragmentShaderFilename, string _vertexShaderFilename="");
#endif

	/*!
	*	\brief Get the Shader manager object.
	*	\return Reference to shader manager.
	*/
	ShaderManager & GetShaderManager(){return m_ShaderManager;}
};

/*!
*	\brief get one unique occurrence of one GpuFilterManager
*	\return GpuFilterManager* -> pointer to one unique GpuFilterManager
*/		
#define GetFilterManager() GpuFilterManager::GetSingleton()


/*!
*	\sa TemplateOperator().
*/		
_GPUCV_CORE_EXPORT
void TemplateOperator(const std::string &_fctName, 
					  const std::string &_filename1, const std::string &_filename2,
					  DataContainer * _src1, DataContainer * _src2, DataContainer * _src3, 
					  DataContainer *_dest, 
					  const float *_params=NULL, unsigned int _param_nbr=0,
					  TextureGrp::TextureGrp_CheckFlag _controlFlag=TextureGrp::TEXTGRP_NO_CONTROL, std::string _optionalMetaTag="",
					  FCT_PTR_DRAW_TEXGRP(_DrawFct)=NULL);

#if _GPUCV_SUPPORT_GS
_GPUCV_CORE_EXPORT 
void TemplateOperatorGeo(const std::string & _fctName, ShaderProgramNames & _ShaderNames,	
						 TextureGrp * _srcGrp, TextureGrp * _dstGrp,
						 const float *_params=NULL, unsigned int _param_nbr=0,
						 TextureGrp::TextureGrp_CheckFlag _controlFlag=TextureGrp::TEXTGRP_NO_CONTROL, std::string _optionalMetaTag="",
						 FCT_PTR_DRAW_TEXGRP(_DrawFct)=NULL);
#endif

_GPUCV_CORE_EXPORT 
void TemplateOperator(const std::string & _fctName, const  std::string & _filename1, const std::string & _filename2,
					  TextureGrp * _srcGrp, TextureGrp * _dstGrp,
					  const float *_params=NULL, unsigned int _param_nbr=0,
					  TextureGrp::TextureGrp_CheckFlag _controlFlag=TextureGrp::TEXTGRP_NO_CONTROL, std::string _optionalMetaTag="",
					  FCT_PTR_DRAW_TEXGRP(_DrawFct)=NULL);

/** @}*/ //GPUCV_SHADER_GRP
}//namespace GCV
#endif

