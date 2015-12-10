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






#ifndef __GPUCV_CORE_SHADER_MANAGER_H
#define __GPUCV_CORE_SHADER_MANAGER_H

#include	<GPUCVCore/GpuHandleManager.h>
namespace GCV{


/** @addtogroup GPUCV_SHADER_GRP
*  @{
*/
#define SHADER_NOT_FOUND -1
#define SHADER_FOUND      1

#define ShaderManager_USE_MANAGER 0
/**
*	\brief store and manager cvgShaderObject objects
*	\author Erwan Guehenneux
*	\author Jean-Philippe Farrugia
*
*  Manage all ShaderObject objects and allow to create, access, remove them safely.
*/
class _GPUCV_CORE_EXPORT ShaderManager
#if ShaderManager_USE_MANAGER
	: public SGE::CL_TplObjManager<ShaderObject, std::string>
	, public CL_Singleton<ShaderManager>
#endif
{
#if ShaderManager_USE_MANAGER

#else
private : 

	vector<ShaderObject *> manager; // shader_object array
	GpuHandleManager stack;

	/*!
	*	\brief Find a ShaderObject in the vector by giving its name (id)
	*	\param name -> name (id) of the corresponding shader_object
	*	\return int -> corresponding index array
	*/ 
	int FindByName(const string & name)const;

public:
	const ShaderObject* FindConst(const string & name)const;
	/*!
	*	\brief  get the number of shader_objects
	*	\return int -> number of shaders
	*/  
	int  GetNumberOfShaders();

#endif

	/*!
	*	\brief Find a cvgShaderObject in the array by giving its GLSL shader filename
	*	\param filename -> filename of the corresponding GLSL shader
	*	\return int -> corresponding index array
	*/    
	int FindByFilename(string filename);

public : 

	/*! \brief  default constructor */
	ShaderManager();

	/*! \brief  destructor*/
	~ShaderManager();

	// management functions

	/*!
	*	\brief Load and add a new ShaderObject in the vector 
	*	\param  _fragmentShaderFilename -> filename of fragment shader source code
	*	\param _vertexShaderFilename -> filename of vertex shader source code
	*	\return string -> name (id) of the new shader_object
	*/ 
#if _GPUCV_SUPPORT_GS
	string AddShader(ShaderProgramNames & _ShaderNames);
#else
	string AddShader(string _fragmentShaderFilename, string _vertexShaderFilename);
#endif




	/*!
	*	\brief  remove a specific ShaderObject by giving its name (id)
	*	\param  name -> name (id) of the shader_object
	*	\return boolean -> status
	*/      
	bool   RemoveShader(string name);

	/*!
	*	\brief Check if the shader file has been changed, and recompile it if needed
	*	\param ShaderName -> Name of the Shader object
	*	\return bool -> true if update is successfull, else false
	*/ 
	bool UpdateShader(string ShaderName);


	// accessors
	/*!
	*	\brief  get the GLSL program handle of a specific shader_object 
	*	\param  name -> name (id) of the shader_object
	*	\param  handle -> GLSL handle program wich will be set by the cvgShaderObject
	*	\return boolean -> status
	*/   
	bool GetShader(string name, GLhandleARB &handle);


	/*!
	*	\brief  get the source vertex & fragment source code of one CggShaderObject
	*	\param  name -> name (id) of the shader_object
	*	\param  filename1 -> String use to store the filename the vertex shader source code 
	*	\param  filename2 -> String use to store the filename the fragment shader source code 
	*/ 
	void GetShaderFiles(string name, string &filename1, string &filename2);


};
/** @}*///GPUCV_SHADER_GRP
}//namespace GCV
#endif
