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
#ifndef __GPUCV_CORE_SHADER_OBJECT_H
#define __GPUCV_CORE_SHADER_OBJECT_H


#include <GPUCVCore/GpuShaderDsc.h>

#if _GPUCV_SUPPORT_GS
#include <GPUCVCore/GpuShaderDsc.h>
#else
#if _GPUCV_CORE_CHECK_SHADER_CHANGE
#include <sys/stat.h>
#include <time.h>
#endif
#endif

#include <GPUCVCore/GpuHandleManager.h>

namespace GCV{
/** @addtogroup GPUCV_SHADER_GRP
*  @{
*/
#define _GPUCV_SUPPORT_GS 1


//===========

class GpuHandleManager;

struct ShaderProgramNames
{
	std::string m_ShaderNames[3];
};

/**
*	\brief load, store and manage GLSL shaders
*	\author Erwan Guehenneux
*	\author Jean-Philippe Farrugia
*	\author Yannick Allusse
*
*  Manage all information linked to GLSL shader loading.
*  Verify filenames, parse and recompile meta-shaders, etc
*/
class _GPUCV_CORE_EXPORT ShaderObject
	:public SGE::CL_BASE_OBJ<std::string>
	,public CL_Profiler
{
#if !_GPUCV_SUPPORT_GS
public:

	enum SHADER_TYPE
	{
		GEOMETRY_SHADER = GL_GEOMETRY_SHADER_EXT,
		VERTEX_SHADER	= GL_VERTEX_SHADER_ARB,
		FRAGMENT_SHADER = GL_FRAGMENT_SHADER_ARB,
		UNDEFINE		= 1,
		NOT_SHADER		= 0,
		FILE_ERROR		= -1
	};
#endif
private : 
	//	   std::string name;                     // shader name (id)

#if _GPUCV_SUPPORT_GS
	ShaderDsc *m_vertex_shader;//!< vertex shader description
	ShaderDsc *m_fragment_shader;//!< fragment shader description
	ShaderDsc *m_geometry_shader;//!< geometry shader description
#else
	string vertex_name;			//!< vertex shader name including path and meta tags.
	string vertex_file;			//!< vertex shader filename
	string vertex_meta_tag;     //!< vertex meta tags

	string fragment_file;		//!< fragment shader filename
	string fragment_name;		//!< fragment shader name including path and meta tags.
	string fragment_meta_tag;   //!< fragment meta tags

	vector <string> Vkeywords;
	vector <string>  Vvalues;

	vector <string> Fkeywords;
	vector <string>  Fvalues;
#endif 
	int    fifo_pos;
	bool   is_loaded;
	GpuHandleManager * stack;

#if !_GPUCV_SUPPORT_GS	  
	/*!
	*   \brief  parse filename for meta-shader calling function, separate keywords, values from the filename 
	*	\ some file are meta-shader, some parameters are set directly in the code before being
	*	 compiled. Those arguments are given in the filename calling
	*	 syntax is : shaderfile($keyword1=value1$keyword2=value).frag / .vert
	*	 the corresponding shader code should present : 
	\code
	#define keyword1 and #define keyword2
	\endcode
	*	The '$' character is used to begin the different parameters, cause '$' is not used in GLSL code.
	*	\param  _tags -> filename wich contains eventually meta-shader informations
	*	\param  _KeyVector -> stack containing tag key(s) found.
	*	\param  _ValVector -> stack containing tag value(s) found.	
	*  <h3>v0.2 updates:</h3>
	<ul><li>Character '$' is used to begin each meta-shader parameters, example :  shaderfile($keyword1=value1$keyword2=value).frag</li>
	<li>Meta-shader parameters can now include divide operations using '/', example :  shaderfile($keyword1=value1/value2).frag</li>
	<li>Meta-shader parameters can now include '()' characters, example :  shaderfile($keyword1=(value1/value2)/value3).frag</li>
	</ul>
	\author Yannick Allusse.
	\author Jean-Philippe Farrugia
	*/
	void	ParseMetaTag(string _tags, vector <string> & _KeyVector, vector <string> & _ValVector);

	/*!
	*	\brief  Insert in shader code informations previously separated by ParseFilename()
	*	\param  _codevalue -> shader code to insert meta values.
	*	\param  _KeyVector -> stack containing tag key to find.
	*	\param  _ValVector -> stack containing tag value to insert.
	*/

	void	ParseShaderCode(string &_codevalue, vector <string> & _KeyVector, vector <string> & _ValVector);
#endif
public : 
	/*!
	*	\brief  default constructor
	*	\param  s -> pointer to one unique instance of GpuHandleManager wich will manage GLSL headers
	*/
	ShaderObject(GpuHandleManager * s);

	/*! \brief  destructor */
	~ShaderObject();

	/*!
	*	\brief  verify and save paths to vertex and fragment shader, parse their filename if necessary
	*	\param  _fragmentShaderFilename -> fragment shader filename
	*	\param  _vertexShaderFilename -> vertex shader filename
	*	\return string -> filter name (id)
	*	\todo check that files exist.
	*/
#if _GPUCV_SUPPORT_GS
	string cvLoadShaders(ShaderProgramNames & _shaderNames);
	static std::string GenerateShaderUniqueName(ShaderProgramNames & _shaderNames);
#else

	string cvLoadShaders(string _fragmentShaderFilename, string _vertexShaderFilename="");


	static void		SplitFileName(const std::string _srcName, std::string & _path, 
		std::string & _name, std::string & _extension, 
		std::string & _metacode);
#endif
	// accessors

	/*!
	*	\brief  get the filter name (id)
	*	\return string -> ShaderObject name (id)
	*/    
	//string GetName();

	/*!
	*	\brief  get path to the vertex shader code filename
	*	\return string : vertex shader filename
	*/
	const std::string  GetVertexFile()const;

	/*!
	*	\brief  get path to the fragment shader code filename
	*	\return string : fragment shader filename
	*/
	const std::string  GetFragmentFile()const;

	/*!
	*	\brief  get path to the geometry shader code filename
	*	\return string : geometry shader filename
	*/
	const std::string  GetGeometryFile()const;


	/*!
	*	\brief Update the index of GpuHandleManager list wich correspond to the handle of current shader 
	*	\param pos -> new position to set
	*/
	void SetQueueIndex(int pos);

	/*!
	*	\brief Set if the current shader is compiles a loaded on Gpu or not
	*	\param status -> true if is loaded, false if not
	*/
	void SetIsLoaded(bool status);


	void PrintInfo(bool _shaderCode, GLhandleARB vertex_program, GLhandleARB fragment_program);
	std::ostringstream &  operator << (std::ostringstream & _stream)const;
	/*!
	*	\brief  get the ShaderObject corresponding GLSL shader openGL handle
	*  If shader has already been loaded, return the GLSL shader handle, then load both vertex and fragment
	*	shader files.
	*	\return GLhandleARB -> GLSL shader handle
	*/
	GLhandleARB GetShaderHandle();


private:
#if !_GPUCV_SUPPORT_GS
	bool LoadShaderFile(string _name, std::string & _filename, std::string & _metatag, GLenum &_shader_type);
#endif
	//GLhandleARB LoadShaderCode(string filename);
#if _GPUCV_CORE_CHECK_SHADER_CHANGE
#if !_GPUCV_SUPPORT_GS
private:
	struct tm VertexClock;
	struct tm FragClock;
#endif
public:
	bool AreFilesChanged(void);
#endif

	//=======================
};

/*!
*	\brief  print info log for a shader
*	\param  obj -> openGL header of the shader to log
*	\return int -> status
*/
int printInfoLog(GLhandleARB obj);

/** @}*///GPUCV_SHADER_GRP
}//namespace GCV
#endif
