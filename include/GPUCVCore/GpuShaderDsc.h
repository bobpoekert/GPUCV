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
#ifndef __GPUCV_CORE_SHADER_DSC_H
#define __GPUCV_CORE_SHADER_DSC_H
#include <GPUCVCore/config.h>


/** @ingroup GPUCV_SHADER_GRP
*  @{
*/
//#define LOADING_ERROR 0
//#define LOADING_FILE_TYPE_DISMISS 1

#define AUTO_UPDATE_FILES 1 //used to reload shader when the files have been modified from outside of the program

#if AUTO_UPDATE_FILES
#include <sys/stat.h>
#include <time.h>
#endif
//===========
namespace GCV{

#if 1
/**
*	\brief Class to describe and manipulate GLSL shader files and program(Frag/VERT/GEO).
*	\author Packaged into the class by Yannick Allusse
*	\author Original source code from Jean-Philippe Farrugia and Erwan Guehenneux
*/
class _GPUCV_CORE_EXPORT ShaderDsc
	:public CL_Profiler
	,public SGE::CL_BASE_OBJ_FILE
{
public:
	/** \brief Shader type enum.
	*/
	enum SHADER_TYPE
	{
		GEOMETRY_SHADER		= GL_GEOMETRY_SHADER_EXT	//!< GLSL Geometry shader
		,VERTEX_SHADER		= GL_VERTEX_SHADER_ARB		//!< GLSL Vertex shader
		,FRAGMENT_SHADER	= GL_FRAGMENT_SHADER_ARB	//!< GLSL Fragment shader
		,UNDEFINED			= 1							//!< Type is not defined
		,NOT_SHADER			= 0
		,FILE_ERROR			= -1						//!< Error occured when loading shader program
	};
protected:
//	std::string			m_file;			//!< shader filename.
	std::string			m_meta_tag;		//!< shader meta tags.
//	std::string			m_extension;	//!< File extension, should be in {frag/vert/geo}.
//	std::string			m_path;			//!< Local path to shader file.
	vector <string>		m_vKeywords;	//!< List of meta tag name.
	vector <string>		m_vValues;		//!< List of meta tag values.
	enum SHADER_TYPE	m_type;			//!< Type of the shader.
	GLhandleARB			m_shaderID;		//!< OpenGL shader ID.
	std::string			m_filebuffer;	//!< String that contain source code when compiling.
	GLuint				m_inputGeometryType;	//!< Used only in geometry shaders. Define geometry input type in {GL_POINTS, GL_LINES, GL_LINES_ADJACENCY_EXT, GL_TRIANGLES, GL_TRIANGLES_ADJACENCY_EXT}.
	GLuint				m_outputGeometryType;	//!< Used only in geometry shaders. Define geometry output type in {GL_POINTS, GL_LINE_STRIP, GL_TRIANGLE_STRIP}.
public:
	/** \brief Constructor 
	*	\param _name => name of the shader file(including local path, filename, extension and meta tags).
	*	\sa SetName().
	*/
	//__GPUCV_INLINE
	ShaderDsc(const std::string &_name);

	/** \brief Default constructor */
	//__GPUCV_INLINE
	ShaderDsc(void);

	/** \brief Destructor */
	__GPUCV_INLINE
		~ShaderDsc();

	/** \brief Set the shader progam name.
	*	\param _name => name of the shader file(including local path, filename, extension and meta tags).
	*	Format of name must be : PATH/FILENAME(META_TAGS).EXTENSION.
	*	\sa ParseMetaTag()
	*/
	void	SetName(const std::string &_name);

	/*!
	*   \brief Parse meta_tags from filename and list them into vectors m_vKeywords and m_vValues.
	*	All GpuCV shader files contains some META_DATA that can be updated at compile time. 
	*	Some of them are set with default values such as {GETTEX, IMGTYPE, GPUCV_FILTER}.
	*	It support also customized META_SHADER set by the calling operators, theses arguments are given in the filename.
	*	Calling syntax is : shaderfile($keyword1=value1$keyword2=value).frag / .vert
	*	The corresponding shader code must have: 
	\code
	#define keyword1 
	#define keyword2
	\endcode
	*   After applying META_TAGS, shader code will be:
	*	The corresponding shader code must have: 
	*	\code
	#define keyword1 value1
	#define keyword2 value2
	*	\endcode
	*	\note The '$' character is used to begin the different parameters, cause '$' is not used in GLSL code.
	*	\author Yannick Allusse.
	*	\author Jean-Philippe Farrugia
	*	\sa ParseShaderCode()
	*	\note For geometry shaders: Geometry input/output types must use meta tags GEO_INPUT/GEO_OUTPUT, see m_inputGeometryType/m_outputGeometryType for details.
	*/
	void	ParseMetaTag();

	/*!
	*   \brief Parse shader code and replace all META_TAGS into source.
	*	\sa ParseMetaTag()
	*/	
	void	ParseShaderCode();

	/*!
	*   \brief Load file given GetFullPath() into m_filebuffer.
	*	\sa CompileShader()
	*/
	virtual 
	int	Load();

	/*!
	*   \brief Compile the GLSL shader program using OpenGL. 
	*	\sa LoadFile()
	*/
	GLhandleARB	CompileShader();

	void PrintLocalLogInfo()const;





	/**	\brief Return full file path.*/
	virtual 
		const std::string GetFullPath()const;

#if 0
	/**	\brief Return file extension.*/
	__GPUCV_INLINE
		const std::string  & GetExtension(void)const {return m_extension;}

	/**	\brief Return file name.*/
	__GPUCV_INLINE
		const std::string  & GetFileName(void)const {return m_file;}

	/**	\brief Return file local path.*/
	__GPUCV_INLINE
		const std::string  & GetPath(void)const {return m_path;}

#endif
	/**	\brief Return META_TAG string.*/
	__GPUCV_INLINE
		const std::string  & GetMetaTags(void)const;

	
	/**	\brief Return shader program type.*/
	__GPUCV_INLINE
		const SHADER_TYPE  & GetType(void)const;

	/**	\brief Return geometry shader input type.
	*	\sa m_inputGeometryType.
	*/
	__GPUCV_INLINE
		const GLuint  & GetGeoInputType(void)const;

	/**	\brief Return geometry shader output type.
	*	\sa m_outputGeometryType. 
	*/
	__GPUCV_INLINE
		const GLuint  & GetGeoOutputType(void)const;

	/**	\brief Output shader program information to given ostringstream.*/
	virtual 
		std::ostringstream & operator << (std::ostringstream & _stream)const;
};

#endif//ShaderDsc
/** @}*///SHADER_GRP
#endif
}//namespace GCV

