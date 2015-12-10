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
#include <GPUCVCore/GpuShaderDsc.h>

namespace GCV{
#if 1

void SplitFileName(const std::string _srcName,
				   std::string & _path,
				   std::string & _name,
				   std::string & _extension,
				   std::string & _metacode)
{
	_name = _srcName;
	//remove extension
	size_t curPos = _name.find_last_of('.')+1;
	if (curPos != std::string::npos)
	{
		_extension = _name.substr(curPos, _name.size()-curPos);
		_name = _name.substr(0, _name.size()-_extension.size()-1);
	}
	else
		_extension = "";

	//remove meta options
	size_t last_pos = _name.find_last_of(')');
	curPos = _name.find_first_of('(');
	if (curPos != std::string::npos && last_pos != std::string::npos)
	{
		_metacode = _name.substr(curPos+1, last_pos-curPos-1);
		_name = _name.substr(0, curPos);
	}
	else
		_metacode = "";

	//remove filepath
	curPos = _name.find_last_of('\\')+1;
	if (curPos==0)
		curPos = _name.find_last_of('/')+1;
	if(curPos != std::string::npos)
		_path = _name.substr(0, curPos);
	else
		_path = "";
	_name = _name.substr(curPos, _name.size()-curPos);
}
//================================================
ShaderDsc::ShaderDsc(const std::string & _name)
:SGE::CL_BASE_OBJ_FILE(_name)
,CL_Profiler("ShaderDsc")
,m_type(UNDEFINED)
,m_shaderID(0)
,m_inputGeometryType(GL_POINTS)
,m_outputGeometryType(GL_POINTS)
{
	SetName(_name);
}
//================================================
ShaderDsc::ShaderDsc()
:SGE::CL_BASE_OBJ_FILE("")
,CL_Profiler("ShaderDsc")
,m_type(UNDEFINED)
,m_shaderID(0)
,m_inputGeometryType(GL_LINE_STRIP)
,m_outputGeometryType(GL_LINE_STRIP)
{
}
//================================================
void ShaderDsc::SetName(const std::string & _name)
{
	SetID(_name);
	SplitFileName(_name, Path, FileName, Extension, m_meta_tag);

	m_type = ((Extension) == "vert") ? VERTEX_SHADER
		: ((Extension) == "frag") ? FRAGMENT_SHADER
		: ((Extension) == "geo") ? GEOMETRY_SHADER
		: ((Extension) == "old") ? NOT_SHADER
		: ((Extension) == "CVS") ? NOT_SHADER
		: ((Extension) == "SVN") ? NOT_SHADER
		: ((Extension) == "") ? UNDEFINED
		: FILE_ERROR;
	//parse meta tags
	ParseMetaTag();
}
//================================================
ShaderDsc::~ShaderDsc()
{

}
//================================================
void ShaderDsc::ParseMetaTag()
{
	//add default tags

	//added by Yann.A. 10/10/2005 : default meta parameters for all shaders(Texture)
	//default Meta for all shaders to specify texture type
	string key = "GETTEX";
	string value;
	if(ProcessingGPU()->IsHardwareFamily(GenericGPU::HRD_FAM_ATI))
	{//ATI only support texture2D now? textureRect does not work with new drivers on my ATI radeon 9700.
		value = "texture2D";
	}
	else
	{
		switch (GetHardProfile()->GetTextType())//! \todo depend on input type..???
		{
		case GL_TEXTURE_2D:				value="texture2D";		break;
		case GL_TEXTURE_2D_ARRAY_EXT:	value="texture2DArray";	break;
		case GL_TEXTURE_RECTANGLE_ARB:	value="textureRect";	break;
		default: SG_Assert(0, "Unknown texture type");
		}
	}

	value+=" // ";
	m_vKeywords.push_back(key);
	m_vValues.push_back(value);

	//default Meta for all shaders to specify image type
	key = "IMGTYPE";
	if(ProcessingGPU()->IsHardwareFamily(GenericGPU::HRD_FAM_ATI))
	{//ATI only support sampler2D now? samplerRect does not work with new drivers on my ATI radeon 9700.
		value = "uniform sampler2D";
	}
	else
	{
		switch (GetHardProfile()->GetTextType())//! \todo depend on input type..???
		{
		case GL_TEXTURE_2D:				value="uniform sampler2D";		break;
		case GL_TEXTURE_2D_ARRAY_EXT:	value="uniform sampler2DArray";	break;
		case GL_TEXTURE_RECTANGLE_ARB:	value="uniform samplerRect";	break;
		default: SG_Assert(0, "Unknown texture type");
		}
	}

	value+=" // ";
	m_vKeywords.push_back(key);
	m_vValues.push_back(value);

	//default Meta to specify that we are compiling with GPUCV and not "Shader designer" app
	key = "GPUCV_FILTER";
	value = "1";
	value+=" // ";
	m_vKeywords.push_back(key);
	m_vValues.push_back(value);


	//parse all other meta tag
	string params = m_meta_tag;
	//int current_pos = 0;
	string::size_type equal_pos=0, dollar_pos=0;

	while (1)
	{
		equal_pos = params.find_first_of('=');
		dollar_pos = params.find_first_of('$');

		SG_Assert(!((equal_pos==std::string::npos)&& (dollar_pos != std::string::npos)),
			"meta parameter error : missing = between keyword & parameter.");

		if ((equal_pos == dollar_pos) && (equal_pos == std::string::npos))
		{//scan finish
			//GPUCV_ERROR("Critical : meta parameter error : missing = between keyword & parameter.\n");
			break;
		}
		if(dollar_pos==-1)
			break;//each keyword must start with $



		key = params.substr(dollar_pos+1, equal_pos-1);
		params = params.substr(equal_pos+1, params.size() - (equal_pos+1));
		dollar_pos = params.find_first_of('$');//find next parameter using $
		value = params.substr(0, (dollar_pos!=-1)?dollar_pos:params.size());

		/*current_pos = params.find('$', current_pos)+1;//Yann.A. 22/11/2005, ',' has been replaced by '$'!!

		if (current_pos == 0)  current_pos = pos_parenthese2;

		value = params.substr(equal_pos+1, current_pos-equal_pos-2);
		*/
		//if geometry shader, we parse again all tags to find geometry type.
		if(m_type==GEOMETRY_SHADER)
		{
			if(key=="GEO_INPUT")
			{
				m_inputGeometryType = GetGeometryTypeFromStr(value);
				//check input supported case:
				switch (m_inputGeometryType)
				{
					//supported format
				case GL_POINTS:
				case GL_LINES:
				case GL_LINES_ADJACENCY_EXT:
				case GL_TRIANGLES:
				case GL_TRIANGLES_ADJACENCY_EXT:
					break;
				default:		//unsupporeted

					SG_Assert(0, "ShaderDsc::ParseMetaTag() unsupported geometry input type.");
					break;
				}
			}
			else if(key=="GEO_OUTPUT")
			{
				m_outputGeometryType = GetGeometryTypeFromStr(value);
				//check input supported case:
				switch (m_outputGeometryType)
				{
					//supported format
				case GL_POINTS:
				case GL_LINE_STRIP:
				case GL_TRIANGLE_STRIP:
					break;
				default:		//unsupporeted
					SG_Assert(0, "ShaderDsc::ParseMetaTag() unsupported geometry output type.");
					break;
				}
			}
		}
		else
			value+=" // ";
		m_vKeywords.push_back(key);
		m_vValues.push_back(value);

		if ((equal_pos != dollar_pos) && (dollar_pos == std::string::npos))
			break;
		else
			params= params.substr(dollar_pos, params.size() - dollar_pos+1);
	}


}
//================================================
void ShaderDsc::ParseShaderCode()
{
	string Msg;
	size_t keyword_pos;
	//remove license text...
	keyword_pos = m_filebuffer.find("CVG_LicenseEnd");
	if(keyword_pos!=std::string::npos)//license found
		m_filebuffer.erase(0, keyword_pos+strlen("CVG_LicenseEnd"));

	//add meta keys and values
	for (int i=(int)m_vKeywords.size(); i--;)
	{
		keyword_pos = m_filebuffer.find("#define "+m_vKeywords[i]);
		if(keyword_pos==std::string::npos)
		{
			GPUCV_WARNING("shader file '"<< GetID() << "' does not have meta-tag '"<<m_vKeywords[i]<<"'. It might not work on all configurations!!");
			//SG_AssertFile(keyword_pos!=std::string::npos, name, "Can't meta compile with tag '"+ _KeyVector[i] + "' and value '" + _ValVector[i] + "'");
		}
		else
			m_filebuffer.insert(keyword_pos + m_vKeywords[i].size() + 8, " "+m_vValues[i]);
	}


	GPUCV_DEBUG("++++++++++++++++++++++++");
	GPUCV_DEBUG("++++++++++++++++++++++++");
	GPUCV_DEBUG("Shader --"<< GetID() << "-- source code");
	GPUCV_DEBUG(m_filebuffer);
	GPUCV_DEBUG("++++++++++++++++++++++++");
	GPUCV_DEBUG("++++++++++++++++++++++++");
}

//================================================
const std::string  ShaderDsc::GetFullPath()const
{
	std::string FilePath = GetGpuCVSettings()->GetShaderPath();
	FilePath += GetPath();
	FilePath += GetFileName();
	FilePath += ".";
	FilePath += GetExtension();
	return FilePath;
}

//================================================
#if _GPUCV_DEPRECATED
void ShaderDsc::ReadFileTime()
{
	struct stat attrib;			// create a file attribute structure
	//vertex FILE
	std::string & path = GetFullPath();
	stat(path.data(), &attrib);		// get the attributes of a file
	FileClock = *gmtime(&(attrib.st_mtime));
}
//================================================
bool ShaderDsc::IsFileChanged(void)
{
	if (isFileModified(GetFullPath().data(), &FileClock))
		return true;
	return false;
}
#endif
//================================================
/*virtual */
std::ostringstream & ShaderDsc::operator << (std::ostringstream & _stream)const
{
	std::string TypeStr;
	switch(m_type)
	{
	case GEOMETRY_SHADER:
		TypeStr = "GEOMETRY SHADER==============";
		break;
	case VERTEX_SHADER:
		TypeStr = "VERTEX SHADER==============";
		break;
	case FRAGMENT_SHADER:
		TypeStr = "FRAGMENT SHADER==============";
		break;
	default:
		TypeStr = "Unkown value ==============";
		break;
	}
	_stream << LogIndent() << "======================================" << std::endl;
	_stream << LogIndent() << TypeStr << "==============" << std::endl;
	LogIndentIncrease();
	_stream << LogIndent() << "m_file:\t"		<< GetFileName()<< std::endl;
	_stream << LogIndent() << "m_extension:\t"	<< GetExtension()	<< std::endl;
	_stream << LogIndent() << "m_path:\t"		<< GetPath()		<< std::endl;
	_stream << LogIndent() << "full path:\t"	<< GetFullPath()<< std::endl;

	_stream << LogIndent() << "m_meta_tag:\t"	<< m_meta_tag	<< std::endl;
	_stream << LogIndent() << "Meta tags====" << std::endl;
	LogIndentIncrease();
	for (unsigned int i =0; i < m_vKeywords.size() && i < m_vValues.size(); i++)
	{
		_stream << LogIndent() <<"\t" << m_vKeywords[i] << " = " << m_vValues[i]<<std::endl;
	}
	LogIndentDecrease();
	_stream << LogIndent() << "Meta tags===="<< std::endl;;
	LogIndentDecrease();
	_stream << LogIndent() << TypeStr		<< std::endl;

	PrintLocalLogInfo();
	return _stream;
}
//=================================================================
int	 ShaderDsc::Load()
{
	//load file
	if (!textFileRead(GetFullPath(), m_filebuffer))
		return false;

	//parse and replace meta tags
	if (m_vKeywords.size())
		ParseShaderCode();

	return SGE::CL_BASE_OBJ_FILE::Load();//read file time.
}
//=================================================================
GLhandleARB	 ShaderDsc::CompileShader()
{
	if(m_filebuffer!="")
	{
		_GPUCV_CLASS_GL_ERROR_TEST();
		m_shaderID = glCreateShaderObjectARB(m_type);
		SG_Assert(m_shaderID>0, "Could not create Shader object");

		char * FileSrc = (char *)m_filebuffer.data();
		glShaderSourceARB(m_shaderID,1, (const GLcharARB**)(&FileSrc),NULL);
		glCompileShaderARB(m_shaderID);

		bool ShaderError=false;

		//look for errors
		int length=0;
		glGetObjectParameterivARB(m_shaderID, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
		if (length>1)
		{
			ShaderError = true;
			PrintLocalLogInfo();
		}
		m_filebuffer="";
		_GPUCV_CLASS_GL_ERROR_TEST();
		return m_shaderID;
	}
	return 0;
}
void ShaderDsc :: PrintLocalLogInfo()const
{
	int length=0;
	char * infoLog = NULL;
	int charsWritten  = 0;
	glGetObjectParameterivARB(m_shaderID, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
	if (length>1)
	{
		infoLog = (char *)malloc(length);
		
		SG_AssertMemory(infoLog, "Could not malloc memory");
		glGetInfoLogARB(m_shaderID, length, &charsWritten, infoLog);
		std::string strInfoLog = infoLog;
		//check if there is one error:
		if(strInfoLog.find("error")!= std::string::npos)
		{//print it.
			GPUCV_ERROR("Infolog:"<< GetID());
			GPUCV_ERROR(infoLog);
		}
#ifdef _DEBUG
		else
		{//print it.
			GPUCV_WARNING("Infolog:"<< GetID());
			GPUCV_WARNING(infoLog);
		}
#endif
		free(infoLog);
	}
	//	GPUCV_WARNING("Warning : no Infolog to print\n");
}
//=================================================================
const std::string  & ShaderDsc ::GetMetaTags(void)const
{return m_meta_tag;}
//=================================================================
const ShaderDsc ::SHADER_TYPE  & ShaderDsc ::GetType(void)const
{return m_type;}
//=================================================================
const GLuint  & ShaderDsc ::GetGeoInputType(void)const
{return m_inputGeometryType;}
//=================================================================
const GLuint  & ShaderDsc ::GetGeoOutputType(void)const
{return m_outputGeometryType;}
//=================================================================


//=================================================================
#endif//ShaderDsc
//=================================================

}//namespace GCV

