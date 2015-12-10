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
#include <GPUCVCore/GpuShaderObject.h>

namespace GCV{

#if !_GPUCV_SUPPORT_GS
void
ShaderObject ::SplitFileName(const std::string _srcName,
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
#endif

ShaderObject :: ShaderObject(GpuHandleManager * s)
:
SGE::CL_BASE_OBJ<std::string>("")
,CL_Profiler("ShaderObject")
//name(),
#if _GPUCV_SUPPORT_GS
,m_vertex_shader(NULL)
,m_fragment_shader(NULL)
,m_geometry_shader(NULL)
#else
,vertex_file(),
fragment_file(),
Vkeywords(),
Vvalues(),
Fkeywords(),
Fvalues(),
#endif
,fifo_pos(-1)
,is_loaded(false)
,stack(s)
#if _GPUCV_CORE_CHECK_SHADER_CHANGE
#if !_GPUCV_SUPPORT_GS
,VertexClock()
, FragClock()
#endif
#endif
{
}
//=================================================
ShaderObject :: ~ShaderObject()
{
	if (fifo_pos != -1 && stack!=NULL)
		stack->RemoveHandle(fifo_pos);
	if(m_geometry_shader)
		delete m_geometry_shader;
	if(m_vertex_shader)
		delete m_vertex_shader;
	if(m_fragment_shader)
		delete m_fragment_shader;
}
//=================================================
/*static*/
//=================================================
#if !_GPUCV_SUPPORT_GS
void ShaderObject :: ParseMetaTag(string _tags,vector <string> & _KeyVector, vector <string> & _ValVector)
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
		value = (GetHardProfile()->GetTextType() ==  GL_TEXTURE_2D)? "texture2D":"textureRect";

	_KeyVector.push_back(key);
	_ValVector.push_back(value);

	//default Meta for all shaders to specify image type
	key = "IMGTYPE";
	if(ProcessingGPU()->IsHardwareFamily(GenericGPU::HRD_FAM_ATI))
	{//ATI only support sampler2D now? samplerRect does not work with new drivers on my ATI radeon 9700.
		value = "uniform sampler2D";
	}
	else
		value = (GetHardProfile()->GetTextType() ==  GL_TEXTURE_2D)? "uniform sampler2D":"uniform samplerRect";

	_KeyVector.push_back(key);
	_ValVector.push_back(value);

	//default Meta to specify that we are compiling with GPUCV and not "Shader designer" app
	key = "GPUCV_FILTER";
	value = "1//";
	_KeyVector.push_back(key);
	_ValVector.push_back(value);

	//parse all other meta tag
	string params = _tags;
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
		_KeyVector.push_back(key);
		_ValVector.push_back(value);

		if ((equal_pos != dollar_pos) && (dollar_pos == std::string::npos))
			break;
		else
			params= params.substr(dollar_pos, params.size() - dollar_pos+1);
	}
}
#endif
//=================================================
#if !_GPUCV_SUPPORT_GS
void ShaderObject :: ParseShaderCode(string &codevalue, vector <string> & _KeyVector, vector <string> & _ValVector)
{
	// if keywords and values were stored when parsing filename
	// they are used and add directly in shader code before it'll be compiled.
	// those modifications are set in program memory, not directly in shader code file  !
	string Msg;
	size_t keyword_pos;
	//remove license text...
	keyword_pos = codevalue.find("CVG_LicenseEnd");
	if(keyword_pos!=std::string::npos)//license found
		codevalue.erase(0, keyword_pos+strlen("CVG_LicenseEnd"));

	//add meta keys and values
	for (int i=(int)_KeyVector.size(); i--;)
	{
		keyword_pos = codevalue.find("#define "+_KeyVector[i]);
		if(keyword_pos==std::string::npos)
		{
			GPUCV_WARNING("shader file '"<< this->GetIDStr() << "' does not have meta-tag '"<<_KeyVector[i]<<"'. It might not work on all configurations!!");
			//SG_AssertFile(keyword_pos!=std::string::npos, name, "Can't meta compile with tag '"+ _KeyVector[i] + "' and value '" + _ValVector[i] + "'");
		}
		else
			codevalue.insert(keyword_pos + _KeyVector[i].size() + 8, " "+_ValVector[i]);
	}


	GPUCV_DEBUG("++++++++++++++++++++++++");
	GPUCV_DEBUG("++++++++++++++++++++++++");
	GPUCV_DEBUG("Shader --"<< GetID() << "-- source code");
	GPUCV_DEBUG(codevalue);
	GPUCV_DEBUG("++++++++++++++++++++++++");
	GPUCV_DEBUG("++++++++++++++++++++++++");
}
#endif
//=================================================
#if _GPUCV_SUPPORT_GS
string ShaderObject :: cvLoadShaders(ShaderProgramNames & _shaderNames)
#else
string ShaderObject :: cvLoadShaders(string _fragmentShaderFilename, string _vertexShaderFilename)
#endif
{
	if (is_loaded)
		return GetID();

#if _GPUCV_DEBUG_MODE
	//SG_Assert(_filename1 != "", "Empty filename");
#endif
#if _GPUCV_SUPPORT_GS
	//std::string ShaderNames[]={_fragmentShaderFilename,_vertexShaderFilename};
	//take first name
	ShaderDsc * TmpShaderDsc = NULL;

	//delete existing objects
	if(m_geometry_shader)
		delete m_geometry_shader;
	if(m_vertex_shader)
		delete m_vertex_shader;
	if(m_fragment_shader)
		delete m_fragment_shader;

	//for all shaders file, we attributes them to correct type
	for(int i =0; i < sizeof(_shaderNames.m_ShaderNames)/sizeof(std::string); i++)
	{
		TmpShaderDsc = new ShaderDsc(_shaderNames.m_ShaderNames[i]);
		switch (TmpShaderDsc->GetType())
		{
		case ShaderDsc::GEOMETRY_SHADER:
			SG_Assert(m_geometry_shader==NULL, "We have already a geometry shader defined");
			m_geometry_shader = TmpShaderDsc;
			break;
		case ShaderDsc::VERTEX_SHADER:
			SG_Assert(m_vertex_shader==NULL, "We have already a vertex shader defined");
			m_vertex_shader = TmpShaderDsc;
			break;
		case ShaderDsc::FRAGMENT_SHADER:
			SG_Assert(m_fragment_shader==NULL, "We have already a fragment shader defined");
			m_fragment_shader = TmpShaderDsc;
			break;
		default:
			delete TmpShaderDsc;
			//SG_AssertFile(0, _shaderNames.m_ShaderNames[i], "Error while getting shader file and extension");
		}
	}

	std::string TmpName=GenerateShaderUniqueName(_shaderNames);
	/*
	int i;
	for(i =0; i < sizeof(_shaderNames.m_ShaderNames)/sizeof(std::string)-1; i++)
	{
	if(_shaderNames.m_ShaderNames[i]!="")
	{
	TmpName+= _shaderNames.m_ShaderNames[i];
	TmpName+= std::string ("#_#");
	}
	}
	TmpName += _shaderNames.m_ShaderNames[i];
	*/

	SetID(TmpName);

	if(m_fragment_shader==NULL)
		m_fragment_shader = new ShaderDsc("FShaders/default.frag");
	if(m_vertex_shader==NULL)
		m_vertex_shader = new ShaderDsc("VShaders/default.vert");
	//no default one for geomatry

	if(m_vertex_shader)
		m_vertex_shader->Load();
	if(m_fragment_shader)
		m_fragment_shader->Load();
	if(m_geometry_shader)
		m_geometry_shader->Load();


#else
	string file1, meta1;
	string file2, meta2;
	GLenum shader_type1;
	GLenum shader_type2;
	string Msg;

	//filename 1
	LoadShaderFile(_fragmentShaderFilename, file1, meta1, shader_type1);
	SG_Assert(shader_type1!=(GLenum)FILE_ERROR && shader_type1 !=NOT_SHADER,
		"Error loading shader file 1");
	SG_Assert(shader_type1==FRAGMENT_SHADER || shader_type1 == UNDEFINE, "File 1 not a fragment shader");

	//filename 2
	LoadShaderFile(_vertexShaderFilename, file2, meta2, shader_type2);
	SG_Assert(shader_type2!=(GLenum)FILE_ERROR && shader_type2 !=NOT_SHADER,
		"Error loading shader file 2");
	SG_Assert(shader_type2==VERTEX_SHADER || shader_type2 == UNDEFINE, "File 2 not a vertex shader");

	//SG_Assert(shader_type2 != shader_type1, "Both shader files have same type.");
	//SG_AssertFile(CL_BASE_OBJ_FILE::FileExists(file1), "File missing");
	//SG_AssertFile(CL_BASE_OBJ_FILE::FileExists(file2), "File missing");

	if( shader_type1 == UNDEFINE)
	{
		fragment_file = fragment_name = "FShaders/default.frag";
		_fragmentShaderFilename = "";//reset it cause it is default.
	}
	else
	{
		fragment_name = _fragmentShaderFilename;
		fragment_file = file1;
		fragment_meta_tag = meta1;
	}

	if( shader_type2 == UNDEFINE)
	{
		vertex_name = vertex_file = "VShaders/default.vert";
		_vertexShaderFilename = "";//reset it cause it is default.
	}
	else
	{
		vertex_name = _vertexShaderFilename;
		vertex_file = file2;
		vertex_meta_tag = meta2;
	}
#endif

#if _GPUCV_SUPPORT_GS
	//read shader meta tags
	//=> done when creating ShaderDsc
	/*
	if(m_vertex_shader)
	m_vertex_shader->ParseMetaTag();
	if(m_fragment_shader)
	m_fragment_shader->ParseMetaTag();
	*/
	//moved up before we affect default values for shader name
	/*
	std::string TmpName="";
	int i;
	for(i =0; i < sizeof(_shaderNames.m_ShaderNames)/sizeof(std::string)-1; i++)
	{
	TmpName+= _shaderNames.m_ShaderNames[i];
	TmpName+= std::string ("#_#");
	}
	TmpName += _shaderNames.m_ShaderNames[i];
	SetID(TmpName);
	*/
#else
	ParseMetaTag(vertex_meta_tag, Vkeywords, Vvalues);
	ParseMetaTag(fragment_meta_tag, Fkeywords, Fvalues);

#if _GPUCV_CORE_USE_FULL_SHADER_NAME
	SetID(_fragmentShaderFilename + std::string ("#_#") + _vertexShaderFilename);
#else
	SetID(_filename1.substr(pos_rep_sep, _filename1.size()-pos_rep_sep-5));
#endif
#endif


#if _GPUCV_CORE_CHECK_SHADER_CHANGE
#if _GPUCV_SUPPORT_GS
	/*
	if(m_vertex_shader)
	m_vertex_shader->ReadFileTime();
	if(m_fragment_shader)
	m_fragment_shader->ReadFileTime();
	if(m_geometry_shader)
	m_geometry_shader->ReadFileTime();
	*/
#else
	//we get the stats of the files...
	struct stat attrib;			// create a file attribute structure
	//vertex FILE
	std::string FilePath = GetGpuCVSettings()->GetShaderPath() + vertex_file;
	stat(FilePath.data(), &attrib);		// get the attributes of afile.txt
	VertexClock = *gmtime(&(attrib.st_mtime));

	//fragment file
	FilePath = GetGpuCVSettings()->GetShaderPath() + fragment_file;
	stat(FilePath.data(), &attrib);		// get the attributes of afile.txt
	FragClock = *gmtime(&(attrib.st_mtime));
#endif
#endif
	return GetID();

	// get type of each file.
	// Type is defined by extension : .frag -> fragment shader, .vert -> vertex shader
}
/*static*/
std::string ShaderObject ::GenerateShaderUniqueName(ShaderProgramNames & _shaderNames)
{
	std::string TmpName="";
	int i;
	for(i =0; i < sizeof(_shaderNames.m_ShaderNames)/sizeof(std::string)-1; i++)
	{
		if(_shaderNames.m_ShaderNames[i]!="")
		{
			TmpName+= _shaderNames.m_ShaderNames[i];
			TmpName+= std::string ("#_#");
		}
	}
	TmpName += _shaderNames.m_ShaderNames[i];
	return TmpName;
}
//=================================================
/*string ShaderObject :: GetName()
{
return name;
}
*/
//=================================================
const std::string  ShaderObject :: GetVertexFile()const
{
#if _GPUCV_SUPPORT_GS
	if(m_vertex_shader)
		return m_vertex_shader->GetID();
	else
		return "";
#else
	return vertex_file;
#endif
}
//=================================================
const std::string  ShaderObject :: GetFragmentFile()const
{
#if _GPUCV_SUPPORT_GS
	if(m_fragment_shader)
		return m_fragment_shader->GetID();
	else
		return "";
#else
	return fragment_file;
#endif
}

const std::string  ShaderObject :: GetGeometryFile()const
{
#if _GPUCV_SUPPORT_GS
	if(m_geometry_shader)
		return m_fragment_shader->GetID();
	else
		return "";
#else
	return fragment_file;
#endif
}

//=================================================
void ShaderObject :: SetQueueIndex(int pos)
{
	fifo_pos = pos;
}
//=================================================
void ShaderObject :: SetIsLoaded(bool status)
{
	is_loaded = status;
}
//=================================================
#if _GPUCV_CORE_CHECK_SHADER_CHANGE
bool ShaderObject::AreFilesChanged(void)
{
#if _GPUCV_SUPPORT_GS
	if(m_vertex_shader)
		if(m_vertex_shader->IsFileModified())
			return true;
	if(m_fragment_shader)
		if(m_fragment_shader->IsFileModified())
			return true;
	if(m_geometry_shader)
		if(m_geometry_shader->IsFileModified())
			return true;
#else
	//if (VertexClock)
	if (isFileModified(GetGpuCVSettings()->GetShaderPath() + vertex_file, &VertexClock))
		return true;

	//if (FragClock)
	if (isFileModified(GetGpuCVSettings()->GetShaderPath() + fragment_file, &FragClock))
		return true;
#endif
	return false;
}
#endif
//=================================================
#if !_GPUCV_SUPPORT_GS
bool ShaderObject:: LoadShaderFile(string _name,  std::string & _filename, std::string & _metatag, GLenum &_shader_type)
{
	if(_name=="")
	{
		_shader_type=UNDEFINED;
		return false;
	}

	string fileName, filePath, fileExt;
	SplitFileName(_name, filePath, fileName, fileExt, _metatag);
	_filename = filePath + fileName +"."+ fileExt;


	_shader_type = ((fileExt) == "vert") ? VERTEX_SHADER
		: ((fileExt) == "frag") ? FRAGMENT_SHADER
		: ((fileExt) == "old") ? NOT_SHADER
		: ((fileExt) == "CVS") ? NOT_SHADER
		: ((fileExt) == "SVN") ? NOT_SHADER
		: FILE_ERROR;

	return true;
}
#endif
//=================================================
GLhandleARB ShaderObject :: GetShaderHandle()
{
	_GPUCV_CLASS_GL_ERROR_TEST();
	if (is_loaded)
	{
		GLhandleARB handle = stack->SetFirst(fifo_pos);
		return handle;
	}
	else
	{
#if _GPUCV_SUPPORT_GS
		/*
		if(m_vertex_shader)
		m_vertex_shader->LoadFile();
		if(m_fragment_shader)
		m_fragment_shader->LoadFile();
		if(m_geometry_shader)
		m_geometry_shader->LoadFile();
		*///moved into cvLoadShaders
#else
		string loaded_shader;
		string second_shader;
		char* vertex_source;
		char* fragment_source;

		string file1, file2;

		int length=0;
		//int charlength=0;

		char tmp[256];
		char tmp2[256];
		strcpy(tmp, vertex_file.data());

		if (!textFileRead(tmp, loaded_shader))
			return 0;

		if (Vkeywords.size())
			ParseShaderCode(loaded_shader, Vkeywords, Vvalues);

		vertex_source = new char[loaded_shader.size()+1];
		strcpy(vertex_source, loaded_shader.data());

		strcpy(tmp2, fragment_file.data());

		if (!textFileRead(tmp2, second_shader))
			return 0;

		if (Fkeywords.size())
			ParseShaderCode(second_shader, Fkeywords, Fvalues);
		fragment_source = new char[second_shader.size()+1];
		strcpy(fragment_source, second_shader.data());
#endif

#if _GPUCV_SUPPORT_GS
		if(1)
#else
		if(vertex_source && fragment_source)
#endif
		{
			_GPUCV_CLASS_GL_ERROR_TEST();

			GLhandleARB vertex_program=0, fragment_program=0, geometry_program=0;
#if _GPUCV_SUPPORT_GS
			if(m_vertex_shader)
				vertex_program = m_vertex_shader->CompileShader();
			if(m_fragment_shader)
				fragment_program = m_fragment_shader->CompileShader();
			if(m_geometry_shader)
				geometry_program = m_geometry_shader->CompileShader();
#else


			vertex_program = glCreateShaderObjectARB(VERTEX_SHADER);
			fragment_program = glCreateShaderObjectARB(FRAGMENT_SHADER);

			// load vertex code
			glShaderSourceARB(vertex_program,1, (const GLcharARB**)(&vertex_source),NULL);
			glCompileShaderARB(vertex_program);

			bool ShaderError=false;
			//look for errors
			glGetObjectParameterivARB(vertex_program, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
			if (length>1)
				ShaderError = true;

			// load fragment code
			glShaderSourceARB(fragment_program,1,(const GLcharARB**)(&fragment_source),NULL);
			glCompileShaderARB(fragment_program);

			//look for errors
			glGetObjectParameterivARB(fragment_program, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
			if (length>1)
			{
				ShaderError = true;
			}


			if(ShaderError || GetGpuCVSettings()->GetOption(GpuCVSettings::GPUCV_SETTINGS_FILTER_DEBUG))
				PrintInfo(true, vertex_program, fragment_program);

			delete [] vertex_source;
			delete [] fragment_source;
#endif


			GLhandleARB shader_handle = glCreateProgramObjectARB();

			if(vertex_program)
				glAttachObjectARB(shader_handle,vertex_program);
			if(fragment_program)
				glAttachObjectARB(shader_handle,fragment_program);
			if(geometry_program)
				glAttachObjectARB(shader_handle,geometry_program);

			_GPUCV_CLASS_GL_ERROR_TEST();

#if _GPUCV_SUPPORT_GS
			//for geometry shader we need to define geometry type.
			if(m_geometry_shader)
			{
				/*SRC:http://cirl.missouri.edu/gpu/glsl_lessons/glsl_geometry_shader/index.html
				We then specify the geometry input and output primitive types.
				We will pass in GL_LINES, and will output GL_LINE_STRIPs from the geometry program.
				I think you can use any of the OpenGL primitive types as the third parameter to the first function below, plus four new types specified by geometry shaders (don't know if they are isolated to geometry shaders). You can read about these types in the OpenGL documentation from the NVIDIA link we provided at the top! I dont like that you have to specify the types, but my tears dont drive hardware!
				We then get the maximum number of vertices that a geometry program can output, and pass it to the GPU. We think our method is inefficent, i.e. we should pass the real number of vertices we expect to generate. We are waiting to find out if overspecifying the number is bad!
				*/
				glProgramParameteriEXT(shader_handle,GL_GEOMETRY_INPUT_TYPE_EXT,	m_geometry_shader->GetGeoInputType());
				glProgramParameteriEXT(shader_handle,GL_GEOMETRY_OUTPUT_TYPE_EXT,	m_geometry_shader->GetGeoOutputType());

				int temp;
				glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT,&temp);
				glProgramParameteriEXT(shader_handle,GL_GEOMETRY_VERTICES_OUT_EXT,temp);
			}
#endif
			glLinkProgramARB(shader_handle);

			if(vertex_program)
				glDeleteObjectARB(vertex_program);
			if(fragment_program)
				glDeleteObjectARB(fragment_program);
			if(geometry_program)
				glDeleteObjectARB(geometry_program);
			_GPUCV_CLASS_GL_ERROR_TEST();

			if(stack)
				stack->cvgPush(shader_handle, vertex_program, fragment_program, this);

			/* is it useful for something...???
			for (int i=18; i--;cout << char(8) << flush);
			for (int i=18; i--;cout << ' ' << flush);
			for (int i=18; i--;cout << char(8) << flush);
			*/

			return shader_handle;
		}
		else
		{
			GPUCV_WARNING("Warning : Read Error on shader files("+GetVertexFile()+':'+GetFragmentFile()+")\n");
			return 0;
		}
	}

}
//=================================================
std::ostringstream &  ShaderObject ::operator << (std::ostringstream & _stream)const
{
	_stream << LogIndent() <<"======================================" << std::endl;
	_stream << LogIndent() <<"ShaderObject==============" << std::endl;
	LogIndentIncrease();
	_stream << LogIndent() <<"fifo_pos:\t"<< fifo_pos << std::endl;
	if(m_geometry_shader)	_stream << *m_geometry_shader;
	if(m_vertex_shader)		_stream << *m_vertex_shader;
	if(m_fragment_shader)	_stream << *m_fragment_shader;
	LogIndentDecrease();
	_stream << LogIndent() <<"ShaderObject==============" << std::endl;
	_stream << LogIndent() <<"======================================" << std::endl;


	return _stream;
}
void ShaderObject :: PrintInfo(bool _shaderCode, GLhandleARB vertex_program, GLhandleARB fragment_program)
{
	if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_FILTER_DEBUG))
		GetGpuCVSettings()->PushSetOptions(
		GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG
		|GpuCVSettings::GPUCV_SETTINGS_GLOBAL_NOTICE
		|GpuCVSettings::GPUCV_SETTINGS_GLOBAL_ERROR
		|GpuCVSettings::GPUCV_SETTINGS_GLOBAL_WARNING
		,true);


	int length=0;
	GPUCV_DEBUG("\n\n>>Shader general info ====================================");
#if _GPUCV_SUPPORT_GS
	if(m_geometry_shader)
		GPUCV_DEBUG(*m_geometry_shader)
	if(m_vertex_shader)
		GPUCV_DEBUG(*m_vertex_shader)
	if(m_fragment_shader)
		GPUCV_DEBUG(*m_fragment_shader)
#else
	if(vertex_program)
	{
		glGetObjectParameterivARB(vertex_program, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
		if (length>1)
		{
			//cout << endl << Vertex_source << endl;
			GPUCV_ERROR(name <<" FRAGMENT PROGRAM : infolog !");
			printInfoLog(vertex_program);
		}
	}
	if(fragment_program)
	{
		glGetObjectParameterivARB(fragment_program, GL_OBJECT_INFO_LOG_LENGTH_ARB, &length);
		if (length>1)
		{
			//cout << endl << fragment_source << endl;
			cerr << endl << name <<" FRAGMENT PROGRAM : infolog !" << endl;
			printInfoLog(fragment_program);
			cout << endl;
		}
	}

	GPUCV_DEBUG("\tShader full name: " + GetIDStr());
	GPUCV_DEBUG("\tVertex shader name: " + GetVertexFile());
	GPUCV_DEBUG("\tFragment shader name: " + GetFragmentFile());


	if(Vkeywords.size()>0)
	{
		GPUCV_DEBUG("\n\t===Vertex shader meta info===");
		for (unsigned int i =0; i < Vkeywords.size() && i < Vvalues.size(); i++)
		{
			GPUCV_DEBUG("\t" + Vkeywords[i] + " = " + Vvalues[i]);
		}
	}
	if(Vkeywords.size()>0)
	{
		GPUCV_DEBUG("\n\t===Fragment shader meta info===");
		for (unsigned int i = 0; i < Fkeywords.size() && i < Fvalues.size(); i++)
		{
			GPUCV_DEBUG("\t" + Fkeywords[i] + " = " + Fvalues[i]);
		}
	}
#endif
	GPUCV_DEBUG("\tFIFO Position: " + fifo_pos);
	GPUCV_DEBUG("\tLoaded: " << (std::string((is_loaded)? "true":"false")));
	GPUCV_DEBUG("\n<<Shader general info ====================================\n\n");
	if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_FILTER_DEBUG))
		GetGpuCVSettings()->PopOptions();
}
//=================================================
#if 0
int printInfoLog(GLhandleARB obj)
{
	int infologLength = 0;
	int charsWritten  = 0;
	char *infoLog;

	if (!(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG)
		|| GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_ERROR)))
		return true;


	glGetObjectParameterivARB(obj, GL_OBJECT_INFO_LOG_LENGTH_ARB,
		&infologLength);

	if (infologLength > 0)
	{
		infoLog = (char *)malloc(infologLength);
		if(infoLog != NULL)
		{
			glGetInfoLogARB(obj, infologLength, &charsWritten, infoLog);
			GPUCV_WARNING(infoLog);
			free(infoLog);
			return(EXIT_SUCCESS);
		}
		GPUCV_ERROR("Critical : malloc failed while printing Infolog\n");
		return(EXIT_FAILURE);
	}
	GPUCV_WARNING("Warning : no Infolog to print\n");
	return(EXIT_FAILURE);
}
#endif
//=================================================

}//namespace GCV

