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
#include <GPUCVCore/GpuShaderManager.h>


namespace GCV{
//=================================================
ShaderManager :: ShaderManager() :
#if ShaderManager_USE_MANAGER
:SGE::CL_TplObjManager<ShaderObject, std::string>(NULL)
,CL_Singleton<ShaderManager>()
#else
manager(),
#endif
stack()
{
}
//=================================================
ShaderManager :: ~ShaderManager()
{
#if ShaderManager_USE_MANAGER
	DeleteAll();
#else
	for (int i=(int)manager.size(); i--; manager[i]->~ShaderObject());
	manager.clear();
#endif
}
//=================================================
int ShaderManager :: FindByFilename(string filename)
{
#if ShaderManager_USE_MANAGER
	Find(filename);
#else
	for (int i=(int)manager.size(); i--;)
	{
		if (manager[i]->GetVertexFile() == filename) return i;
		else if (manager[i]->GetFragmentFile() == filename) return i;
	}
	return SHADER_NOT_FOUND;
#endif

}
//=================================================
int ShaderManager :: FindByName(const string & name)const
{
	for (int i=(int)manager.size(); i--;)
	{
		if (manager[i]->GetID() == name) return i;
	}


	return SHADER_NOT_FOUND;
}
//=================================================
const ShaderObject*  ShaderManager :: FindConst(const string & name)const
{
	for (int i=(int)manager.size(); i--;)
	{
		if (manager[i]->GetID() == name) return manager[i];
	}
	return NULL;
}
//=================================================
#if _GPUCV_SUPPORT_GS
string ShaderManager :: AddShader(ShaderProgramNames &_ShaderNames)
#else
string ShaderManager :: AddShader(string _fragmentShaderFilename, string _vertexShaderFilename)
#endif
{
	int shader_pos = SHADER_NOT_FOUND;

#if _GPUCV_SUPPORT_GS
	std::string TmpName	=	ShaderObject::GenerateShaderUniqueName(_ShaderNames);

	shader_pos = FindByName(TmpName);

	if(shader_pos == SHADER_NOT_FOUND)
	{//unknown shader, we create it
		ShaderObject  *new_shader = new ShaderObject(&stack);
		string name;
		name = new_shader->cvLoadShaders(_ShaderNames);

		if (name != "" && (FindByName(name) == SHADER_NOT_FOUND))
			manager.push_back(new_shader);
		else
			delete new_shader;//was already here...?????

		return name;
	}
	else
		return TmpName;
#else
	if (shader_pos != SHADER_NOT_FOUND) // if one shader is id, then test second
		shader_pos = FindByFilename(_vertexShaderFilename);


	if (shader_pos == SHADER_NOT_FOUND) // if one or more shader is different, load a new set !
	{
		ShaderObject  *new_shader = new ShaderObject(&stack);
		string name;
		name = (new_shader->cvLoadShaders(_fragmentShaderFilename, _vertexShaderFilename));

		if (name != "" && (FindByName(name) == SHADER_NOT_FOUND))
			manager.push_back(new_shader);
		else
			delete new_shader;

		return name;
	}
	else
		return string(manager[shader_pos]->GetID());
#endif
}
//=================================================
bool ShaderManager :: UpdateShader(string ShaderName)
{
	int shader_pos = FindByName(ShaderName);

	if (shader_pos == SHADER_NOT_FOUND)
		return false;//no shader to update

	//check file time
#if _GPUCV_CORE_CHECK_SHADER_CHANGE
	if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_CHECK_SHADER_UPDATE))
	{
		if(manager[shader_pos]->AreFilesChanged())
		{
			cout << "Some shader files have been modified, we reload them!" << endl;
			//we remove the shader and add it again...
#if _GPUCV_SUPPORT_GS
			ShaderProgramNames Names;
			Names.m_ShaderNames[0] = manager[shader_pos]->GetVertexFile();
			Names.m_ShaderNames[1] = manager[shader_pos]->GetFragmentFile();
			Names.m_ShaderNames[2] = manager[shader_pos]->GetGeometryFile();

			RemoveShader(ShaderName);
			if (AddShader(Names) != "")
				return true;
			else
				return false;
#else
			std::string File1, File2;
			File2 = manager[shader_pos]->GetVertexFile();
			File1 = manager[shader_pos]->GetFragmentFile();

			RemoveShader(ShaderName);
			if (AddShader(File1, File2) != "")
				return true;
			else
				return false;
#endif
		}
	}
#endif
	return true;
}
//=================================================
bool ShaderManager :: RemoveShader(string name)
{
	int shader_pos = FindByName(name);
	if (shader_pos != SHADER_NOT_FOUND)
	{
		vector<ShaderObject * >::iterator it = manager.begin();
		for(int i=0; i< shader_pos; i++ ) it++;
		(*it)->~ShaderObject();
		manager.erase(it);
		return true;
	}
	else    return false;
}
//=================================================
bool ShaderManager :: GetShader(string name, GLhandleARB &handle)
{
	int shader_pos = FindByName(name);
	if (shader_pos != SHADER_NOT_FOUND)
	{
		if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_CHECK_SHADER_UPDATE))
		{
			UpdateShader	(name);
		}
		handle = manager[shader_pos]->GetShaderHandle();
		if(handle>0)
			return true;
		else
			return false;
	}
	else
		return false;
}
//=================================================
void ShaderManager :: GetShaderFiles(string name, string &filename1, string &filename2)
{
	int shader_pos = FindByName(name);
	if (shader_pos != SHADER_NOT_FOUND)
	{
		filename1 = manager[shader_pos]->GetVertexFile();
		filename2 = manager[shader_pos]->GetFragmentFile();
	}
}
//=================================================
int ShaderManager :: GetNumberOfShaders()
{ return (int)manager.size(); }
//=================================================

}//namespace GCV

