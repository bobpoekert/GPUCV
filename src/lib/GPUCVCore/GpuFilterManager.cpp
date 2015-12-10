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
#include <GPUCVCore/GpuFilterManager.h>
#include <GPUCVCore/coretools.h>

namespace GCV{

//Initialize singleton
//template <> GpuFilterManager * CL_Singleton<GpuFilterManager>::m_registeredSingleton = NULL;
//=================================================
GpuFilterManager :: GpuFilterManager()
: TplManager(NULL),
m_ShaderManager(),
CL_Singleton<GpuFilterManager>()
{
}
//=================================================
GpuFilterManager :: ~GpuFilterManager()
{
}
//=================================================
#if _GPUCV_DEPRECATED
string GpuFilterManager :: GetFilterName(string _fragmentShaderFilename, string _vertexShaderFilename)
{
	string filtername = _fragmentShaderFilename + std::string("#_#") + _vertexShaderFilename;
	GpuFilter * CurrFilter = Get(filtername);
	if(CurrFilter)//filter already exist
		return filtername;
	else
	{//we create it
		CurrFilter = new GpuFilter();
		CurrFilter->SetShaderManager(&m_ShaderManager);

		string fpname;
		fpname = (CurrFilter->SetShadersFiles(_fragmentShaderFilename, _vertexShaderFilename));
		SG_Assert(fpname != "", "Error getting Shader program name");

		//check that no other shader have the same name...,???
		GpuFilter * CheckFilter = Get(fpname);
		SG_Assert(!CheckFilter, "DEBUG, A shader using the same name already exist...???");

		AddObj(CurrFilter);
		return fpname;
	}
}
#endif
//=================================================
#if _GPUCV_SUPPORT_GS
GpuFilter * GpuFilterManager :: GetFilterByName(std::string & file1, std::string file2/*=""*/,std::string file3/*=""*/)
{
	ShaderProgramNames Names;
	Names.m_ShaderNames[0] = file1;
	Names.m_ShaderNames[1] = file2;
	Names.m_ShaderNames[2] = file3;
	return GetFilterByName(Names);
}
//==================================================================
GpuFilter * GpuFilterManager :: GetFilterByName(ShaderProgramNames & _ShaderNames)
#else
GpuFilter * GpuFilterManager :: GetFilterByName(string _fragmentShaderFilename, string _vertexShaderFilename)
#endif
{
#if _GPUCV_SUPPORT_GS
	string filtername = ShaderObject::GenerateShaderUniqueName(_ShaderNames);
#else
	string filtername = _fragmentShaderFilename + std::string("#_#") + _vertexShaderFilename;
#endif
	GpuFilter * CurrFilter = Get(filtername);
	if(CurrFilter)//filter already exist
		return CurrFilter;
	else
	{//we create it
		CurrFilter = new GpuFilter();
		CurrFilter->SetShaderManager(&m_ShaderManager);

		string fpname;
#if _GPUCV_SUPPORT_GS
		fpname = CurrFilter->SetShadersFiles(_ShaderNames);
#else
		fpname = CurrFilter->SetShadersFiles(_fragmentShaderFilename, _vertexShaderFilename);
#endif
		SG_Assert(fpname != "", "Error getting Shader program name");

		//check that no other shader have the same name...,???
		GpuFilter * CheckFilter = Get(fpname);
		SG_Assert(!CheckFilter, "DEBUG, A shader using the same name already exist...???");

		AddObj(CurrFilter);
		return CurrFilter;
	}
}
//=================================================
void TemplateOperator(const std::string &_fctName, const std::string &_filename1, const std::string &_filename2,
					  DataContainer * _src1, DataContainer * _src2, DataContainer * _src3,
					  DataContainer *_dest,
					  const float *_params/*=NULL*/, unsigned int _param_nbr/*=0*/,
					  TextureGrp::TextureGrp_CheckFlag _controlFlag/*=0*/, std::string _optionalMetaTag/*=""*/,
					  FCT_PTR_DRAW_TEXGRP(_DrawFct)/*=NULL*/)
{
	SG_Assert(_dest && _src1, "No input or ouput image");

	static TextureGrp InputGrp;
	static TextureGrp OutputGrp;
	InputGrp.Clear();
	OutputGrp.Clear();

	InputGrp.SetGrpType(TextureGrp::TEXTGRP_INPUT);
	OutputGrp.SetGrpType(TextureGrp::TEXTGRP_OUTPUT);

	InputGrp.AddTextures(&_src1,1);
	if(_src2)
		InputGrp.AddTextures(&_src2,1);
	if(_src3)
		InputGrp.AddTextures(&_src3,1);
	OutputGrp.AddTextures(&_dest,1);

	TemplateOperator(_fctName, _filename1, _filename2,
		&InputGrp,
		&OutputGrp,
		_params, _param_nbr,
		_controlFlag, _optionalMetaTag,
		_DrawFct);
}
//==================================================================
#if _GPUCV_SUPPORT_GS
void TemplateOperatorGeo(const std::string & _fctName, ShaderProgramNames & _ShaderNames,
						 TextureGrp * _srcGrp, TextureGrp * _dstGrp,
						 const float *_params/*=NULL*/, unsigned int _param_nbr/*=0*/,
						 TextureGrp::TextureGrp_CheckFlag _controlFlag/*=0*/, std::string _optionalMetaTag/*=""*/,
						 FCT_PTR_DRAW_TEXGRP(_DrawFct)/*=NULL*/)
{

	//manage groups
	SG_Assert(_srcGrp && _dstGrp, "No input or ouput image");

	//set type
	_srcGrp->SetGrpType(TextureGrp::TEXTGRP_INPUT);
	_dstGrp->SetGrpType(TextureGrp::TEXTGRP_OUTPUT);

	//set check flags
	_srcGrp->SetControlFlag(_controlFlag);
	//_dstGrp->SetControlFlag(_controlFlag);??
	SG_Assert(_srcGrp->CheckControlFlag(), "CvArr properties does not match");

	// for choosing the particular shader
	//string chosen_filter;
	if(_optionalMetaTag!="")
	{
		SG_Assert(0, "TemplateOperator():We have metaTags but we are using _GPUCV_SUPPORT_GS");
	}

	GpuFilter * CurrentFilter = GetFilterManager()->GetFilterByName(_ShaderNames);
	SG_AssertFile(CurrentFilter, _ShaderNames.m_ShaderNames[0]+" & "+_ShaderNames.m_ShaderNames[1]+" & "+_ShaderNames.m_ShaderNames[2], "No filter found");

	//passing the parameters to the shader
	CurrentFilter->SetParams(_params, _param_nbr);

	//This applies the shader and outputs the result
	CurrentFilter->Apply(
		_srcGrp/*InputGrp*/,
		_dstGrp/*Output Image*/,
		_dstGrp->GetTex(0)->GetDataDsc<DataDsc_GLTex>()/*Size of output image*/,
		_DrawFct
		);
}
#endif
void TemplateOperator(const std::string & _fctName, const std::string & _filename1, const std::string & _filename2,
					  TextureGrp * _srcGrp, TextureGrp * _dstGrp,
					  const float *_params/*=NULL*/, unsigned int _param_nbr/*=0*/,
					  TextureGrp::TextureGrp_CheckFlag _controlFlag/*=0*/, std::string _optionalMetaTag/*=""*/,
					  FCT_PTR_DRAW_TEXGRP(_DrawFct)/*=NULL*/)
{
#if 0//_GPUCV_SUPPORT_GS

	if(_optionalMetaTag!="")
	{
		if(_filename1!="")
			_filename1+= "(" + _optionalMetaTag + ").frag";
		if(_filename2!="")
			_filename2+= "(" + _optionalMetaTag + ").vert";
	}
	else
	{
		if(_filename1!="")
			_filename1+= ".frag";
		if(_filename2!="")
			_filename2+= ".vert";
	}
	ShaderProgramNames Names;
	Names.m_ShaderNames[0] = _filename1;
	Names.m_ShaderNames[1] = _filename2;
	Names.m_ShaderNames[2] = "";
	TemplateOperatorGeo(_fctName, Names, _srcGrp, _dstGrp, _params,_param_nbr,
		_controlFlag, _optionalMetaTag, _DrawFct);

#else
	//manage groups
	SG_Assert(_srcGrp && _dstGrp, "No input or ouput image");

	//set type
	_srcGrp->SetGrpType(TextureGrp::TEXTGRP_INPUT);
	_dstGrp->SetGrpType(TextureGrp::TEXTGRP_OUTPUT);

	//set check flags
	_srcGrp->SetControlFlag(_controlFlag);
	//_dstGrp->SetControlFlag(_controlFlag);??
	SG_Assert(_srcGrp->CheckControlFlag(), "CvArr properties does not match");

	// for choosing the particular shader
	//string chosen_filter;
	std::string File1, File2;
	if(_optionalMetaTag!="")
	{
		if(_filename1!="")
			File1 = _filename1 + "(" + _optionalMetaTag + ").frag";
		if(_filename2!="")
			File2 = _filename2 + "(" + _optionalMetaTag + ").vert";
	}
	else
	{
		if(_filename1!="")
			File1 = _filename1+ ".frag";
		if(_filename2!="")
			File1 = _filename2+ ".vert";
	}

	GpuFilter * CurrentFilter = GetFilterManager()->GetFilterByName(File1 , File2);
	SG_AssertFile(CurrentFilter, File1 + "&" + File2,  "No filter found");

	//passing the parameters to the shader
	CurrentFilter->SetParams(_params, _param_nbr);

	//This applies the shader and outputs the result
	CurrentFilter->Apply(
		_srcGrp/*InputGrp*/,
		_dstGrp/*Output Image*/,
		_dstGrp->GetTex(0)->GetDataDsc<DataDsc_GLTex>()/*Size of output image*/,
		_DrawFct
		);
#endif
}
//=================================================

}//namespace GCV

