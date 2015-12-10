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
#include <GPUCVCore/GpuFilter.h>
#include <GPUCVCore/coretools.h>
#include <GPUCVTexture/TextureTemp.h>
#include <GPUCVTexture/DataDsc_GLTex.h>


namespace GCV{

// FOR DEBUG DISPLAY : each filter displays temporary result while applying
#if _GPUCV_DEBUG_MODE
#define DEBUG_DISPLAY 0//can be changed to 1
#endif

GpuFilter :: GpuFilter()
: CL_Profiler("GpuFilter"),
SGE::CL_BASE_OBJ<std::string>(),
m_display_list(-1),
m_params(NULL),
m_nb_params(0)
{
	CLASS_FCT_SET_NAME("GpuFilter");
	CLASS_FCT_PROF_CREATE_START();
	/*   if (!glIsList(1))
	{
	glNewList(1, GL_COMPILE);

	if (GetHardProfile()->GetTextType() == GL_TEXTURE_2D)
	drawQuad();
	else
	drawQuadRect();

	glEndList();
	}
	*/
	m_params = new float[_GPUCV_SHADER_MAX_PARAM_NB];
	GpuCVInit();
}
//=================================================

GpuFilter :: GpuFilter(std::string _name)
: CL_Profiler("GpuFilter"),
SGE::CL_BASE_OBJ<std::string>(_name),
m_display_list(-1),
m_params(NULL),
m_nb_params(0)
{
	CLASS_FCT_SET_NAME("GpuFilter");
	CLASS_FCT_PROF_CREATE_START();
	/*   if (!glIsList(1))
	{
	glNewList(1, GL_COMPILE);

	if (GetHardProfile()->GetTextType() == GL_TEXTURE_2D)
	drawQuad();
	else
	drawQuadRect();

	glEndList();
	}
	*/
	m_params = new float[_GPUCV_SHADER_MAX_PARAM_NB];
	GpuCVInit();
}
//====================================
GpuFilter :: ~GpuFilter()
{
	CLASS_FCT_SET_NAME("~GpuFilter");
	CLASS_FCT_PROF_CREATE_START();
	if (m_params) delete [] m_params;
	if (m_display_list != -1)  glDeleteLists(m_display_list, 1);
	m_ShaderManager->RemoveShader(GetName());
}
//====================================
std::ostringstream &  GpuFilter :: operator << (std::ostringstream & _stream)const
{
	_stream << "======================================" << std::endl;
	_stream << LogIndent() <<"GpuFilter==============" << std::endl;
	LogIndentIncrease();
	//_stream << LogIndent() <<"m_shader_name: \t"	<< m_shader_name << std::endl;
	_stream << LogIndent() <<"m_ID: \t"	<< GetID() << std::endl;
	_stream << LogIndent() <<"m_display_list: \t"	<< m_display_list << std::endl;
	if(m_nb_params>0)
	{
		_stream << LogIndent() <<"m_nb_params: \t"	<< m_nb_params << std::endl;
		for(int i =0; i < m_nb_params; i++)
			_stream << LogIndent() <<"params[" << i<<"]: \t"	<< m_params[i] << std::endl;
	}

	//m_ShaderManager
	const ShaderObject * shader = m_ShaderManager->FindConst(GetName());
	if(shader)
	{
		_stream << *shader;
	}
	LogIndentDecrease();
	_stream << LogIndent() <<"GpuFilter==============" << std::endl;
	_stream << LogIndent() <<"======================================" << std::endl;
	return _stream;
}
//====================================
void GpuFilter :: SetShaderManager(ShaderManager *sm)
{
	CLASS_ASSERT(sm, "NULL pointer assigment.\n");
	m_ShaderManager = sm;
}
//====================================
void GpuFilter :: SetDisplayList(GLuint list)
{
	m_display_list = list;
}
//====================================
#if _GPUCV_SUPPORT_GS
string	 GpuFilter :: SetShadersFiles(ShaderProgramNames & _Names)
#else
string	 GpuFilter :: SetShadersFiles(string _fragmentShaderFilename, string _vertexShaderFilename)
#endif
{
	CLASS_FCT_SET_NAME("SetShadersFiles");
	CLASS_FCT_PROF_CREATE_START();

	if (GetName() != "")
	{
		// only if one shader is linked to one unique filter
		if (!m_ShaderManager->RemoveShader(GetName()))
		{
			GPUCV_ERROR("Critical : can't remove shader.\n");
			return "";
		}
	}

	//if(filename1 == "" && filename2 =="")
	string res;
#if _GPUCV_SUPPORT_GS
	res = m_ShaderManager->AddShader(_Names);
	CLASS_ASSERT_FILE (res!="", _Names.m_ShaderNames[0]+" & "+_Names.m_ShaderNames[1]+" & "+_Names.m_ShaderNames[2], "Shader programmes not associated\n");
#else
	res = m_ShaderManager->AddShader(_fragmentShaderFilename, _vertexShaderFilename);
	CLASS_ASSERT_FILE (res!="", _fragmentShaderFilename #"#_#" #_vertexShaderFilename #" not associated\n");
#endif
	SetName(res);
	return res;
}
//====================================
void   GpuFilter :: GetShadersFiles(string &filename1, string &filename2)
{
	CLASS_FCT_SET_NAME("GetShadersFiles");
	CLASS_FCT_PROF_CREATE_START();
	m_ShaderManager->GetShaderFiles(GetName(), filename1, filename2);
}
//====================================
void GpuFilter :: ClearParams()
{
	CLASS_ASSERT(m_params, "m_params empty");
	for(int i=_GPUCV_SHADER_MAX_PARAM_NB;i--;)
		m_params[i]=0.0; // Filling with zeros

	m_nb_params = 0;
}
//====================================
void GpuFilter :: SetParamI(unsigned int i,float new_param)
{
	CLASS_ASSERT(i< _GPUCV_SHADER_MAX_PARAM_NB, "Index out of bound:"<<i);
	CLASS_ASSERT(i< (unsigned int) m_nb_params, "Index out of bound:"<<i);
	m_params[i]=	new_param;
}
//====================================
void GpuFilter :: AddParam(float new_param)
{
	CLASS_ASSERT(m_nb_params+1<= _GPUCV_SHADER_MAX_PARAM_NB, "Buffer overflow:"<<_GPUCV_SHADER_MAX_PARAM_NB);

	m_params[m_nb_params]=new_param;
	m_nb_params++;
}
//====================================
void GpuFilter::SetParams(const float * Params , int ParamNbr)
{
	if(Params==NULL && ParamNbr ==0)
	{
		ClearParams();
		return;
	}

	CLASS_ASSERT(ParamNbr<= _GPUCV_SHADER_MAX_PARAM_NB, "Buffer overflow:"<< _GPUCV_SHADER_MAX_PARAM_NB);
	if(ParamNbr>0 && Params == NULL)
		CLASS_ASSERT(Params, "Empty parmeters");

	for (int i =0; i< ParamNbr; i++)
		m_params[i]=Params[i];
	m_nb_params = ParamNbr;
}
//====================================
GLhandleARB GpuFilter :: GetProgram()
{
	CLASS_FCT_SET_NAME("GetProgram");
	CLASS_FCT_PROF_CREATE_START();

	GLhandleARB program=0;
	CLASS_ASSERT(m_ShaderManager->GetShader(GetName(),program), "Shader program can't be found:" << GetName());
	return program;
}
//====================================
float* GpuFilter :: GetParams()
{
	return(m_params);
}
//====================================
int GpuFilter :: GetNbParams()
{
	return(m_nb_params);
}
//====================================
const std::string & GpuFilter :: GetName() const
{
	return SGE::CL_BASE_OBJ<string>::GetID();
}
//=================================================
void      GpuFilter ::SetName(string _name)
{
	SGE::CL_BASE_OBJ<string>::SetID(_name);
}
//=================================================
GLuint GpuFilter :: FindParam(GLhandleARB prog, const char* name)
{
	GLint loc = glGetUniformLocationARB(prog, name);
	if (loc == -1)
		GPUCV_WARNING(GetName() << " error : \"" << name << "\"  doesn't exists in shader code.");

	return loc;
}
//=================================================
int GpuFilter :: Apply(GPUCV_TEXT_TYPE th_s,GPUCV_TEXT_TYPE th_d,
					   FilterSize * _size, GPUCV_TEXT_TYPE * more_tex, int nb_tex,
					   FCT_PTR_DRAW_TEXGRP(_DrawFct)/*=NULL*/)
{
	CLASS_FCT_SET_NAME_TPL(GPUCV_TEXT_TYPE,"Apply");
	CLASS_FCT_PROF_CREATE_START();
	//create texture grp
	static TextureGrp InputTexGrp;
	static TextureGrp OutputTexGrp;
	InputTexGrp.Clear();
	OutputTexGrp.Clear();
	InputTexGrp.SetGrpType(TextureGrp::TEXTGRP_INPUT);
	OutputTexGrp.SetGrpType(TextureGrp::TEXTGRP_OUTPUT);


	InputTexGrp.AddTextures(&th_s,1);
	InputTexGrp.AddTextures(more_tex, nb_tex);
	OutputTexGrp.AddTextures(&th_d,1);
	//===============================================
	return Apply(&InputTexGrp, &OutputTexGrp, _size, _DrawFct);
}
//=================================================

#define ENABLE_MRT 0
int GpuFilter :: Apply(TextureGrp * InputGrp, TextureGrp * OutputGrp,
					   FilterSize * _size,
					   FCT_PTR_DRAW_TEXGRP(_DrawFct)/*=NULL*/)
					   //int GpuFilter :: Apply(TextureGrp * InputGrp, TextureGrp * OutputGrp, int imageWidth,int imageHeight)

{
	CLASS_FCT_SET_NAME_TPL(TextureGrp,"Apply");
	CLASS_FCT_PROF_CREATE_START();
#if _GPUCV_CORE_DEBUG_FILTER
	GPUCV_DEBUG("Filter "<<  GetName() << " START:");
#endif
	//	size_t InputNbr = InputGrp->GetTextNbr();
	//	size_t OutputNbr = OutputGrp->GetTextNbr();
	//	CL_Options::OPTION_TYPE * OptionBackup = new CL_Options::OPTION_TYPE[InputNbr+OutputNbr];


	//manage destination Textures ========================
	GPUCV_TEXT_TYPE th_dest = OutputGrp->operator[](0);
	CLASS_ASSERT(th_dest, "GpuFilter :: Apply() => No destination 1 CvgArr");
	PreProcessDataTransfer(OutputGrp,InputGrp);
	//manage sources Textures ========================
	PreProcessDataTransfer(InputGrp);
	//====================================================

#if _GPUCV_DEBUG_MODE
	if(GetGpuCVSettings()->GetOption(GpuCVSettings::GPUCV_SETTINGS_FILTER_DEBUG))
	{
		int i=0;
		GPUCV_DEBUG("Input images:");
		TEXTURE_GRP_EXTERNE_DO_FOR_ALL(InputGrp, TEX,
			GPUCV_DEBUG("Texture number"<< i++);
		TEX->Print();
		);
		GPUCV_DEBUG("Output images:");

		TEXTURE_GRP_EXTERNE_DO_FOR_ALL(OutputGrp, TEX,
			GPUCV_DEBUG("Texture number"<< i++);
		TEX->Print();
		);
	}
#endif

	//check that no source and destination are equal..???


#if 0
	vector <CvArr *> changed;
	// we get corresponding textures of source and destination images
	if (src == dst && GetHardProfile()->IsFBOCompatible())
	{ // non compatible FBO : can't write & read in the same time in the texture
		GPUCV_WARNING(" => src == dst && GetHardProfile()->IsFBOCompatible()\n");
		CvArr * tmp = cvgCloneImage(src);
		changed.push_back(tmp);
		th_s = GetTextureManager()->Add(tmp);

	}
	else
#endif

#if _GPUCV_CORE_DEBUG_FILTER
		if(th_dest->GetOption(DataContainer::CPU_RETURN))
		{		GPUCV_DEBUG(th_dest->GetValStr() << ": CPURETURN TRUE");	}
		else { 	GPUCV_DEBUG(th_dest->GetValStr() << ": CPURETURN FALSE");	}
#endif
		//---------------------
		GLhandleARB program=0;

		//==========================
		//drawing part
		//==========================
		DataDsc_GLTex * DDGLtex_Dest = th_dest->GetDataDsc<DataDsc_GLTex>();

		if(!GetGpuCVSettings()->GetOption(GpuCVSettings::GPUCV_SETTINGS_FILTER_SIMULATE))
		{//to simulate shader for benchmarking CPU
			if(OutputGrp->GetTextNbr() > 1)
				RenderBufferManager()->SetContext(OutputGrp, _size->_GetWidth(), _size->_GetHeight());
			else
				DDGLtex_Dest->SetRenderToTexture();

			// gl initialisation
			DDGLtex_Dest->InitGLView();
			// get shader handle
			program = GetProgram();

			// Using filter's shader program
			glUseProgramObjectARB(program);
			_GPUCV_CLASS_GL_ERROR_TEST();

			// bind all textures
			//glActiveTextureARB(GL_TEXTURE0);
			DataDsc_GLTex * DDGLtex_Src = NULL;
			TEXTURE_GRP_EXTERNE_DO_FOR_ALL(InputGrp,
				TEXT,
				DDGLtex_Src = TEXT->GetDataDsc<DataDsc_GLTex>();
				DDGLtex_Src->_SetARBId((GLuint)(GL_TEXTURE0+iTexID));
				DDGLtex_Src->_BindARB();
				if(DDGLtex_Src->_GetTextureName()!="")
				{
					GPUCV_DEBUG("Texture using name '" << DDGLtex_Src->_GetTextureName() << "' with ID "<< iTexID);
					glUniform1iARB(FindParam(program, DDGLtex_Src->_GetTextureName().data()),(GLint)iTexID);
				}
				else
				{
					if(iTexID == 0)
					{
						GPUCV_DEBUG("Texture using name " << "'BaseImage'" << " with ID "<< iTexID);
						glUniform1iARB(FindParam(program, "BaseImage"),0);
					}
					else
					{
						char chiffre = char(iTexID -1 +48);
						string paramname="Image";
						paramname.push_back(chiffre);
						glUniform1iARB(FindParam(program, paramname.data()), (GLint)iTexID);
						GPUCV_DEBUG("Texture using name '" << paramname << "' with ID "<< iTexID);
					}
				}
			);
			

			// set shader's texture parameters
		//	if(InputGrp[0]->glUniform1iARB(FindParam(program, "BaseImage"),0);
			if (m_nb_params > 0)
				glUniform1fvARB(FindParam(program, "Parameters"),m_nb_params,m_params);

			/*
			#if		_GPUCV_CORE_DEBUG_FILTER
			for (int i=m_nb_params; i--; )
			{
			std::cout << std::endl << "param " << i << ":" << m_params[i];
			}
			std::cout << std::endl;
			#endif
			*/
/*
			for (int i=(int)InputGrp->GetTextNbr() -1; i--; )
			{
				char chiffre = char(i+48);
				string paramname="Image";
				paramname.push_back(chiffre);
				glUniform1iARB(FindParam(program, paramname.data()), i+1);
			}
*/

			// draw the calculation scene
			if(_DrawFct)
				_DrawFct(InputGrp, OutputGrp, DDGLtex_Dest->_GetWidth(), DDGLtex_Dest->_GetHeight());
			else if (m_display_list != -1) glCallList(m_display_list);
			else
				//drawQuad(_size->_GetWidth(),_size->_GetHeight());
				InputGrp->DrawMultiTextQuad(_size->_GetWidth(),_size->_GetHeight());

			glFlush();

			// DEBUG
#if DEBUG_DISPLAY
			cvNamedWindow("Src Filter Image",1);
			cvShowImage("Src Filter Image",src);
			cvgShowFrameBufferImage("Dest Filter Image", imageWidth,imageHeight,cvgConvertCVTexFormatToGL(dst->nChannels,dst->channelSeq), cvgConvertCVPixTypeToGL(dst->depth));
			cvDestroyWindow("Src Filter Image");
#endif	// End DEBUG

			//--------------------
			// All images have been convert into texture, but only dst image has been modified
			// we force cpu setting on all other images




			//if render manager is FBO, we read destination back to CPU while they are binded
			//in the framebuffer
			if((RenderBufferManager()->GetType() == TextureRenderBuffer::RENDER_OBJ_FBO))
				PostProcessDataTransfer(OutputGrp);

			//we don't force any more
			// if destination format stands on GPU and FBO aren't compatible, we copy result in destination texture
#if 0 //WARNING this should be done, must update the RenderBufferManager PBUFF to manage texture group.
			if (RenderBufferManager()->GetType() == TextureRenderBuffer::RENDER_OBJ_PBUFF && !th_dest->GetOption(DataContainer::CPU_RETURN))
			{
				RenderBufferManager()->GetResult();
			}
			else if (th_dest->GetOption(DataContainer::CPU_RETURN)) // if we want a CPU return, we have to transfer data
			{
				// force configuration of the corresponding cvgImage
				// to be cpu last one
				th_dest->SetLocation(DataContainer::LOC_CPU, true);

			}
#endif
			//---------------------

			// Return to normal shading
			glUseProgramObjectARB(0);

			TEXTURE_GRP_EXTERNE_DO_FOR_ALL(InputGrp,
				TEXT,
				DDGLtex_Src = TEXT->GetDataDsc<DataDsc_GLTex>();
			DDGLtex_Src->_UnBindARB();
			);
			//drawing done

			if(OutputGrp->GetTextNbr() > 1)
				RenderBufferManager()->UnSetContext();
			else
				DDGLtex_Dest->UnsetRenderToTexture();

#if _GPUCV_GL_USE_GLUT
			if(GetGpuCVSettings()->GetGlutDebug())
				ViewerDisplayFilter(this, InputGrp, OutputGrp);
#endif
			//if render manager is PBuffer, we read destination back to CPU after they were unbinded
			// from the framebuffer
			if((RenderBufferManager()->GetType() == TextureRenderBuffer::RENDER_OBJ_PBUFF))
				PostProcessDataTransfer(OutputGrp);

			PostProcessDataTransfer(InputGrp);

		}//to simulate shader for benchmarking CPU

		_GPUCV_CLASS_GL_ERROR_TEST();
		//CLASS_ASSERT(!ShowOpenGLError(__FILE__, __LINE__), "Error in GpuFilter::Apply()");

#if 0
		for (size_t i=changed.size(); i--;)
		{
			CvArr * truc = changed[i];
			cvgReleaseImage(&truc);
			changed.pop_back();
		}
#endif

		//	delete []OptionBackup;
#if _GPUCV_CORE_DEBUG_FILTER
		GPUCV_DEBUG("Filter "<<  GetName() << " stop.");
#endif
		return(EXIT_SUCCESS);
}
//=================================================
void GpuFilter::PreProcessDataTransfer(TextureGrp * _Grp, TextureGrp * _OptControlGrp)
{
	CLASS_ASSERT(_Grp, "GpuFilter::PreProcessDataTransfer()=>EmptyGroup");
	/*
	if(_OptControlGrp)
	{
	//check that input.tex != ouput.tex and create temp texture if required.
	TEXTURE_GRP_EXTERNE_DO_FOR_ALL(_Grp,iTex,
	if (_OptControlGrp->IsTextureInGroup(iTex))
	{
	GPUCV_NOTICE("PreProcessDataTransfer(T)=>texture replacement!");
	_Grp->ReplaceTexture(iTex, (DataContainer*) new TextureTemp(iTex));
	}
	);
	}
	*/
	if(_Grp->GetGrpType() == TextureGrp::TEXTGRP_OUTPUT)
	{//manage destination Textures ========================
		TEXTURE_GRP_EXTERNE_DO_FOR_ALL(_Grp, TEX,
			//OptionBackup[iTexID] = TEX->GetOption(0xFFFF) | DataContainer::DEST_IMG;//avoid unnecessary transfer
			TEX->PushSetOptions(DataContainer::DEST_IMG, true);
		TEX->SetLocation<DataDsc_GLTex>(false);
		);
	}
	else if(_Grp->GetGrpType() == TextureGrp::TEXTGRP_INPUT)
	{//manage sources Textures ========================
		TEXTURE_GRP_EXTERNE_DO_FOR_ALL(_Grp, TEX,
			TEX->PushSetOptions(DataContainer::UBIQUITY, true);	//is used to preserve CPU image if we have cpu_return set
		//OptionBackup[iTexID+OutputNbr] = TEX->GetOption(0xFFFF) | DataContainer::UBIQUITY;
		TEX->SetLocation<DataDsc_GLTex>(true);//make sure image is on GPU
		);
	}
	else
		CLASS_ASSERT(0, "GpuFilter::PreProcessDataTransfer()=>Unkown group Type");


}
//=================================================
void GpuFilter::PostProcessDataTransfer(TextureGrp * _Grp)
{
	CLASS_ASSERT(_Grp, "GpuFilter::GpuFilterPostProcessDataTransfer()=>EmptyGroup");

	//check if the group contain temporary textures and reaffect their data to the corresponding texture.
	//we only switch texture IDs(opengl) cause the filter does not affect CPU data.
	/*	TextureTemp * TmpTex = NULL;
	TEXTURE_GRP_EXTERNE_DO_FOR_ALL(_Grp,iTex,
	//		DataContainer * iTex = NULL;
	TmpTex = dynamic_cast<TextureTemp*>(iTex);
	if (TmpTex)
	{
	GPUCV_NOTICE("PostProcessDataTransfer(T)=>texture replacement!");
	_Grp->ReplaceTexture(TmpTex, TmpTex->GetSourceTexture());
	TmpTex->_SwitchTexID(*TmpTex->GetSourceTexture());
	delete TmpTex;
	}
	);

	*/
	if(_Grp->GetGrpType() == TextureGrp::TEXTGRP_OUTPUT)
	{//manage destination Textures ========================
		TEXTURE_GRP_EXTERNE_DO_FOR_ALL(_Grp, TEX,
			if (TEX->GetOption(DataContainer::CPU_RETURN))
				//if(OptionBackup[iTexID]&DataContainer::CPU_RETURN)
				TEX->SetLocation<DataDsc_CPU>(true);
		TEX->PopOptions();
		//TEX->ForceAllOptions(OptionBackup[iTexID]);
		);
	}
	else if(_Grp->GetGrpType() == TextureGrp::TEXTGRP_INPUT)
	{//manage sources Textures ========================
		TEXTURE_GRP_EXTERNE_DO_FOR_ALL(_Grp, TEX,
			TEX->PopOptions();
		//TEX->ForceAllOptions(OptionBackup[iTexID+OutputNbr]);
		if (TEX->GetOption(DataContainer::CPU_RETURN))
			//if(OptionBackup[iTexID+OutputNbr]&DataContainer::CPU_RETURN)
			TEX->SetLocation<DataDsc_CPU>(true);
		);
	}
	else
		CLASS_ASSERT(0, "GpuFilter::GpuFilterPostProcessDataTransfer()=>Unkown group Type");


}

#if _GPUCV_GL_USE_GLUT
void ViewerDisplayFilter(GpuFilter *, TextureGrp * _ingrp, TextureGrp * _outgrp)
{
#ifdef _WINDOWS
	GPUCV_DEBUG("Start GLUT image viewer");
	//glDrawBuffer(GL_BACK);
	InitGLView(0,GetGpuCVSettings()->GetWindowSize()[0], 0, GetGpuCVSettings()->GetWindowSize()[1]);
	glClearColor(0.2, 0.2, 0.2,1);
	glClear(GL_COLOR_BUFFER_BIT);
	ViewerDisplayTextureGroup(_ingrp, SGE::CL_Vector2Df(-0.45,0), SGE::CL_Vector2Df(0.3,0.3));
	ViewerDisplayTextureGroup(_outgrp, SGE::CL_Vector2Df(+0.45,0), SGE::CL_Vector2Df(0.3,0.3));
	glClearColor(0, 0, 0,1);
	glutSwapBuffers();
	GPUCV_DEBUG("Stop GLUT image viewer");
#endif
}
#endif

}//namespace GCV

