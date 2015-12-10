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
/** \brief cpp file containing definitions of some tools to optimize coding
\author Yannick Allusse
*/
#include "StdAfx.h"
#include <GPUCVCore/GpuFilterManager.h>
#include <GPUCVCore/coretools.h>
#include <GPUCVHardware/revision.h>




#if _GPUCV_SHADER_LOAD_FORCE
#include <dirent.h>
#endif


namespace GCV{


#if _GPUCV_GL_USE_GLUT
void DbCallbackDisplay()
{
	glutSwapBuffers();
}
#endif
//=================================================
/**
\todo Save OpenGL context for Linux/Mac OS
\todo OpenGL init should be move at a higher level.
*/
int  GpuCVInit(bool InitGLContext/*=true*/, bool isMultiThread/*=false*/)
{
	GPUCV_DEBUG("\nGpuCVInit()--- Debug mode is ON ---");

	if (GetGpuCVSettings()->IsInitDone())
	{
	//	GPUCV_WARNING("GpuCV initializing already done.");
		return 0;//already done
	}

	/*********************************************************/
	/*                     OpenGL/GLUT/GLEW Init                      */
	GPUCV_NOTICE("======================================");
	GPUCV_NOTICE("Starting GpuCV");
	GPUCV_NOTICE("==============");
	GPUCV_NOTICE(GetGpuCVSettings()->GetVersionDescription());
	GPUCV_NOTICE("======================================");


	if (InitGLContext)
	{
#if _GPUCV_GL_USE_GLUT
		char ** argv = new char *[1];
		argv[0] = new char[1];
		argv[0][0] = 0;
		int argc = 1;

		GPUCV_DEBUG("Creating GL context...");
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_ALPHA);

		//glutInitWindowPosition(0/*1280*/,0);
		//glutInitWindowSize(512, 512);

		//create main rendering window
		GPUCV_DEBUG("Creating rendering windows...");
		GetGpuCVSettings()->AddGlutWindowsID(glutCreateWindow("GPUCV_computing_window"));


		if(GetGpuCVSettings()->GetGlutDebug())
		{
			//show the windows...
			glutInitWindowPosition(1280,0);
			glutInitWindowSize(512, 512);
			glutDisplayFunc(DbCallbackDisplay);
		}
		else
			glutHideWindow();

		delete argv[0];
		delete [] argv;
#else//OPENGL direct support
#ifdef _WINDOWS
	HDC hDC;//
	HGLRC hRC;//
#endif
		HWND hwndC = GetConsoleWindow() ; /* Great!!  This function
		  cleverly "retrieves the window handle
		  used by the console associated with the calling process",
		  as msdn says */

		  // Then we could just get the HINSTANCE:
		 // HINSTANCE hInstC = GetModuleHandle( 0 ) ;
		  //HINSTANCE hInstCons = (HINSTANCE)GetWindowLong( hwndC, GWL_HINSTANCE );

		hDC = GetDC( hwndC );
		PIXELFORMATDESCRIPTOR pfd;
		ZeroMemory( &pfd, sizeof( pfd ) );
		pfd.nSize = sizeof( pfd );
		pfd.nVersion = 1;
		pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL |
					  PFD_DOUBLEBUFFER;
		pfd.iPixelType = PFD_TYPE_RGBA;
		pfd.cColorBits = 24;
		pfd.cDepthBits = 16;
		pfd.iLayerType = PFD_MAIN_PLANE;
		int iFormat = ChoosePixelFormat( hDC, &pfd );
		SetPixelFormat( hDC, iFormat, &pfd );
		
		hRC = wglCreateContext( hDC );
		wglMakeCurrent(hDC, hRC);

		
		//wglDeleteContext( hRC );
		//ReleaseDC( hWnd, hDC );
#endif
	}

	//Initialize all CL_Singleton manager with default manager:
	GPUCV_DEBUG("Initializing Singleton managers...");
	TextureManager::GetSingleton();
	GpuFilterManager::GetSingleton();
	//=================================

	GPUCV_DEBUG("Saving GL Context for multi-threading...");
	GLContext::SetMultiContext(isMultiThread);

#ifdef _WINDOWS
	GLContext * MainContext = NULL;
	if(isMultiThread)//create context
		MainContext = new GLContext(wglGetCurrentDC(),wglCreateContext(wglGetCurrentDC()));
	else//use existing one
		MainContext = new GLContext(wglGetCurrentDC(),wglGetCurrentContext());
#else
	//???
#endif


	//Set default options for some objects:
	unsigned long int DataContainerOpt = /*DataContainer::PRESERVE_MEMORY |*/ DataContainer::UBIQUITY;

#if _GPUCV_DEBUG_MODE
	DataContainerOpt= DataContainerOpt
		|DataContainer::LCL_OPT_DEBUG \
		|DataContainer::LCL_OPT_WARNING \
		|DataContainer::LCL_OPT_ERROR \
		|DataContainer::DBG_IMG_LOCATION\
		|(DataContainer::DBG_IMG_MEMORY && GetGpuCVSettings()->GetOption(GpuCVSettings::GPUCV_SETTINGS_DEBUG_MEMORY))\
		|DataContainer::DBG_IMG_TRANSFER\
		|DataContainer::DBG_IMG_MEMORY;
		//|DataContainer::DBG_IMG_FORMAT;
	//		|DataContainer::SMART_TRANSFER
	//|DataContainer::LCL_OPT_NOTICE
	//|DataContainer::PRESERVE_MEMORY\

#endif
	GetGpuCVSettings()->SetDefaultOption("DataContainer", DataContainerOpt, true);
	//...more options to come here....
	//=====================================
	SetThread();

	GPUCV_DEBUG("Glew init...");
	glewInit();
	GPUCV_NOTICE("Reading OpenGL extensions");
	GetHardProfile();//create hardware singleton
#if _GPUCV_DEBUG
	MainGPU()->m_glExtension.PrintExtension();
#endif

	if (GLEW_ARB_vertex_shader && GLEW_ARB_fragment_shader && GLEW_EXT_geometry_shader4)
	{	GPUCV_NOTICE("Vertex, Geometry and Fragment shaders supported");	}

	if (glewIsSupported("GL_VERSION_2_1"))
	{	GPUCV_NOTICE("Ready for OpenGL 2.1");	}
	else if (glewIsSupported("GL_VERSION_2_0"))
	{	GPUCV_NOTICE("Ready for OpenGL 2.0");	}
	else if (glewIsSupported("GL_VERSION_1_0"))
	{	GPUCV_NOTICE("Ready for OpenGL 1.0");		}
	else if (GLEW_ARB_vertex_shader && GLEW_ARB_fragment_shader && GLEW_EXT_geometry_shader4)
	{	GPUCV_NOTICE("Vertex, Geometry and Fragment shaders supported");	}
	else if (GLEW_ARB_vertex_shader && GLEW_ARB_fragment_shader)
	{	GPUCV_NOTICE("Vertex and Fragment shaders supported");	}
	else
	{
		GPUCV_ERROR("No GLSL support->No operators using GLSL shaders will be available");
	}

	GPUCV_DEBUG("Create render buffer manager...");
	if(GetHardProfile()->IsCompatible(GenericGPU::HRD_PRF_1))
	{
		RenderBufferManager(TextureRenderBuffer::RENDER_OBJ_AUTO,true);

		//RenderBufferManager(TextureRenderBuffer::RENDER_OBJ_PBUFF,true);
		//RenderBufferManager(TextureRenderBuffer::RENDER_OBJ_OPENGL_BASIC,true);

		if (InitGLContext)
		{
			//PBuffersManager()->SetUniqueContext();
			//PBuffersManager()->Block();
		}
		//initialize GLview and renderbuffer object(FBo or Pbuffers) a first time, to save time later....
#if 0
		RenderBufferManager()->SetContext(RenderBufferManager()->SelectInternalTexture(TextureRenderBuffer::RENDER_FP8_RGB));
		InitGLView(512,512);
		RenderBufferManager()->UnSetContext();
#endif
	}
	/*********************************************************/

#if _GPUCV_SHADER_LOAD_FORCE
	//AUTO load shaders?
	if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_LOAD_ALL_SHADERS))
		load_default_shaders();
#endif

	UnsetThread();



	//init the profiler with local informations such as:
	/*
		-revision...
		-debug/release
		-CPU
		-GPU
	*/
		SG_TRC::CL_TRACE_BASE_PARAMS * GlobalParams=new SG_TRC::CL_TRACE_BASE_PARAMS();
	#ifdef _DEBUG
		GlobalParams->AddChar("mode", "DEBUG");
	#else
		GlobalParams->AddChar("mode", "RELEASE");
	#endif
	#ifdef _LINUX
		GlobalParams->AddChar("OS", "LINUX");
	#elif defined (_WINDOWS)
		GlobalParams->AddChar("OS", "WINDOWS");
	#elif defined (_MACOS)
		GlobalParams->AddChar("OS", "MACOS");
	#else
		GlobalParams->AddChar("OS", "Unknown");
	#endif
		GlobalParams->AddChar("version", _GPUCV_VERSION_MAJOR);
		GlobalParams->AddChar("revision", _GPUCV_REVISION_VAL);
		//opencv version:
		//done in cvgInit() .. GlobalParams->AddChar("opencv-V", CV_VERSION);
		
		//add GPU name
		GlobalParams->AddChar("GPU", ProcessingGPU()->GetRenderer().data());



		//add CPU Name
		std::string strCPUName = GetCpuName();
		GlobalParams->AddChar("CPU", strCPUName.data());





		GlobalParams->SetAsCommonParam();

#if _GPUCV_PROFILE_TOCONSOLE
	CL_Profiler::GetTimeTracer().EnableConsole();
#endif
	//------------------------------

	GetGpuCVSettings()->SetInitDone();

	return 1;//success
}
//=================================================
void GpuCVTerminate()
{
	//GetTextureManager()->~TextureManager();
	//GetGpuCVSettings()->~GpuCVSettings();
	//GetHardProfile()->~HardwareProfile();
}
//=======================================================
#if _GPUCV_SHADER_LOAD_FORCE
char* strcat(char* s, char* t)
{
	char* buffer;
	int i=0, j=0;

	while( s[i] != 0) {
		i=i+1;
	}

	while( t[j] != 0) {
		i = i+1;
		j = j+1;
	}

	buffer = (char*)malloc(i+1);

	i = 0;
	while( s[i] != 0) {
		buffer[i] = s[i];
		i=i+1;
	}

	j = 0;
	while( t[j] != 0) {
		buffer[i] = t[j];
		i=i+1;
		j=j+1;
	}
	buffer[i] = 0;

	return(buffer);
}
//=======================================================
int load_default_shaders()
{
	if(!GetHardProfile()->IsCompatible(GenericGPU::PROFILE_1))
	{
		GPUCV_ERROR("load_default_shaders():Hardware profile 1 not reached, can't load shaders");
	}
	else
	{
		SetThread();

		DIR *rep;
		struct dirent *read;
		char* vertexdir = "VShaders\\";
		char* fragmentdir = "FShaders\\";

		vector <string> vertprog;
		vector <string> fragprog;

		// vertex shaders
		rep = opendir(vertexdir);
		while ((read = readdir(rep)))
		{
			if (strcmp(read->d_name,"default.vert") && strcmp(read->d_name,".") && strcmp(read->d_name,".."))
				vertprog.push_back(string(strcat(vertexdir, read->d_name)));
		}
		closedir(rep);


		// fragment shaders
		rep = opendir(fragmentdir);
		while ((read = readdir(rep)))
		{
			if (strcmp(read->d_name,"default.frag") && strcmp(read->d_name,".") && strcmp(read->d_name,".."))
				fragprog.push_back(strcat(fragmentdir, read->d_name));
		}
		closedir(rep);

		string filename1, filename2, file1, file2;
		size_t pos_rep_sep;

		float size = fragprog.size();
		float i=0;

		string filtername;

		while(fragprog.size())
		{
			i++;
			for (int k=32; k--; cout << char(8) << flush);
			cout << "Loading default filters..." << (int)(100*(i/size)) << "%" << flush;

			filename1 = fragprog[fragprog.size()-1];
			pos_rep_sep = filename1.find_last_of('\\')+1;
			file1 = filename1.substr(pos_rep_sep, filename1.size()-pos_rep_sep-5);
			for (int j=(int)vertprog.size(); j--; )
			{
				filename2 = vertprog[j];
				pos_rep_sep = filename2.find_last_of('\\')+1;
				file2 = filename2.substr(pos_rep_sep, filename2.size()-pos_rep_sep-5);
				if (file2 == file1) break;
			}

			if (file2 == file1)
			{
				filtername = GetFilterManager()->GetFilterName(filename1, filename2);
				GetFilterManager()->ForceShaderLoading(filtername);
				fragprog.pop_back();
				vertprog.pop_back();
			}
			else
			{
				filtername = GetFilterManager()->GetFilterName(filename1);
				GetFilterManager()->ForceShaderLoading(filtername);
				fragprog.pop_back();
			}
		}
		cout << endl;

		while (vertprog.size())
		{
			filename2 = vertprog[vertprog.size()-1];
			filtername = GetFilterManager()->GetFilterName(filename2);
			GetFilterManager()->ForceShaderLoading(filtername);
			vertprog.pop_back();
		}

		for (int k=32; k--; cout << char(8) << flush);
		cout << endl;
		cout << endl;

#if _GPUCV_DEBUG_MODE
		//GetFilterManager()->ListFilters(cout);
#endif
		UnsetThread();
		return 1;
	}
}
#endif
//================================================

//=================================================
}//namespace GCV

