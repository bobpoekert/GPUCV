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
/**
Main GpuCV sample application to test all the function developed with GpuCV
*/
#include "StdAfx.h"
#include "mainSampleTest.h"
#include "commands.h"
#include "misc_test.h"
#include "cvg_test.h"
#include "cxcoreg_test.h"
#include <SugoiTracer/HtmlReport.h>
#include <GPUCVSwitch/Cl_Dll.h>
using namespace GCV;



//=============================================================================
void PrintMsg()
{
	GPUCV_NOTICE("\n\n-------- Welcome to the cvGPU sample testing interface ---------");
	GPUCV_NOTICE("List of available cvgxxxx operators for testing :");

	//print information from testing libraries
	cxcore_test_print();
	cv_test_print();

	GPUCV_NOTICE("===other operators========");
	//GPUCV_NOTICE(" 'and'\t : cvgAnd()");
	GPUCV_NOTICE(" 'isdiff'\t : cvgIsDiff()");
	GPUCV_NOTICE(" 'tocpu'\t : cvgSetLocation(CPU)");
	GPUCV_NOTICE(" 'togpu'\t : cvgSetLocation(GPU)");
	GPUCV_NOTICE(" 'line'\t : cvgLine()");
	GPUCV_NOTICE(" 'rectangle'\t : cvgRectangle()\n" );
	GPUCV_NOTICE(" 'circle'\t : cvgCircle()");
	GPUCV_NOTICE(" 'stats'\t : Calculate image stats like Min/Max/Average");
	GPUCV_NOTICE("=== global command========");
	GPUCV_NOTICE("\n\nList of action for benchmarking:");
	GPUCV_NOTICE(" 'imgformat %img %type %channels'\t : Change image format with img=[src1|src2Lmask], type=[8u, 8s, 16u, 16s, 32u, 32s, 32f, 64f] and channels=[1|2|3|4].");
	GPUCV_NOTICE(" 'loopnbr %loop'\t : define the loop number for benchmark");
	GPUCV_NOTICE(" 'load %ImgId(1-2) %ImgFileName'\t : load a new source image()");
	GPUCV_NOTICE(" 'resizeimg %width %height'\t : resize both source image to width and height()");
	GPUCV_NOTICE(" 'benchmode'\t : switch GpuCV to benchmarking mode, disabling error/warning/debug output and internal profiling.");
	GPUCV_NOTICE(" 'runbench #loopnbr'\t : Run a benchmark with all the operators()");
	GPUCV_NOTICE(" 'clearbench'\t : Clear all the profiling data already recorded");
	GPUCV_NOTICE(" 'savebench'\t : Save profiling data to HTML and TXT files");
	GPUCV_NOTICE(" 'benchreport #path'\t : Save profiling report as a web page and SVG images");
	GPUCV_NOTICE(" 'showsrc'\t : show the 2 source images()");
	GPUCV_NOTICE(" 'gpucvstats'\t : show GpuCV statistics");
	GPUCV_NOTICE(" 'imginfo 0xXXXX' => Give information about the given image ID");
	GPUCV_NOTICE(" 'transfertest %dataTransfer' => test data transfer between all known types.");
	GPUCV_NOTICE(" 'cmdfile %filePath'\t => Load a text file containing commands.");
	GPUCV_NOTICE(" 'cmdclear'\t => Clear the command stack, used mainly with command files.");
	GPUCV_NOTICE(" 'wait %sec'\t => Wait X seconds, used mainly with command files.");
	GPUCV_NOTICE("=== global settings========");
	GPUCV_NOTICE(" 'disable/enable %\t enable/disable special flag");
	GPUCV_NOTICE("\t 'warning' => GpuCVSettings::GPUCV_SETTINGS_GLOBAL_WARNING");
	GPUCV_NOTICE("\t 'error' => GpuCVSettings::GPUCV_SETTINGS_GLOBAL_ERROR");
	GPUCV_NOTICE("\t 'debug' => GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG");
	GPUCV_NOTICE("\t 'notice' => GpuCVSettings::GPUCV_SETTINGS_GLOBAL_NOTICE");
	GPUCV_NOTICE("\t 'useopencv' => GpuCVSettings::GPUCV_SETTINGS_USE_OPENCV");
	GPUCV_NOTICE("\t 'internprofiling' => GpuCVSettings::GPUCV_SETTINGS_PROFILING");
	GPUCV_NOTICE("\t 'profileclass' => GpuCVSettings::GPUCV_SETTINGS_PROFILING_CLASS");
	GPUCV_NOTICE("\t 'profileoper' => GpuCVSettings::GPUCV_SETTINGS_PROFILING_OPER");
	GPUCV_NOTICE("\t 'profiletransfer' => GpuCVSettings::GPUCV_SETTINGS_PROFILING_TRANSFER");
	GPUCV_NOTICE("\t 'glerrorexception' => GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_RISE_EXCEPTION");
	GPUCV_NOTICE("\t 'glerrorcheck' => GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_CHECK");
	GPUCV_NOTICE("\t 'simulefilter' => GpuCVSettings::GPUCV_SETTINGS_FILTER_SIMULATE");
	GPUCV_NOTICE("\t 'datapreload' => DataPreloading");
	GPUCV_NOTICE(" 'q' to quit...");
	GPUCV_NOTICE(" 'h' or '?' to show this guide...");
	std::cout << "\n>";;
}

//=============================================================================
//=============================================================================
//						Parsing commands
//=============================================================================
//=============================================================================
bool use_scalar=false;
bool use_mask=false;
CvScalar *global_scalar=NULL;
std::string ConsoleCommand = "";

bool ParseCommands(const int _argc, char ** _argv)
{
	if (!(_argc && _argv))
		return false;

	NB_ITER_BENCH = 10;
	ShowImage = true;
	AvoidCPUReturn = true;

	std::string AllCommands;// = _argv[1];
	for(int i=1; i < _argc; i++)
	{
#if defined _GPUCV_SUPPORT_CUDA && !_GCV_CUDA_EXTERNAL 
		if(strcmp(_argv[i], "cudaid")==0)
		{
			_argv[i+1][1]='\n';
			cvgCudaSetProcessingDevice(atoi(_argv[i+1]));

			std::cout << "Selected CUDA device ID: "<< _argv[i+1] << std::endl;
		}
#endif

		if(i!=1)
			AllCommands += " ";
		AllCommands += _argv[i];

	}
	ConsoleCommand = AllCommands;
	return true;
}


/** \brief Read commands string from the queue and process it.
* <br>
This function reads commands string from the queue and process them. It calls cxcoreg_processCommand()/cvg_processCommand()/misc_processCommand() to process commands related
to OpenCV libraries and then look for internal commands.
* \par List of internal commands available:
<ul>
	<li>Benchmarking:
		<ul>
			<li><b>'cv_all'</b>: Call all existing GpuCV/cv.lib operators</li>
			<li><b>'cxcore_all'</b>: Call all existing GpuCV/cxcore.lib operators</li>
			<li><b>'savebench %filename'</b>: Save benchmarks file</li>
			<li><b>'loadbench %filename'</b>: Load benchmarks file</li>
			<li><b>'clearbench'</b>: Clear benchmarks</li>
			<li><b>'loopnbr'</b>: Set the loop number to use in test functions, default is 1.</li>
			<li><b>'testbenchloop'</b>: Test benchmarking loop to record the benchmarking cost.</li>
		</ul>
	</li>
	<li>Misc:
		<ul>
			<li><b>'cmdfile %filename'</b>: Load a file into the command queue. All the command from this file should end by ';'</li>
			<li><b>'clearcmd'</b>: Clear the command queue</li>
			<li><b>'wait %time'</b>: Application sleep for %time second</li>
			<li><b>'?'</b> or <b>'h'</b>: Print command list</li>
			<li><b>'gpucvstats'</b>: Show internal informations about the library, see ShowGpuCVStats().</li>
			<li><b>'q'</b> or <b>'exit'</b>: Exit program</li>
		</ul>
	</li>
	<li>Image manipulation:
		<ul>
			<li><b>'resizeimg'</b>: Resize all source images to new %width and %height, see resizeImage().</li>
			<li><b>'imgformat'</b>: Change image format, see changeImageFormat().</li>
			<li><b>'showsrc'</b>: Show source images, see ShowSrcImgs().</li>
			<li><b>'imginfo'</b>: Show images informations, see PrintImageInfo().</li>
		</ul>
	</li>

</ul>
\return -1 to exit program, 0 for errors=> no command found, 1 for success => command found and executed.
*/
int ProcessCommand(std::string &LocalCmdStr)
{
	std::string Arg1;

	std::string CurCmd;
	SGE::GetNextCommand(LocalCmdStr, CurCmd);

	if (CurCmd=="")
	{
		return 0;
	}
	else if(cxcoreg_processCommand(CurCmd,LocalCmdStr))
	{
		return 1;
	}
	else if(cvg_processCommand(CurCmd,LocalCmdStr))
	{
		return 1;
	}
	else if(misc_processCommand(CurCmd,LocalCmdStr))
	{
		return 1;
	}

	CurLibraryName="";

	if (CurCmd=="testbenchloop")
	{
		testBenchLoop(GlobSrc1);
		CmdOperator = false;
	}
	//Beta functions, should be moved into the corresponding c**_processCommand() when stable
#if _GPUCV_DEVELOP_BETA
	else if (CurCmd=="hamed")
	{
		runHamed(GlobSrc1);
		CmdOperator = true;
	}
	else if (CurCmd=="render")
	{
		runRenderBufferTest(GlobSrc1, GlobSrc2);
		CmdOperator = true;
	}
	else if (CurCmd=="depth")
	{
		runRenderDepth(GlobSrc1, GlobSrc2);
		CmdOperator = true;
	}
	else if (CurCmd=="art")
	{
		RunART();
		CmdOperator = true;
	}

	else if (CurCmd=="connected")	runConnectedComp(GlobSrc1);

	else if (CurCmd=="stats")
	{
		CmdOperator = false;
		runStats(GlobSrc1);
	}
#endif

#if 0//TEST_GPUCV_CV
	else if (CurCmd=="resizefct")
	{
		GPUCV_NOTICE("\nwidth, height and GLSL function name?");
		int Width, Height;
		std::string  FctName;
		//scanf("%d %d %s",&Width, &Height, FctName);
		SGE::GetNextCommand(LocalCmdStr, Width);
		SGE::GetNextCommand(LocalCmdStr, Height);
		SGE::GetNextCommand(LocalCmdStr, FctName);
		runResizeFct(GlobSrc1, Width, Height, (char*)FctName.data());
	}

#endif
	//else if (CurCmd=="vbo")			runVBO(GlobSrc1);

	/*======CONSOLE COMMANDES ============================*/
	//change settings
	else if (CurCmd=="loopnbr")
	{
		GPUCV_NOTICE("\nLoop number for benchmark");
		SGE::GetNextCommand(LocalCmdStr, NB_ITER_BENCH);
		CmdOperator = false;
	}
	else if (CurCmd=="resizeimg")
	{
		GPUCV_NOTICE("\nwidth and height?");
		int Width, Height;
		SGE::GetNextCommand(LocalCmdStr, Width);
		SGE::GetNextCommand(LocalCmdStr, Height);
		resizeImage(&GlobSrc1, Width, Height);
		resizeImage(&GlobSrc2, Width, Height);
		resizeImage(&MaskBackup, Width, Height);
		CmdOperator=false;
	}
	else if (CurCmd=="imgformat")
	{
		changeImageFormat(LocalCmdStr);
		CmdOperator = false;
	}


#if _GPUCV_DEVELOP_BETA
	else if (CurCmd=="convertscale")
	{
		convertS(LocalCmdStr);
		CmdOperator = false;
	}
#endif
#if _GPUCV_DEPRECATED
	else if (CurCmd=="localsum")
	{
		localS(LocalCmdStr);
		CmdOperator = false;
	}
#endif

	else if (CurCmd=="load")
	{
		GPUCV_NOTICE("\nSource Image 1 or 2?[1||2]");
		int ImgNumber;
		SGE::GetNextCommand(LocalCmdStr, ImgNumber);
		GPUCV_NOTICE("\nImage Name?");
		std::string ImgName;
		SGE::GetNextCommand(LocalCmdStr, ImgName);
		if (ImgNumber == 1){loadImage(GlobSrc1, (char*)ImgName.data()); }
		else if (ImgNumber == 2){loadImage(GlobSrc2, (char*)ImgName.data()); }
		else printf("\nWrong image number, 1 or 2!");
		CmdOperator=false;
	}
	else if (CurCmd=="benchmode")
	{
		SwitchToBenchMode();
		CmdOperator = false;
	}
	else if (CurCmd=="enable")
	{
		CmdOperator = false;
		GPUCV_NOTICE("Flag to enable?");
		SGE::GetNextCommand(LocalCmdStr, Arg1);
		GPUCV_NOTICE(Arg1);
		if(!EnableDisableSettings(Arg1, true))
		{
			GPUCV_NOTICE("Enable=> Wrong parameter");
		}
	}
	else if (CurCmd=="disable")
	{
		CmdOperator = false;
		GPUCV_NOTICE("Flag to disable?");
		SGE::GetNextCommand(LocalCmdStr, Arg1);
		GPUCV_NOTICE(Arg1);
		if(!EnableDisableSettings(Arg1, false))
		{
			GPUCV_NOTICE("Disable=> Wrong parameter");
		}
	}
	//show informations
	else if (CurCmd=="gpucvstats")
	{
		CmdOperator = false;
		ShowGpuCVStats();
	}
	else if ( (CurCmd=="?") || (CurCmd=="h"))
	{
		CmdOperator = false;
		PrintMsg();
	}
	else if ((CurCmd=="-h") || (CurCmd=="--help"))
	{
		CmdOperator = false;
		PrintMsg();
		exit(1);//used to get help from the shell, not to run anything
	}
	else if (CurCmd=="showsrc")
	{
		ShowSrcImgs(GlobSrc1, GlobSrc2, GlobMask);
		CmdOperator=false;
	}
	else if (CurCmd=="imginfo")
	{
		CmdOperator = false;
		GPUCV_NOTICE("\nImage ID in HEXA?");
		SGE::GetNextCommand(LocalCmdStr, Arg1);
		PrintImageInfo(Arg1);
	}
	//commands
	else if ((CurCmd=="q")||(CurCmd=="exit"))
	{
		CmdOperator = false;
		return -1;//exit
	}
	else if (CurCmd=="runbench")
	{
		GPUCV_NOTICE("Loop number for benchmark:");
		SGE::GetNextCommand(LocalCmdStr, NB_ITER_BENCH);
		CmdOperator = false;
		SelectIpp = false;
		runBench(&GlobSrc1,&GlobSrc2, &GlobMask);
	}
	else if (CurCmd=="clearbench")
	{
		CmdOperator = false;
		SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance().ClearAll();
		CL_Profiler::GetTimeTracer().ClearAll();
	}
	else if (CurCmd=="savebench")
	{
		CmdOperator = false;
		std::string FileName;
		GPUCV_NOTICE("Filename?");
		SGE::GetNextCommand(LocalCmdStr, FileName);
		SaveBench(FileName.data());
		//save also internal benchmarks
#if 0//_GPUCV_PROFILE
		FileName+="_internal.xml";
		CL_Profiler::Savebench(FileName);
#endif
	}
	else if (CurCmd=="loadbench")
	{
		CmdOperator = false;
		std::string FileName;
		GPUCV_NOTICE("Filename?");
		SGE::GetNextCommand(LocalCmdStr, FileName);
		LoadBench(FileName.data());
		//save also internal benchmarks
#if 0//_GPUCV_PROFILE
		GPUCV_NOTICE("Filename?");
		SGE::GetNextCommand(LocalCmdStr, FileName);
		FileName+="_internal.xml";
		CL_Profiler::Loadbench(FileName);
#endif
	}
	else if (CurCmd=="benchreport")
	{
		CmdOperator = false;
		std::string FileName;
		GPUCV_NOTICE("Filename?");
		SGE::GetNextCommand(LocalCmdStr, FileName);
		BenchReport(FileName.data());
	}
	else if (CurCmd=="cmdfile")
	{//load external command file
		std::string FileName;
		SGE::GetNextCommand(LocalCmdStr, FileName);
		//FileName = GetGpuCVSettings()->GetShaderPath() + "/" + FileName;
		std::string DataFile;
		GPUCV_NOTICE("Loading external command file:" << FileName);
		if (!textFileRead(FileName, DataFile))
		{//eror while loading.
		}
		LocalCmdStr = DataFile;
		CmdOperator = false;
	}
	else if (CurCmd=="cmdclear")
	{
		CmdOperator = false;
		LocalCmdStr = "";
	}
	else if (CurCmd=="wait")
	{
		GPUCV_NOTICE("Time to Wait in sec?");
		int Time;
		SGE::GetNextCommand(LocalCmdStr, Time);
		SGE::Sleep(Time*1000);
		CmdOperator = false;
	}
	//switching mechanism test.
//	else if (CurCmd=="clean_h")      cleanHeaders();
//	else if (CurCmd=="parse_h")      ParseHeaderFiles();

	//=========================
	else
	{
		CmdOperator = false;
		GPUCV_NOTICE("Wrong command '" << CurCmd << "'");
		return 0;
	}
	return 1;
}
//=============================================================================
//=============================================================================
//						Benchmarking commands
//=============================================================================
//=============================================================================
void runBench(IplImage **src1, IplImage ** src2, IplImage ** mask)
{
	bool Loop=true;
	ShowImage = false;

#if 0//non-square
	int imageSize[][2] = {
		//{4096,4096},
		{2048,2048},
		//	{2047,2047},
		//		{64,1024},
		{1920,1080},//format:1080i
		//{1024,1024},
		//	{950,950},
		//	{900, 900},
		//	{800,800},
		//	{768,768},
		//	{1024,256},
		{1080,720},//format:720p
		{720,480},//480p
		//{512,512},
		{320,240},
		//{256,256},
		{160,120},
		{128,128},
		//{64,64}
		//	{32,32}
	};
#else//square
	int imageSize[][2] = {
		//{4096,4096},
		{2048,2048},
		//		{64,1024},
		{1024,1024},
		//	{950,950},
		//	{900, 900},
		//	{800,800},
		{768,768},
		//	{1024,256},
		{512,512},
		{256,256},
		{128,128},
		//{64,64}
		//	{32,32}
	};
#endif
	int NbrOfSize = sizeof(imageSize)/sizeof(int)/2;

	AvoidCPUReturn=true;
	if (NB_ITER_BENCH <2)NB_ITER_BENCH++;


	//init default scalar value
	CvScalar value;
	value.val[0]=123;
	value.val[1]=123;
	value.val[2]=123;
	value.val[3]=123;

	SwitchToBenchMode();

	//choose to use IPP or not.
	cvUseOptimized(SelectIpp);

	//LoadDefaultImages(GetGpuCVSettings()->GetShaderPath());

	int LoopId = 0;
	int NbrOfOptions = 4; //no mask, mask, no mask+scalar, mask+scalar
	int OptionsId=0;
	while(LoopId < NbrOfSize)
	{
		//init image size
		resizeImage( src1, imageSize[LoopId][0], imageSize[LoopId][1]);
		resizeImage( src2, imageSize[LoopId][0], imageSize[LoopId][1]);
		resizeImage( &MaskBackup, imageSize[LoopId][0], imageSize[LoopId][1]);
		mask = &MaskBackup;
		GlobMask = MaskBackup;
		GPUCV_NOTICE("\nNew image size : " << imageSize[LoopId][0] << "*" << imageSize[LoopId][1]);

		//start operator benchs, they depends on some variables:
		//cv_test_enable/cxcore_test_enable/transfer_test_enable
		RunIplImageTransferTest(*src1, true);
		RunIplImageTransferTest(*src1, false);
		cxcoreg_runAll(src1, src2, mask);
		cvg_runAll(src1, src2, mask);

		//check loop and custum settings
		LoopId++;
		if(LoopId >= NbrOfSize)
		{
			if(SelectIpp==false && (GpuCVSelectionMask&OperIPP))
			{
				SelectIpp=true;
				cvUseOptimized(SelectIpp);
//				enableDisable("CUDA");
//				enableDisable("GLSL");

#ifdef _LINUX
#else
				Sleep(500);
#endif
				LoopId = 0;
				GPUCV_NOTICE("Active IPP mode.");
			}
		}
	}

	GPUCV_NOTICE("\nBenchmark terminated...");
	std::string strBenchmarkFile="Benchmarks_";
	#ifdef _LINUX
		strBenchmarkFile+="LINUX";
	#elif defined (_WINDOWS)
		strBenchmarkFile+="WINDOWS";
	#elif defined (_MACOS)
		strBenchmarkFile+="MACOS";
	#else
		strBenchmarkFile+="Unknown";
	#endif
	strBenchmarkFile+=".xml";
	SaveBench(strBenchmarkFile.data());


#if _GPUCV_PROFILE
	strBenchmarkFile+="_internal.xml";
	CL_Profiler::Savebench(strBenchmarkFile);
#endif
	NB_ITER_BENCH--;
	GetGpuCVSettings()->PopOptions();
}


//======================================================
void SwitchToBenchMode()
{
	//tell GPUCV to avoid all debug/warning messages
	// and not to check shader file changes.

	GetGpuCVSettings()->PushSetOptions(
		0
		| GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG
		| GpuCVSettings::GPUCV_SETTINGS_GLOBAL_WARNING
		| GpuCVSettings::GPUCV_SETTINGS_PROFILING_OPER
		| GpuCVSettings::GPUCV_SETTINGS_CHECK_SHADER_UPDATE
		| GpuCVSettings::GPUCV_SETTINGS_SYNCHRONIZE_ON_ERROR
		//#if 1//!_GPUCV_DEBUG_MODE
		//| GpuCVSettings::GPUCV_SETTINGS_GLOBAL_NOTICE
		| GpuCVSettings::GPUCV_SETTINGS_CHECK_IMAGE_ATTRIBS
		| GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_CHECK
		//#endif
		,
		false);

	
	GetGpuCVSettings()->PushSetOptions(
		GpuCVSettings::GPUCV_SETTINGS_PROFILING,
		true);
		
	std::cout<<"Current GPUCV settings:" << std::endl << GetGpuCVSettings()->GetOptionsDescription();
	benchmark=true;
}
//=============================================================================
/** \todo Create load benchmark function
*/
void LoadBench(const char * filename)
{
	std::string path = cvgGetShaderPath();
	path += "Benchmarks/";
	path += filename;
	SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance().XMLLoadFromFile(path.data());
}
//=============================================================================
void SaveBench(const char * filename)
{
	//clean benchmarking results
	//AppliTracer()->CleanMaxTime();
	//AppliTracer()->CleanMaxTime();
	//!\todo Remove the 2 maximum values..???
	std::string path = cvgGetShaderPath();
	path += "Benchmarks/";
	path += filename;
	SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance().WaitProcessingThread();
    SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance().XMLSaveToFile(path.data());

	//!\todo Save internal benchmarks...
}

//======================================================
void BenchReport(const char * _filename)
{
//define color table
	std::vector <SG_TRC::ColorFilter*> vectorColorFilters;
	SG_TRC::ColorFilter ColorTable;
	ColorTable.m_InputColors= "";//clear it, cause it might contains global parameter set
	GCV::DllManager::Instance().GenerateLibraryColorFilters(ColorTable, 6);
	ColorTable.m_strColorField = "type";

	/*
	ColorTable.m_InputColors.AddParam("OpenCV_3", "rgb(140,20,20)");
	ColorTable.m_InputColors.AddParam("OpenCV_2", "rgb(180,20,20)");
	ColorTable.m_InputColors.AddParam("OpenCV_1", "rgb(220,20,20)");
	ColorTable.m_InputColors.AddParam("OpenCV_0", "rgb(255,20,20)");

	ColorTable.m_InputColors.AddParam("GpuCV-GLSL_3", "rgb(20,140,20)");
	ColorTable.m_InputColors.AddParam("GpuCV-GLSL_2", "rgb(20,180,20)");
	ColorTable.m_InputColors.AddParam("GpuCV-GLSL_1", "rgb(20,220,20)");
	ColorTable.m_InputColors.AddParam("GpuCV-GLSL_0", "rgb(20,255,20)");
	ColorTable.m_InputColors.AddParam("GLSL_3", "rgb(20,140,20)");
	ColorTable.m_InputColors.AddParam("GLSL_2", "rgb(20,180,20)");
	ColorTable.m_InputColors.AddParam("GLSL_1", "rgb(20,220,20)");
	ColorTable.m_InputColors.AddParam("GLSL_0", "rgb(20,255,20)");

	ColorTable.m_InputColors.AddParam("GpuCV-CUDA_3", "rgb(20,20,140)");
	ColorTable.m_InputColors.AddParam("GpuCV-CUDA_2", "rgb(20,20,180)");
	ColorTable.m_InputColors.AddParam("GpuCV-CUDA_1", "rgb(20,20,220)");
	ColorTable.m_InputColors.AddParam("GpuCV-CUDA_0", "rgb(20,20,255)");
	ColorTable.m_InputColors.AddParam("CUDA_3", "rgb(20,20,140)");
	ColorTable.m_InputColors.AddParam("CUDA_2", "rgb(20,20,180)");
	ColorTable.m_InputColors.AddParam("CUDA_1", "rgb(20,20,220)");
	ColorTable.m_InputColors.AddParam("CUDA_0", "rgb(20,20,255)");
*/	
	vectorColorFilters.push_back(&ColorTable);

//add global informations about the Application version, used for the benchmarks
	SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance().SetCopyRight("Copyright Telecom SudParis");
	SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance().SetAppURL(GetGpuCVSettings()->GetURLHome());
	SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance().SetAppName("GpuCV");
	SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance().SetAppVersion(GetGpuCVSettings()->GetVersion());
	SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance().SetAppDescription("GpuCV benchmarks");

	//create a folder..
	std::string strPath, strFilename, strFilename_short, strCmd;
	strFilename = _filename;
	SGE::ReformatFilePath(strFilename);
	SGE::ParseFilePath(strFilename, &strPath, &strFilename);
	SGE::ParseFileType(strFilename, &strFilename_short);

	std::string strNewPath = cvgGetShaderPath();
	strNewPath += "/Benchmarks/";
	strNewPath += strPath;
	strNewPath+=strFilename_short;
	strNewPath+="/";
	SGE::ReformatFilePath(strNewPath);

	if(strNewPath!="")
	{
		strCmd = "mkdir ";
		strCmd += strNewPath;
		GPUCV_NOTICE("Creating report destination folder: "<< strCmd );
		system(strCmd.c_str());
	}

	//generate report
	strNewPath+="/";
	strNewPath+=strFilename;
	SG_TRC::HtmlReport newReport(strNewPath);
	//newReport.SetColorParams(ColorTable);
	newReport.SetColorFilters(&vectorColorFilters);
	//newReport.SetFilterParams(FilterTable);
	
	newReport.Draw<SG_TRC::SG_TRC_Default_Trc_Type>();
}

//======================================================
#define TESTBENCHLOOP_MACRO()\
{\
	_PROFILE_BLOCK("TracerMinTimeTest", NULL);\
}

void testBenchLoop(IplImage * src1)
{
	std::string localParams="";
	IplImage * src2 = NULL;
	GPUCV_FUNCNAME("TestBenchLoop");
	__CreateImages__(cvGetSize(src1) ,src1->depth, src1->nChannels, OperALL);

	_SW_benchLoop(TESTBENCHLOOP_MACRO(), localParams);
	_CV_benchLoop(TESTBENCHLOOP_MACRO(), localParams);
	_GPU_benchLoop(TESTBENCHLOOP_MACRO(), destGLSL,localParams);
	_CUDA_benchLoop(TESTBENCHLOOP_MACRO(),destCUDA, localParams);

	__ReleaseImages__();
}
//=============================================================================
//=============================================================================
//						Parsing commands
//=============================================================================
//=============================================================================
void PrintImageInfo(std::string _imgID)
{
	void* ImgPtrHexa = (void*)strtol(_imgID.data(), NULL, 16);
	GPUCV_DEBUG("Image Pointer ID: " << ImgPtrHexa);

	if(ImgPtrHexa==NULL)
	{
		GPUCV_ERROR("Wrong ID, can't convert it to HEXA" << _imgID);
		return;
	}
	DataContainer * TexPtr = GetTextureManager()->Get<DataContainer>(ImgPtrHexa);
	if(!TexPtr)
	{
		GPUCV_ERROR("No image with ID:" << _imgID);
	}
	else
	{
		TexPtr->Print();
	}
}
//======================================================
void changeSize(int w, int h) {

	// Prevent a divide by zero, when window is too short
	// (you cant make a window of zero width).
	if(h == 0)
		h = 1;

	float ratio = (float)(1.0* w / h);

	// Reset the coordinate system before modifying
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// Set the viewport to be the entire window
	glViewport(0, 0, w, h);

	// Set the correct perspective.
	gluPerspective(45,ratio,1,1000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0,0.0,5.0,
		0.0,0.0,-1.0,
		0.0f,1.0f,0.0f);


}
//======================================================
void ShowGpuCVStats()
{
	GPUCV_NOTICE("GpuCV statistics=============================");
	GPUCV_NOTICE("Texture manager size: " << GetTextureManager()->GetCount());
	GPUCV_NOTICE("Filter manager size: " <<  GetFilterManager()->GetCount());
	GPUCV_NOTICE("Shader manager size: " <<  GetFilterManager()->GetShaderManager().GetNumberOfShaders());
	GPUCV_NOTICE("=============================================");
	GPUCV_NOTICE("Texture manager==============================");
	GetTextureManager()->PrintAllObjects();
	GPUCV_NOTICE("==================================================");

	GPUCV_NOTICE("Filter manager=============================");
	GetFilterManager()->PrintAllObjects();
	GPUCV_NOTICE("==================================================");


	std::cout << "Current GPUCV settings:" << std::endl << GetGpuCVSettings()->GetOptionsDescription();
}
//=======================================================
void ShowSrcImgs(IplImage * Src1, IplImage * Src2, IplImage * mask)
{
	//Creating windows
	cvNamedWindow("Src1",1);
	cvNamedWindow("Src2",1);
	cvNamedWindow("Mask",1);
	cvShowImage  ("Src1",Src1);
	cvShowImage  ("Src2",Src2);
	cvShowImage  ("Mask",mask);
	cvWaitKey(0);
	//destroys windows
	cvDestroyWindow("Src1");
	cvDestroyWindow("Src2");
	cvDestroyWindow("Mask");
}
//======================================================
bool loadImage(IplImage * Img, char * filename/*=""*/)
{
	if (Img)
		cvRelease((void**)&Img);

	std::string ImageName = "data/pictures/";
	ImageName+= filename;

	IplImage * tempImg = NULL;
	if( (tempImg = cvLoadImage(ImageName.data(),1)) == 0 )
	{
		GPUCV_ERROR("Error loading image " << ImageName << ", the file may not exist...");
		return false;
	}
#if 0 //convert into b/w
	Img = cvgCreateImage(cvGetSize(tempImg), tempImg->depth, 1);
	cvCvtColor(tempImg, Img, CV_BGR2GRAY);
#else
	Img = tempImg;
#endif
	return true;
}
//======================================================

/** @ingroup GPUCVCONSOLE_GRP_PARAMETERS
\brief Enable/Disable the parameter given by the _arg string.
\param _arg => Parameter to edit.
\param _flag => new value.

* List of available parameter is:
<ul>
<li><b>'showimage'</b>: linked to variable ShowImage</li>
<li><b>'cpureturn'</b>: linked to variable AvoidCPUReturn</li>
<li><b>'datapreload'</b>: linked to variable DataPreloading</li>
<li><b>'glerrorcheck'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_CHECK</li>
<li><b>'glerrorexception'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_RISE_EXCEPTION</li>
<li><b>'internprofiling'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_PROFILING</li>
<li><b>'profileclass'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_PROFILING_CLASS</li>
<li><b>'profileoper'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_PROFILING_OPER</li>
<li><b>'profiletransfer'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_PROFILING_TRANSFER</li>
<li><b>'useopencv'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_USE_OPENCV</li>
<li><b>'debug'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG</li>
<li><b>'error'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_GLOBAL_ERROR</li>
<li><b>'warning'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_GLOBAL_WARNING</li>
<li><b>'notice'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_GLOBAL_NOTICE</li>
<li><b>'simulefilter'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_FILTER_SIMULATE</li>
<li><b>'shaderupdate'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_CHECK_SHADER_UPDATE</li>
<li><b>'shaderdebug'</b>: linked to GpuCV option GpuCVSettings::GPUCV_SETTINGS_FILTER_DEBUG</li>
<li><b>'opencv'</b>: linked to variable GpuCVSelectionMask</li>
<li><b>'glsl'</b>: linked to variable GpuCVSelectionMask</li>
<li><b>'cuda'</b>: linked to variable GpuCVSelectionMask</li>
<li><b>'ipp'</b>: linked to variable GpuCVSelectionMask</li>
<li><b>'scalar'</b>: linked to variable global_scalar</li>
<li><b>'mask'</b>: linked to variable GlobMask</li>
<li><b>'controloper'</b>: linked to variable ControlOperators</li>
</ul>
*/
bool EnableDisableSettings(std::string  _arg, bool _flag)
{
	//mainsampletest settings...
	if (_arg=="showimage")
	{
		ShowImage=_flag;
		return true;
	}
	else if (_arg=="cpureturn")
	{
		AvoidCPUReturn=_flag;
		return true;
	}
	else if (_arg=="datapreload")
	{
		DataPreloading=_flag;
		return true;
	}
	//Global settings...
	else if (_arg=="glerrorcheck")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_CHECK, _flag);
		return true;
	}
	else if (_arg=="glerrorexception")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_RISE_EXCEPTION, _flag);
		return true;
	}
	else if (_arg=="internprofiling")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_PROFILING, _flag);
		return true;
	}
	else if (_arg=="profileclass")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_PROFILING_CLASS, _flag);
		return true;
	}
	else if (_arg=="profileoper")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_PROFILING_OPER, _flag);
		return true;
	}
	else if (_arg=="profiletransfer")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_PROFILING_TRANSFER, _flag);
		return true;
	}
	else if (_arg=="useopencv")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_USE_OPENCV, _flag);
		return true;
	}
	else if (_arg=="debug")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_DEBUG, _flag);
		return true;
	}
	else if (_arg=="error")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_ERROR, _flag);
		return true;
	}
	else if (_arg=="warning")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_WARNING, _flag);
		return true;
	}
	else if (_arg=="notice")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GLOBAL_NOTICE, _flag);
		return true;
	}
	else if (_arg=="simulefilter")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_FILTER_SIMULATE, _flag);
		return true;
	}
	else if (_arg=="shaderupdate")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_CHECK_SHADER_UPDATE, _flag);
		return true;
	}
	else if (_arg=="shaderdebug")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_FILTER_DEBUG, _flag);
		return true;
	}
	else if (_arg=="opencv")
	{
		if(_flag)
			GpuCVSelectionMask = GpuCVSelectionMask |OperOpenCV;
		else if (GpuCVSelectionMask & OperOpenCV)
			GpuCVSelectionMask -= OperOpenCV;
		return true;
	}
	else if (_arg=="glsl")
	{
		if(_flag)
		{
			if(GCV::ProcessingGPU()->GetGLSLProfile()==GCV::GenericGPU::HRD_PRF_0)
			{
				GPUCV_WARNING("Your GPU is not compatible with the minimum profile required to run GLSL shaders or CUDA operators. GLSL can not be enabled!");
			}
			else
				GpuCVSelectionMask = GpuCVSelectionMask |OperGLSL;
		}
		else if (GpuCVSelectionMask & OperGLSL)
			GpuCVSelectionMask -= OperGLSL;
		return true;
	}
	else if (_arg=="cuda")
	{
#if 1//def _GPUCV_SUPPORT_CUDA
		if(_flag)
		{
			if(GCV::ProcessingGPU()->GetGLSLProfile()==GCV::GenericGPU::HRD_PRF_0 
				|| GCV::ProcessingGPU()->GetGLSLProfile() != GCV::GenericGPU::HRD_PRF_CUDA)
			{
				GPUCV_WARNING("Your GPU is not compatible with the minimum profile required to run GLSL shaders or CUDA operators. CUDA can not be enabled!");
				//disable flag in any case
				if (GpuCVSelectionMask & OperCuda)
					GpuCVSelectionMask -= OperCuda;
			}
			else
				GpuCVSelectionMask = GpuCVSelectionMask |OperCuda;
		}
		else if (GpuCVSelectionMask & OperCuda)
			GpuCVSelectionMask -= OperCuda;
#else
		GPUCV_ERROR("GPUCVConcole has not been compiled with '_GPUCV_SUPPORT_CUDA' flag enabled (mainSampleTest.h). CUDA is not supported in this build.");
		if (GpuCVSelectionMask & OperCuda)
			GpuCVSelectionMask -= OperCuda;
#endif
		return true;
	}
	else if (_arg=="ipp")
	{
		if(_flag)
			GpuCVSelectionMask = GpuCVSelectionMask |OperIPP;
		else if (GpuCVSelectionMask & OperIPP)
			GpuCVSelectionMask -= OperIPP;
		return true;
	}
	else if (_arg=="switch")
	{
		if(_flag)
			GpuCVSelectionMask = GpuCVSelectionMask |OperSW;
		else if (GpuCVSelectionMask & OperSW)
			GpuCVSelectionMask -= OperSW;
		return true;
	}
	else if (_arg=="switchlog")
	{
		SET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_SWITCH_LOG, _flag);
		return true;
	}
	else if (_arg=="scalar")
	{
		use_scalar = _flag;
		if(use_scalar)
		{
			global_scalar = new CvScalar;
			GPUCV_NOTICE("\nValue 4 float needed?");
			SGE::GetNextCommand(_arg, global_scalar->val[0]);
			for(int i=0;i<4;i++)
				SGE::GetNextCommand(_arg, global_scalar->val[i]);
		}
		else
		{
			if (global_scalar)
			{
				delete global_scalar;
				global_scalar = NULL;
			}
		}
		return true;
	}
	else if (_arg=="mask")
	{
		use_mask = _flag;
		if(use_mask)
			GlobMask = MaskBackup;
		else
			GlobMask = NULL;

		return true;
	}
	else if (_arg=="controloper")
	{
		ControlOperators = _flag;
		return true;
	}
	else if (_arg=="cxcore")
	{
		cxcore_test_enable = _flag;
		return true;
	}
	else if (_arg=="cv")
	{
		cv_test_enable = _flag;
		return true;
	}
	else if (_arg=="transfertest")
	{
		transfer_test_enable = _flag;
		return true;
	}
	return false;
}
//======================================================

bool resizeImage(IplImage ** Img, int width, int height, int interpolation/*=CV_INTER_LINEAR*/)
{
	GPUCV_DEBUG("Resizing image");
	if (Img)
	{
		cvgSynchronize(*Img);

		IplImage *TempImg=cvgCreateImage(cvSize(width, height),((IplImage *)(*Img))->depth, ((IplImage *)(*Img))->nChannels);
		cvResize(*Img, TempImg, interpolation);
		std::string Label = cvgGetLabel(*Img);
		cvgReleaseImage(Img);

		*Img = cvgCreateImage(cvSize(TempImg->width, TempImg->height), TempImg->depth, TempImg->nChannels);
		cvgSetLabel(*Img, Label);
		cvCopy(TempImg, *Img);
		cvgSetDataFlag<DataDsc_IplImage>(*Img, true, true);
		cvgReleaseImage(&TempImg);
		cvgSetOptions(*Img, DataContainer::UBIQUITY, true);
		cvgSetOptions(*Img, DataContainer::CPU_RETURN, false);

		cvgShowImageProperties(*Img);

		//cvgSetLocation<DataDsc_GLTex>(*Img, true);
		//cvgSetLocation<DataDsc_CUDA>(*Img, true);
		return true;
	}
	else
	{
		GPUCV_ERROR("Error resizing image");
		return false;
	}

	return true;
}
//======================================================
bool changeImageFormat2(IplImage ** Img, int channels, int depth, float scale)
{
	GPUCV_DEBUG("changeImageFormat");
	if (Img)
	{
		cvgSynchronize(*Img);
		GPUCV_NOTICE("Converting image '"<< cvgGetLabel(*Img)<< "' from :"<< GetStrCVPixelType(GetCVDepth(*Img)) << " to " << GetStrCVPixelType(depth));

		IplImage *TempImg=cvgCreateImage(cvGetSize(*Img),depth, channels);
		if(channels == (*Img)->nChannels)
		{
			cvConvertScale(*Img, TempImg,scale);
		}
		else
		{
			IplImage *TempImg3=NULL;
			IplImage *TempImg2=NULL;

			if(depth==IPL_DEPTH_32S || depth==IPL_DEPTH_32S)
			{
				//set the channels
				TempImg3=cvgCreateImage(cvGetSize(*Img),(*Img)->depth, 1);

				if((*Img)->nChannels==1)
					cvCopy(*Img, TempImg3);
				else if((*Img)->nChannels !=3)
					cvSplit(*Img, TempImg3, NULL, NULL, NULL);
				else
					cvCvtColor(*Img, TempImg3, CV_RGB2GRAY);

				//convert type
				TempImg2=cvgCreateImage(cvGetSize(*Img),depth, (TempImg3)->nChannels);
				cvConvertScale(TempImg3, TempImg2,scale);
				SWITCH_VAL(IplImage*, TempImg3, TempImg2);

			}
			else
			{
				//convert type first
				TempImg2=cvgCreateImage(cvGetSize(*Img),depth, (*Img)->nChannels);
				cvConvertScale(*Img, TempImg2,scale);

				//then set the channels
				TempImg3=cvgCreateImage(cvGetSize(*Img),depth, 1);

				if(TempImg2->nChannels==1)
					cvCopy(TempImg2, TempImg3);
				else if(TempImg2->nChannels !=3)
					cvSplit(TempImg2, TempImg3, NULL, NULL, NULL);
				else
					cvCvtColor(TempImg2, TempImg3, CV_RGB2GRAY);
			}

			switch(channels)
			{
			case 1:  cvCopy(TempImg3, TempImg);break;
			case 2:  cvMerge(TempImg3, TempImg3, NULL, NULL, TempImg);break;
			case 3:  cvMerge(TempImg3, TempImg3, TempImg3, NULL, TempImg);break;
			case 4:  cvMerge(TempImg3, TempImg3, TempImg3, TempImg3, TempImg);break;
			}
		}
		std::string Label = cvgGetLabel(*Img);
		cvgReleaseImage(Img);

		*Img = TempImg;
		cvgSetLabel(*Img, Label);
		//cvCopy(TempImg, *Img);
		cvgSetDataFlag<DataDsc_IplImage>(*Img, true, true);
		//cvgReleaseImage(&TempImg);
		cvgSetOptions(*Img, DataContainer::UBIQUITY, true);
		cvgSetOptions(*Img, DataContainer::CPU_RETURN, false);

		// cvgShowImageProperties(*Img);

		//cvgSetLocation<DataDsc_GLTex>(*Img, true);
		//cvgSetLocation<DataDsc_CUDA>(*Img, true);
		return true;
	}
	else
	{
		GPUCV_ERROR("Error resizing image");
		return false;
	}

	return true;
}
//======================================================
bool changeImageFormat(std::string & Command)
{
	std::string Curmd;
	SGE::GetNextCommand(Command, Curmd);

	std::string DebugCommand="changeImageFormat(";

	//get image
	IplImage ** CurrentImage=NULL;
	if(Curmd=="src1")
		CurrentImage = &GlobSrc1;
	else if(Curmd=="src2")
		CurrentImage = &GlobSrc2;
	else if(Curmd=="mask")
		CurrentImage = &MaskBackup;
	else
	{
		GPUCV_ERROR("Unkown command");
		return false;
	}
	DebugCommand+=Curmd+",";

	//get format
	SGE::GetNextCommand(Command, Curmd);
	int PixelFormat=0;
	float Scale = 1.;
	if(Curmd=="8u")
	{
		PixelFormat = IPL_DEPTH_8U;
	}
	else if(Curmd=="8s")
	{
		PixelFormat = IPL_DEPTH_8S;
	}
	else if(Curmd=="16s")
	{
		PixelFormat = IPL_DEPTH_16S;
		Scale = 256;
	}
	else if(Curmd=="16u")
	{
		PixelFormat = IPL_DEPTH_16U;
		Scale = 256;
	}
	else if(Curmd=="32s")
	{
		PixelFormat = IPL_DEPTH_32S;
		Scale = 256.;
	}
	else if(Curmd=="32f")
	{
		PixelFormat = IPL_DEPTH_32F;
		Scale =1./256.;
	}
	else if(Curmd=="64f")
	{
		PixelFormat = IPL_DEPTH_64F;
		Scale = 1./256.;
	}
	else
	{
		GPUCV_ERROR("Unkown command");
		return false;
	}
	DebugCommand+=Curmd+",";

	if((*CurrentImage)->depth >8)
		Scale = 1.;

	//get Channels
	SGE::GetNextCommand(Command, Curmd);
	int channels = 0;
	if(Curmd=="1")
		channels = 1;
	else if(Curmd=="2")
		channels = 2;
	else if(Curmd=="3")
		channels = 3;
	else if(Curmd=="4")
		channels = 4;
	else
	{
		GPUCV_ERROR("Unkown command");
		return false;
	}
	DebugCommand+=Curmd+")";

	GPUCV_NOTICE(DebugCommand);

	return changeImageFormat2(CurrentImage, channels, PixelFormat, Scale);
}
//======================================================

void testImageFormat(IplImage * src1, int channels, int depth, float scale)
{
	GPUCV_FUNCNAME("runCvtColor");
	__CreateImages__(cvGetSize(src1) ,8, channels, OperALL);
	__CreateWindows__();

	if(__CurrentOperMask&OperGLSL)
	{
		DataDsc_IplImage * DD_Ipl = new DataDsc_IplImage();
		DD_Ipl->_SetIplImage(&src1);

		DataDsc_GLTex * DD_Tex = new DataDsc_GLTex();
		DD_Tex->SetFormat(GL_LUMINANCE, GL_INT);
		DD_Tex->_SetSize(src1->width,src1->height);
		DD_Tex->Allocate();

		uchar * Data = new uchar[DD_Tex->GetDataSize()];
		DD_Tex->_Writedata((const DataDsc_Base::PIXEL_STORAGE_TYPE**)&Data, true);
		DD_Tex->_ReadData((DataDsc_Base::PIXEL_STORAGE_TYPE**) &Data, 0,0,src1->width,src1->height);
	}

	__ShowImages__();
	__ReleaseImages__();
}



void SynchronizeOper(enum OperType _operType, CvArr * _pImage)
{
#if _GCV_CUDA_EXTERNAL
	if(_pImage)
		cvgFlush(_pImage);
#else
		if(_operType & OperGLSL)
		{
			glFlush();
			glFinish();
		}
	#ifdef _GPUCV_SUPPORT_CUDA
		else if(_operType & OperCuda)
		{
			cvgCudaThreadSynchronize();
		}
	#endif
#endif
}

//=========================================================
float ControlResultsImages(CvArr * srcRef, CvArr * srcTest, const std::string & FctName, const std::string & Params)
{
	return ControlResultsImages(srcRef,srcTest,FctName.data(), Params.data());
}
float ControlResultsImages(CvArr * srcRef, CvArr * srcTest, const char* FctName, const char* Params)
{
	GPUCV_DEBUG("Memory usage: " << MainGPU()->GetMemUsage());



	SGE::LoggerAutoIndent LocalIndent;
//	if(ControlOperators && destCV!=NULL && !CV_IS_MAT(destCV))
	if (!srcRef || !srcTest)
		return 1000;//error

#if 0//debug
	cvNamedWindow("ControlResultsImages-ref");
	cvNamedWindow("ControlResultsImages-src");
	cvgShowImage("ControlResultsImages-ref", srcRef);
	cvgShowImage("ControlResultsImages-src", srcTest);
	cvWaitKey(0);
#endif


	{\
		cvgSynchronize(srcRef);
		cvgSynchronize(srcTest);
	//	GPUCV_LOG(GpuCVSettings::GPUCV_SETTINGS_DEBUG_MEMORY, "MEM", "\nMemory allocated after operator: " << DataDsc_Base::ms_totalMemoryAllocated);s



		CvArr * srcRefTmp = NULL;
		CvArr * srcTestTmp = NULL;

		if(GetnChannels(srcRef)==1)
			srcRefTmp = srcRef;
		else
		{
			srcRefTmp = cvCreateImage(cvGetSize(srcRef), GetCVDepth(srcRef), 1);
			cvCvtColor(srcRef, srcRefTmp, CV_BGR2GRAY);
		}

		if(GetnChannels(srcTest)==1)
			srcTestTmp = srcTest;
		else
		{
			srcTestTmp = cvCreateImage(cvGetSize(srcTest), GetCVDepth(srcTest), 1);
			cvCvtColor(srcTest, srcTestTmp, CV_BGR2GRAY);
		}

		CvArr * TestArr = (CV_IS_IMAGE_HDR(srcRefTmp))? (CvArr *) cvCloneImage((IplImage*)srcRefTmp):
							(CV_IS_MAT(srcRefTmp))? (CvArr *)cvCloneMat((CvMat*)srcRefTmp):NULL;

		cvSub(srcRefTmp, srcTestTmp, TestArr);
		CvScalar Sum = cvSum(TestArr);

        double diffNorm = cvNorm(TestArr);
        double refNorm = cvNorm(srcRefTmp);

	//	float AvgSum = (Sum.val[0] + Sum.val[1] + Sum.val[2] + Sum.val[3]);/*/4./GetWidth(destCV)/GetHeight(destCV);*/
	//	GPUCV_NOTICE(FctName << "("<< Params << ")" << ": Pixels Differents: "<< AvgSum << " / ratio: " << AvgSum/GetWidth(srcRefTmp)/GetWidth(srcRefTmp));

		float error; 
		if (refNorm == 0)
		{
			GPUCV_ERROR("Divide by 0");
			error =  100000;
		}
		else
			error = float(diffNorm/refNorm);

		if(srcRefTmp != srcRef)
			cvRelease(&srcRefTmp);

		if(srcTestTmp != srcTest)
			cvRelease(&srcTestTmp);

		cvRelease(&TestArr);//Memleak 11/03/10
		return error;
	}
}
