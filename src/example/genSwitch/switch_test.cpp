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
#include <GPUCVSwitch/macro.h>
#include <GPUCVSwitch/Cl_FctSw.h>
#include <GPUCVSwitch/Cl_GenSw_Fct.h>
#include <GPUCVCore/ToolsTracer.h>

#ifdef _WINDOWS
	#define SED_EXE_PATH			"sed.exe"
#else
	#define SED_EXE_PATH			"sed"
#endif
#define CV_H_CLEAN				"etc/cv_h_clean.sed"
#define REMOVE_COMMENT_SED		"etc/remove_comment.sed"


using namespace GCV;
//==========================================



struct StHeaderToParse
{
	std::string inPath;
	std::string inName;
	std::string outPath;
	std::string outName;
	std::string dstPath_CPP;
	std::string dstPath_H;
};

#if CV_MAJOR_VERSION >1 //OPENCV 2

	std::string CV_PATH			= "/include/opencv/";
	#define CXCORE_PATH				CV_PATH
	#define HIGHGUI_PATH			CV_PATH
	std::string CVG_PATH		= "include/GPUCV";
	std::string CVGCU_PATH		= "include/GPUCVCuda";
	std::string TEMP_PATH		= "../bin/switch";
	std::string GpuCVRoot_PATH	= "";
	std::string IN_FILE_PATH	=		"%OPENCV_PATH%/include/opencv/";
	std::string OUT_FILE_PATH	=		"bin/switch";
#else
	std::string CV_PATH			= '"$(OPENCV_PATH)"/../resources/include/cv/include';
	std::string CXCORE_PATH		= "../resources/include/cxcore/include";
	std::string HIGHGUI_PATH	= "../resources/include/otherlibs/highgui";
	std::string CVG_PATH		= "include/GPUCV";
	std::string CVGCU_PATH		= "include/GPUCVCuda";
	std::string TEMP_PATH		= "bin/switch";
	std::string GpuCVRoot_PATH	= "";
	#define IN_FILE_PATH			"../resources/include/cv/include"
	#define OUT_FILE_PATH			"bin/switch"
#endif

StHeaderToParse FileToCleanTable[]={
	{CV_PATH,		"cv.h",		TEMP_PATH, "cv_switch",			"../src/lib/cv_switch/",		"../include/cv_switch/"}
	,
	{CXCORE_PATH,	"cxcore.h",	TEMP_PATH, "cxcore_switch",		"../src/lib/cxcore_switch/",	"../include/cxcore_switch/"}
	,{HIGHGUI_PATH,	"highgui.h",TEMP_PATH, "highgui_switch",	"../src/lib/highgui_switch/",	"../include/highgui_switch/"}
};


void cleanHeader(std::string SED_H_CLEAN, std::string Infilename, std::string IN_FILEPATH, std::string Outfilename, std::string OUT_FILEPATH)
{
	//"C:\Program Files\GnuWin32\bin\sed.exe" -f cv_h_clean.sed D:\saipriya_work\experimental\resources\include\cv\include\cv.h > D:\saipriya_work\experimental\gpucv\src\lib\GPUCVSwitch\cv_new.h
	string Command = "";
	string FileName = "";

	FileName = GpuCVRoot_PATH;
	FileName += SED_H_CLEAN;

	SGE::ReformatFilePath(FileName);
	std::string strLocCmd = "echo ";
	strLocCmd += FileName;
	system(strLocCmd.data());
	SG_AssertFile(SGE::CL_BASE_OBJ_FILE::FileExists(FileName),FileName, "does not exist");

	Command += SED_EXE_PATH;

	Command += " -f \"";
	Command += FileName;
	Command += "\" \"";

#if CV_MAJOR_VERSION >1 //OPENCV 2
	FileName = IN_FILEPATH;
	FileName += "/";
	FileName += Infilename;
	SGE::ReformatFilePath(FileName);
#else
	FileName = IN_FILEPATH;
	FileName += "/";
	FileName += Infilename;
	SGE::ReformatFilePath(FileName);
#endif
	Command += FileName;
	strLocCmd = "echo ";
	strLocCmd += FileName;
	system(strLocCmd.data());

	SG_AssertFile(SGE::CL_BASE_OBJ_FILE::FileExists(FileName),FileName, "does not exist");

	
	//ouput file
	FileName = OUT_FILEPATH;
	FileName += "/";
	FileName += Outfilename;
	SGE::ReformatFilePath(FileName);
	
	Command += "\" > \"";
	Command += FileName;
	Command += "\"";

	string cmdline = "";
	cmdline += Command.data();
	GPUCV_NOTICE(Outfilename);
	GPUCV_NOTICE("System command: "<< cmdline);
	system(cmdline.data());
	printf("\n");
	}
	
enum SEDLIKE_COMMAND
{
	SED_SUP,
	SED_REPLACE
};

void ManualCleanHeader(std::string Infilename, std::string IN_FILEPATH, 
					   std::string Outfilename, std::string OUT_FILEPATH,
					   std::string * _patterns,
					   int _patternsNbr
					   )
{
	SGE::ReformatFilePath(Infilename);
	SGE::ReformatFilePath(Outfilename);
	SGE::ReformatFilePath(IN_FILEPATH);
	SGE::ReformatFilePath(OUT_FILEPATH);

	string FileIn = IN_FILEPATH;
		FileIn += "/";
		FileIn += Infilename;
	SGE::ReformatFilePath(FileIn);
	std::string InBuffer;
	std::string tempSource;
	std::string DstBuffer;

	SG_AssertFile(GCV::textFileRead(FileIn, InBuffer), FileIn, "no file");


	size_t Pos = -1;
	size_t Start = 0;
	size_t End = 0;
	size_t Command =-1;
	tempSource = InBuffer;
	std::string FirstPart, LastPart;
	bool Loop = true;

	for(int i =0; i < _patternsNbr; i++)
	{
		Loop = true;
		DstBuffer="";
		//SED_SUP
		Start=_patterns[i].find("'*'");
		if(Start!=std::string::npos)
		{
			FirstPart= _patterns[i].substr(0,Start);
			LastPart = _patterns[i].substr(Start+strlen("'*'"), _patterns[i].size());
			Command = SED_SUP;
		}
		else
		{
		//SED_REPLACE
			Start=_patterns[i].find("'->'");
			if(Start!=std::string::npos)
			{
				FirstPart= _patterns[i].substr(0,Start);
				LastPart = _patterns[i].substr(Start+strlen("'->'"), _patterns[i].size());
				Command = SED_REPLACE;
			}

		}

		do
		{
			
			//Find the first tabulation
			switch(Command)
			{
				case SED_SUP:
					Start	= tempSource.find(FirstPart);
					End		= tempSource.find(LastPart);
					if (Start==std::string::npos || End==std::string::npos || End==0)
					{	
						Loop = false;
					}
					else
					{
						DstBuffer += tempSource.substr(0,Start);
						tempSource = tempSource.substr(End+1,tempSource.size());
					}
					break;
				case SED_REPLACE:
					Start	= tempSource.find(FirstPart);
					if (Start==std::string::npos)
					{	
						Loop = false;
					}
					else
					{
						DstBuffer += tempSource.substr(0,Start);
						DstBuffer += LastPart;
						tempSource = tempSource.substr(Start+FirstPart.size(),tempSource.size());
					}
					break;
				default:
					Start = tempSource.find(_patterns[i]);
					if (Start==std::string::npos)
					{
						Loop = false;
					}
					else
					{
						DstBuffer += tempSource.substr(0,Start);
						tempSource = tempSource.substr(Start+_patterns[i].size(),tempSource.size());
					}
			}
		}
		while(Loop);
		DstBuffer += tempSource;
		tempSource = DstBuffer;
	}
	


	string FileOut =OUT_FILEPATH;
		FileOut += "/";
		FileOut += Outfilename;
	SG_AssertFile(GCV::textFileWrite(FileOut.data(), DstBuffer.data()), FileOut, "no file");
	                
}

void InitPaths()
{
	char * OpenCVPath = getenv("OPENCV_PATH");
	CV_PATH = OpenCVPath + CV_PATH;

	int j =0;
	std::string localAppPath = GetGpuCVSettings()->GetShaderPath();
	for(int i = 0; i < sizeof(FileToCleanTable)/sizeof(StHeaderToParse); i++)
	{//update destination path 
		FileToCleanTable[i].dstPath_CPP = localAppPath + FileToCleanTable[i].dstPath_CPP;
		FileToCleanTable[i].dstPath_H = localAppPath + FileToCleanTable[i].dstPath_H;
		FileToCleanTable[i].inPath		= localAppPath + FileToCleanTable[i].inPath;
		FileToCleanTable[i].outPath		= localAppPath + FileToCleanTable[i].outPath;
	}
}

void cleanHeaders()
{
	GpuCVRoot_PATH = GetGpuCVSettings()->GetShaderPath();
	GpuCVRoot_PATH += "/../"; 
	SGE::ReformatFilePath(GpuCVRoot_PATH);
	//Create destination folder:
	string Command = "mkdir ";
	Command += GpuCVRoot_PATH;
	Command += "bin/switch";
	system(Command.data());


	std::string RulesFileTable []={
		"etc/sed/tab.sed"
		,"etc/sed/comment.sed"
		,"etc/sed/preproc.sed"
		,"etc/sed/c_code.sed"
		,"etc/sed/opencv_code.sed"
		,"etc/sed/misc.sed"
		,"etc/sed/emptylines.sed"	
	};

#if CV_MAJOR_VERSION >1 //OPENCV 2
	std::string PatternsTableCV[] = {
		//,"typedef'*';"
		"{'*'}"
		,",\n'->', " //concat line ending by ,\n
		,"; '->';\n" //lines that have '; ' split and return to new line  
		,"                 '->' " //remove spaces
		,"             '->' " //remove spaces
		,"  '->' " //remove double spaces
		,"  '->' " //remove double spaces
		,"  '->' " //remove double spaces
		,"}'->'"
		,"{'->'"
		,"struct CV_EXPORTS CvModule struct CV_EXPORTS CvModule'->'"
		,"struct CV_EXPORTS CvType struct CV_EXPORTS CvType'->'"
		,"DOUBLE'->'double"

/*	
		//,"\\\n'*' "
		
		//,"int accumulate CV_DEFAULT(0), const CvArr* mask CV_DEFAULT(NULL) )" 
		,"(\n'->'( "
		,"( '->'("
		,"\\'->' "
		//opencv code that remain after SED
		,"(new_cn), (new_dims), (new_sizes))'->'"
		,"cvSetAdd(set_header, NULL, (CvSetElem**)&elem );'->'"
		//==================
		//,"combine the flag with one of the above CV_THRESH_* values *"
		,"}"
*/
		};
#else//V1
	std::string PatternsTableCV[] = {
		//,"typedef'*';"
		"{'*'}"
		"/*'*'*/"
		//,"\\\n'*' "
		,"}'->'\n"
		//,"int accumulate CV_DEFAULT(0), const CvArr* mask CV_DEFAULT(NULL) )" 
		,",\n'->', " 
		,"(\n'->'( "
		,"( '->'("
		,"DOUBLE'->'double"
		,"\\'->' "
		//opencv code that remain after SED
		,"(new_cn), (new_dims), (new_sizes))'->'"
		,"cvSetAdd(set_header, NULL, (CvSetElem**)&elem );'->'"
		//==================
		//,"combine the flag with one of the above CV_THRESH_* values *"
		,"}"
	};
#endif


	GPUCV_NOTICE("\nGenerating new header free files...");


	int j =0;
	for(int i = 0; i < sizeof(FileToCleanTable)/sizeof(StHeaderToParse); i++)
	{
		j =0;
#if 1//do cleaning
		//do main cleaning with SED
		//if(i==0)
			cleanHeader(RulesFileTable[0],	
				FileToCleanTable[i].inName,		CV_PATH/*FileToCleanTable[i].inPath*/,
				FileToCleanTable[i].outName+".h." + SGE::ToCharStr(0),	FileToCleanTable[i].outPath);
			
		j++;

		for(; j < sizeof(RulesFileTable)/sizeof(std::string); j++)
		{
			cleanHeader(RulesFileTable[j],	
				FileToCleanTable[i].outName+".h." + SGE::ToCharStr(j-1),	FileToCleanTable[i].outPath,
				FileToCleanTable[i].outName+".h." + SGE::ToCharStr(j),	FileToCleanTable[i].outPath);
		}
		//do some manual clean that SED could not do, why?
		ManualCleanHeader(FileToCleanTable[i].outName+".h." + SGE::ToCharStr(j-1),	FileToCleanTable[i].outPath,	
				FileToCleanTable[i].outName+".h.final",	FileToCleanTable[i].outPath, 
				PatternsTableCV, 
				sizeof(PatternsTableCV)/sizeof(std::string));
#endif
		//list all the function from this header file
	//	CL_GenSwFn::AddObjsToFns(FileToCleanTable[i].outName+".final",FileToCleanTable[i].outPath,FileToCleanTable[i].outName+".cpp","../src/lib/GPUCVSwitch/"/*FileToCleanTable[i].outPath*/);
	}


}

void ParseHeaderFiles()
{
	int j =0;
	GPUCV_NOTICE("\nParsing OpenCV header files...");
	for(int i = 0; i < sizeof(FileToCleanTable)/sizeof(StHeaderToParse); i++)
	{
		GPUCV_NOTICE("\n\tParsing "<< FileToCleanTable[i].outName);
		CL_GenSwFn::GetFctObjMngr()->DeleteAllLocal();
		//list all the function from this header file
		CL_GenSwFn::AddObjsToFns_CPP(FileToCleanTable[i].outName+".h.final",FileToCleanTable[i].outPath,FileToCleanTable[i].outName+".cpp", FileToCleanTable[i].dstPath_CPP);
		CL_GenSwFn::AddObjsToFns_H(FileToCleanTable[i].outName+".h", FileToCleanTable[i].dstPath_H);
		CL_GenSwFn::AddObjsToFns_H_WRAPPER(FileToCleanTable[i].outName, FileToCleanTable[i].dstPath_H);
	}
}
//==========================================
