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

//see the following string 'help' for keyboard details...
#include "genswitch.h"
using namespace GCV;



/** Help text shown in the console when starting the application.
*/
string help ="";

//================================================================================
//Application main variables
std::string AppPath="";			//!< Path of the executable file.
bool bCleanHeaders = false;
bool bGenFileWrapper = false;

bool printError(const char* _errorMsg) 
{
	std::cout << _errorMsg << std::endl;
	return false;
}
//parse command line options...
/**
Command line options:
<table>
<tr>
<td>Option</td>
<td>Values</td>
<td>Description</td>
</tr>
<tr>
<td>-f videofilname</td>
<td>videofilname: Path to a video file</td>
<td>Open the given video file as input stream</td>
</tr>
<tr>
<td>-c cameraid</td>
<td>cameraid: id of the camera(from 0 to x)</td>
<td>Open the given video camera as input stream</td>
</tr>
<td>-w width - height</td>
<td>width & height: size of the video input</td>
<td>Try to open the video input using the given size, otherwise resize input to fit the given size</td>
</tr>


</table>
*/
bool parseCommandLine(int argc, char * argv[]) 
{

	std::string strCurrentCmd; 
	for(int i = 1; i< argc; i++)
	{
		strCurrentCmd= argv[i];
		
		if(strCurrentCmd=="-q")//quit application, used to see if it compiles and run. Further tests must be done
		{
			//printError("Exiting application");
			exit(0);
		}
		if(strCurrentCmd=="-g")//generate source files
		{
			bGenFileWrapper = true;
		}
		if(strCurrentCmd=="-c")//clean header files
		{
			bCleanHeaders = true;
		}
#if 0
		if(strCurrentCmd=="-h")//input height
		{
			if(i+1<argc)
			{
				ImageSize.height = atoi(argv[++i]);
				ImageSize_Next.height = ImageSize.height;
				continue;
			}
			else
				return printError("Error while parsing arguments: missing height value");
		}
		if(strCurrentCmd=="-c")//camera id
		{
			if(i+1<argc)
			{
				iCameraID = atoi(argv[++i]);
				continue;
			}
			else
				return printError("Error while parsing arguments: missing camera id value");
		}
		if(strCurrentCmd=="-f")//video file
		{
			if(i+1<argc)
			{
				VideoSeqFile = argv[++i];
				continue;
			}
			else
				return printError("Error while parsing arguments: missing video filename");
		}
#endif
	}

	return true;
}
//================================================================================
int main(int argc, char **argv) 
{
	GPUCV_FUNCNAME("genSwitch");
	try
	{
		//parse command line options...
		if(parseCommandLine(argc,argv)==false)
		{
			GPUCV_ERROR("Exiting application...");
			exit(-1);
		}
		//get application path
			std::string AppName= argv[0];
			std::string AppPath;
			GPUCV_DEBUG("argv[0]: " << AppName);
			SGE::ReformatFilePath(AppName);
			SGE::ParseFilePath(AppName, &AppPath);

			GPUCV_NOTICE("Current application location: " << AppPath);

			size_t iPos = AppPath.find("lib");
			if(iPos == std::string::npos)
				iPos = AppPath.find("bin");



			if(iPos != std::string::npos)
			{
				GPUCV_DEBUG("String found in path at pos:" << iPos);
				AppPath = AppPath.substr(0, iPos);
				AppPath += "data";
				GPUCV_NOTICE("Changing to: " << AppPath);
			}

			GPUCV_NOTICE("Set shader application path: " << AppPath);
			GetGpuCVSettings()->SetShaderPath(AppPath);
			//return AppPath;

		InitPaths();


		//create output path
		std::string strCmd;
		std::string strNewPath = AppPath;
		strNewPath += "/../bin/switch/";
		SGE::ReformatFilePath(strNewPath);
		if(strNewPath!="")
		{
			strCmd = "mkdir ";
			strCmd += strNewPath;
			GPUCV_NOTICE("Creating report destination folder: "<< strCmd );
			system(strCmd.c_str());
		}
		if(bCleanHeaders)cleanHeaders();
		if(bGenFileWrapper)ParseHeaderFiles();

		GPUCV_NOTICE("Current application path: " << AppPath);
	}
	catch(SGE::CAssertException &e)
	{//catch any exceptions and log message
		GPUCV_NOTICE("=================== Exception catched Start =================");
		GPUCV_NOTICE(e.what());
		GPUCV_NOTICE("=================== Exception catched End =================");
		GPUCV_NOTICE("Press a key to continue...");
		//		SGE::Sleep(5000);
		getchar();
	}
}