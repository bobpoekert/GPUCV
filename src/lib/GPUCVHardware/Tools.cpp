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
#include "GPUCVHardware/Tools.h"
#include "GPUCVHardware/hardware.h"
#include "GPUCVHardware/GlobalSettings.h"
#include <SugoiTools/exceptions.h>
#ifdef _WINDOWS
	#include <iostream>
	#include <intrin.h>
#endif
namespace GCV{

//=================================================
bool textFileRead(const std::string &fn, std::string &fdata)
{
	FILE *fp;
	char *content = NULL;

	size_t count;
	std::string newPath = fn;
	SGE::ReformatFilePath(newPath);//used to make sure path is compatible with linux or windows
	fp = fopen(newPath.c_str(), "r");
	if (fp == NULL)//not found..
	{
		newPath = GetGpuCVSettings()->GetShaderPath() + newPath;
		fp = fopen(newPath.c_str(), "r");
	}
	if(fp==NULL)
	{
		GPUCV_ERROR("Could not load file '" << newPath << "'");
		//SG_AssertFile(fp, newPath, "textFileRead()");
		return false;
	}
	fdata = "";

	fseek(fp, 0, SEEK_END);
	count = ftell(fp);

	fclose(fp);

	if (newPath !="") {
		fp = fopen(newPath.c_str(),"rt");

		if (fp != NULL) {
			if (count > 0) {
				content = (char *)malloc(sizeof(char) * (count+1));
				count = fread(content,sizeof(char),count,fp);
				content[count] = '\0';
				fdata = content;
				delete []content;
			}
			fclose(fp);
		}
	}
	return content != NULL;
}
//=================================================
int textFileWrite(const std::string & fn, const std::string &s) {

	FILE *fp;
	int status = 0;

	if (fn != "") {
		fp = fopen(fn.data(),"w");
		if(fp==NULL)
		{
			GPUCV_ERROR("Could not write to file '" << fn << "'");
			//SG_AssertFile(fp, newPath, "textFileRead()");
			return 0;
		}

		//SG_AssertFile(fp != NULL,fn, "textFileWrite()");
		if (fwrite(s.data(),sizeof(char),s.size(),fp) == s.size())
			status = 1;
		fclose(fp);
	}
	return(status);
}
//=================================================
void SetLoggingOutput(const int _output)
{
	if(_output == LoggingToConsole)
	{
		SGE::ILogger::SetLogger(new SGE::CLoggerDebug());
	}
	else if (_output == LoggingToFile)
	{
		SGE::ILogger::SetLogger(new SGE::CLoggerFile ("output.log"));
	}
}
//=================================================
bool isFileModified(std::string FileName,  struct tm* OrigFileClock)
{
	struct tm* CurFileClock=NULL;
	struct stat attrib;			// create a file attribute structure
	stat(FileName.data(), &attrib);		// get the attributes of afile.txt
	CurFileClock = gmtime(&(attrib.st_mtime));	// Get the last modified time and put it into the time structure

	if (!CurFileClock)
		return false;

	if (CurFileClock->tm_sec != OrigFileClock->tm_sec) /* seconds after the minute - [0,59] */
		return true;
	if (CurFileClock->tm_min != OrigFileClock->tm_min)/* minutes after the hour - [0,59] */
		return true;
	if (CurFileClock->tm_hour != OrigFileClock->tm_hour) /* hours since midnight - [0,23] */
		return true;
#if 0//sec, min and hours should be far enought to test in our case...
	if (CurFileClock->tm_mday != OrigFileClock->tm_mday)/* day of the month - [1,31] */
		return true;
	if (CurFileClock->tm_mon != OrigFileClock->tm_mon)/* months since January - [0,11] */
		return true;
	if (CurFileClock->tm_year != OrigFileClock->tm_year)/* years since 1900 */
		return true;
	if (CurFileClock->tm_wday != OrigFileClock->tm_wday)/* days since Sunday - [0,6] */
		return true;
	if (CurFileClock->tm_yday != OrigFileClock->tm_yday)/* days since January 1 - [0,365] */
		return true;
	if (CurFileClock->tm_isdst != OrigFileClock->tm_isdst) /* daylight savings time flag */
		return true;
#endif
	return false;
}
//================================================
std::string GetCpuName(void)
{
#ifdef _WINDOWS
		char CPUBrandString[49];
		__cpuid((int*)CPUBrandString, 0x80000002);
		__cpuid((int*)(CPUBrandString+16), 0x80000003);
		__cpuid((int*)(CPUBrandString+32), 0x80000004);
		CPUBrandString[48] = 0;
#else
	std::string strCPUDescription;
	std::string CPUBrandString="Unknown";
	system("cat /proc/cpuinfo | grep 'model name' >> cpuname.txt");
	if(textFileRead("cpuname.txt",strCPUDescription))
	{
		std::string strToRemove="model name	: ";
		CPUBrandString = strCPUDescription.substr(strToRemove.size(),strCPUDescription.size()-strToRemove.size());
		int endlPos = strCPUDescription.find("\n");
		if(endlPos != std::string::npos)
		{
			if(endlPos>47)
				endlPos=47;
			CPUBrandString = CPUBrandString.substr(0, endlPos);
		}
	}
#endif
	return CPUBrandString;
}
//=============================================================

bool DirectoryExists(const std::string & _path)
{
	std::string localPath=_path;

	SGE::ReformatFilePath(localPath);

    if ( access( localPath.c_str(), 0 ) == 0 )
    {
        struct stat status;
        stat( localPath.c_str(), &status );

        if ( status.st_mode & S_IFDIR )
        {
            GPUCV_DEBUG("DirectoryExists()->Given path '"<< localPath <<"' exists.");
			return true;
        }
        else
        {
            GPUCV_WARNING("DirectoryExists()->Given path '"<< localPath <<"' is a file.");
			return false;
        }
    }
    else
    {
        GPUCV_DEBUG("DirectoryExists()->Given path '"<< localPath <<"' does not exist.");
		return false;
    }
}

}//namespace GCV
