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
#ifndef __GPUCV_HARDWARE_TOOLS_H
#define __GPUCV_HARDWARE_TOOLS_H

#include "GPUCVHardware/config.h"
namespace GCV{


/** @addtogroup GPUCV_MACRO_GRP
@{*/
#define IS_MULTIPLE_OF(VAL, DIV)((VAL)%(DIV)==0)
#define IS_INTEGER(VAL)((float)(VAL)-(int)(VAL)==0)
#define SWITCH_VAL(TYPE, VAL1, VAL2){\
	TYPE tmpVal = VAL1;\
	VAL1 = VAL2;\
	VAL2 = tmpVal;\
}
/** @}*/ //GPUCV_MACRO_GRP

#ifdef __cplusplus
#define HTML_OPEN_TABLE		std::string ("<table border='1' width='100%'><tbody>")
#define HTML_CLOSE_TABLE	std::string ("</tbody></table>")
#define HTML_OPEN_ROW		std::string ("<tr>")
#define HTML_CLOSE_ROW	    std::string ("</tr>")
#define HTML_CELL(Cell)	    std::string ("<td>") + std::string(Cell) + std::string ("</td>")




/*!
*	\brief  read a file text and return it as a char *
*	\param  fn => file text name
*	\param  fdata => corresponding text store in a string
*	\return bool : true if success
*	\author www.lighthouse3d.com
*/
_GPUCV_HARDWARE_EXPORT
bool textFileRead(const std::string &fn, std::string &fdata);

/*!
*	\brief  write a char array in a file
*	\param  fn => destination file text name
*	\param  s => source char* data
*	\return int : status
*	\author www.lighthouse3d.com
*/
_GPUCV_HARDWARE_EXPORT
int textFileWrite(const std::string &fn, const std::string &s);

/*!
*	\brief Check if the given file properties correspond to the given time properties to find out if the file has been changed.
*	\param  _FileName => source filename.
*	\param	_FileClock1 => Time properties.
*	\return True if the filed has been modified.
*	\author Yannick Allusse
*/
_GPUCV_HARDWARE_EXPORT
bool isFileModified(std::string _FileName,  struct tm* _FileClock1);


/**	\brief Define output destination.
\sa SetLoggingOutput().
*/
enum LoggingOutput{
	LoggingToConsole,
	LoggingToFile
};

/**	\brief Set output destination.
\sa LoggingOutput().
*/
_GPUCV_HARDWARE_EXPORT void SetLoggingOutput(const int _output);


/**	\brief Control that the given directory exists.
\return true if found.
*/
_GPUCV_HARDWARE_EXPORT bool DirectoryExists(const std::string & _path);

/**
Function used to create a GPU plug in
*/
#if _GPUCV_SUPPORT_GPU_PLUGIN
template <typename GPUType>
bool createGPU(GenericGPU * _GpuTable, int * _gpuNbr)
{
	return new GPUType();
}
#endif


/** \return a string containing the CPU type and frequency.
*/ 
_GPUCV_HARDWARE_EXPORT std::string GetCpuName(void);


//! Return OS NAME
#ifdef _WINDOWS
#	define 	GetOSName()  "windows"
#elif defined (_LINUX)
#	define	GetOSName()  "linux"
#elif defined (_MACOSX)
#	define	GetOSName()  "macosx"
#else
#	define	GetOsName() "Unknown OS name"
#endif

	
//! Return ARCH name
#ifdef __x86_32__
#	define	GetArchName() "x32"
#elif defined (__x86_64__)
#	define	GetArchName() "x64"
#else
#	define	GetArchName() "Unknown arch type");
#pragma error "Unknown arch type, define __x86_64__ or __x86_32__"
#endif

	

//colored strings
#ifdef _WINDOWS
#include <windows.h>
#endif
// Copyleft Vincent Godin

inline std::ostream& blue(std::ostream &s)
{
#ifdef _WINDOWS

    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE); 
    SetConsoleTextAttribute(hStdout, FOREGROUND_BLUE
              |FOREGROUND_GREEN|FOREGROUND_INTENSITY);
#endif
    return s;
}

inline std::ostream& red(std::ostream &s)
{
#ifdef _WINDOWS
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE); 
    SetConsoleTextAttribute(hStdout, 
                FOREGROUND_RED|FOREGROUND_INTENSITY);
#endif
    return s;
}

inline std::ostream& green(std::ostream &s)
{
#ifdef _WINDOWS
	HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE); 
    SetConsoleTextAttribute(hStdout, 
              FOREGROUND_GREEN|FOREGROUND_INTENSITY);
#endif
    return s;
}

inline std::ostream& yellow(std::ostream &s)
{
#ifdef _WINDOWS
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE); 
    SetConsoleTextAttribute(hStdout, 
         FOREGROUND_GREEN|FOREGROUND_RED|FOREGROUND_INTENSITY);
#endif
    return s;
}

inline std::ostream& white(std::ostream &s)
{
#ifdef _WINDOWS
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE); 
    SetConsoleTextAttribute(hStdout, 
       FOREGROUND_RED|FOREGROUND_GREEN|FOREGROUND_BLUE);
#endif
    return s;
}

struct color {
#ifdef _WINDOWS
    color(WORD attribute):m_color(attribute){};
    WORD m_color;
#endif
};

template <class _Elem, class _Traits>
std::basic_ostream<_Elem,_Traits>& 
      operator<<(std::basic_ostream<_Elem,_Traits>& i, color& c)
{
#ifdef _WINDOWS
    HANDLE hStdout=GetStdHandle(STD_OUTPUT_HANDLE); 
    SetConsoleTextAttribute(hStdout,c.m_color);
#endif
    return i;
}
//end of colored strings

#ifndef _WINDOWS
/** \todo set beep for Linux ?????*/
#define Beep(a, b)
#endif

#endif//__cplusplus



}//namespace GCV
#endif
