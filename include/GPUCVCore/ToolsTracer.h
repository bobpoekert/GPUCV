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



//! Define structures and function to generate dynamic tracing data and get result from the application
#if 0//ndef __GPUCV_CORE_TOOLS_TRACER_H
#define __GPUCV_CORE_TOOLS_TRACER_H

#include <string>
#include <time.h>
#include <sys/timeb.h> 
#include <GPUCVCore/config.h> 

/** @ingroup GPUCV_MACRO_BENCH_GRP
*  @{*/


#ifdef _WINDOWS
#include <winsock.h>
#else
#include <sys/time.h>
#endif

#include <definitions.h>
#include <vector>
#include <iostream>





//==========================================
//	constants
//==========================================
#define TRC_MAX_NBR_FCT			4096
#define TRC_MAX_NBR_RECORDS		16384
#define TRC_MAX_NBR_TIMES		16384
#define TRC_FCT_CV 1
#define TRC_FCT_CVG 2


int Ygettimeofday(struct timeval * tp);
void cleartimeval(struct timeval * t1);

/**
*	\brief Class to manipulate system time
*	\author Yannick ALLUSSE
*  Manipulate system time, compatible with MS Windows XP and 2000, UNIX like and MAC OS
*/
class CL_TimerVal
{
public :
	struct timeval Time;	//!< Store the time value recorded
	CL_TimerVal(void){};	//!< Constructor
	~CL_TimerVal(void){}; //!< Destructor
	void   operator  =( CL_TimerVal  Time2);//!< operator =.
	void   operator +=( CL_TimerVal Time2);//!< operator += for adding times.
	void   operator -=( CL_TimerVal Time2);//!< operator -= for subtracting times.
	void   operator /=(float a);//!< operator /= for dividing times.
	void   operator *=(float a);//!< operator *= for multiplying times.
	//      void   operator /=(CL_TimerVal Time2);//!< operator = for dividing times.
	//      void   operator *=(CL_TimerVal Time2);//!< operator = for multiplying times.
	bool   operator <(CL_TimerVal Time2);//!< operator < for comparing times.
	bool   operator >(CL_TimerVal Time2);//!< operator > for comparing times.

	void GetTime(void);//!< Get the current system time from boot.
	long GetTimeInUsec(void);//!< Return the time value stored in Time as a long.
	void PrintTime();//!< Print the time value as "\n%ds %d �s"
	void Clear();//!< Clear the time value
};
bool   operator ==(CL_TimerVal Time1, CL_TimerVal  Time2);//!< operator == for comparing times.


//================================================================
//! struct FCT_TRACER stores profiling information of one function
/*! There is differences between _WINDOWS code and UNNIX_LIKE code, the time in [_WINDOWS] is in sec and msec, 
and the time in [UNIX_LIKE] is in sec and usec.
*/
//================================================================
class CL_FCT_SUB_ARGS
{
public:
	std::vector<CL_TimerVal *> CLTimes;
	string  Params;
	int NbrOfTime;
	unsigned int SizeIn[2];
	unsigned int SizeOut[2];
	CL_TimerVal  StartTime, StopTime, TotalTime;	//!< Starting and stopping Time of the function in sec and �sec
	CL_TimerVal MaxTime, MinTime;
	string Type;
	CL_FCT_SUB_ARGS(string _Type, string _Params, unsigned int _widthIn=0, unsigned int _heightIn=0, unsigned int _widthOut=0, unsigned int _heightOut=0);
	~CL_FCT_SUB_ARGS();	
	void AddRecord();//int _type, string _Params=0, GLuint _widthIn=0, unsigned int _heightIn=0, unsigned int _widthOut=0, unsigned int _heightIn=0);
	void Clear();
	CL_TimerVal * GetLastTime();
	void CleanMaxTime();
};


/**
*	\brief Class to store profiling data about one function
*	\author Yannick ALLUSSE
*  Class to store profiling data about one function
*/
class FCT_TRACER
{
public:

	int TimeID;
	int TimeNbr;
	void NextTime();
	CL_TimerVal MaxTime, MinTime;

	~FCT_TRACER(void);
	void StartTrc();	
	void StopTrc();
	void Clear();

	string FCT_NAME;	    //!< Name of the function traced
	std::vector <CL_FCT_SUB_ARGS*> Records;		//!< Array of the last processor time calculated, in ms
	FCT_TRACER(string _FctName);//, bool _CVG, char *_Params, GLuint _widthIn=0, GLuint _heightIn=0, GLuint _widthOut=0, GLuint _heightIn=0);
	//void AddRecord(int _type, const char * _Params, unsigned int _widthIn=0, unsigned int _heightIn=0, unsigned int _widthOut=0, unsigned int _heighOut=0);
	void AddRecord(string _type, const char * _Params, unsigned int _widthIn=0, unsigned int _heightIn=0, unsigned int _widthOut=0, unsigned int _heightOut=0);
	int getNbrOfParams();

	void Sort();
	CL_TimerVal *GetLastTime();
	char* GetIndent();
	int ActRecID;
	void CleanMaxTime();
};
//=================================================================

bool operator<(const CL_FCT_SUB_ARGS& x, const CL_FCT_SUB_ARGS& y);
bool operator==(const CL_FCT_SUB_ARGS& x, const CL_FCT_SUB_ARGS& y);
bool operator>(const CL_FCT_SUB_ARGS& x, const CL_FCT_SUB_ARGS& y);



struct STBenchValue{
	unsigned long max;
	unsigned long min;
	unsigned long avg;
	unsigned long total;
	int	 width;
	int height;
	int nbrOfTime;
	std::string type;
	double pointPos[2];
};

/*!
*	\brief Class APPLI_TRACER stores every class FCT_TRACER created and more profiling data
*	\author Yannick ALLUSSE
*  generate profiling informations about functions
*/
class CL_APPLI_TRACER
{
public:
	std::vector<FCT_TRACER *> FctTrc;//!< Table of all the function profiling data, with different parrallele mode
	CL_TimerVal Max;
	bool Console;

	CL_APPLI_TRACER();
	FCT_TRACER * AddRecord(string _FctName, string _type="", std::string _Params="", unsigned int _widthIn=0, unsigned int _heightIn=0, unsigned int _widthOut=0, unsigned int _heightOut=0);
	FCT_TRACER * AddRecord(string _FctName, string _type="", float _Params=0, 	  unsigned int _widthIn=0, unsigned int _heightIn=0, unsigned int _widthOut=0, unsigned int _heightOut=0);
	int GenerateTextFile(char * Filename);
	void SetMax(CL_TimerVal * _Max);
	void Clear();
	void Sort();
	int FctNbr;
	int MaxFctNbr;
	FILE * File;									//!< File handler used to write profiling data to a text file
	~CL_APPLI_TRACER(void);

	void OpenFile(char * filname);
	void AddFctToFile(FCT_TRACER * CurFct, int MaxValues);
	void AddFctToHtmlFile(FCT_TRACER * CurFct,  long int MaxValue,std::string SvgPath="");

	void CloseFile();
	int GenerateHtmlFile(char * Filename);
	int GenerateHtmlFile2(char * Filename,
		char * Title,
		char * Start,
		char * svgPath
		);
	int  GenerateSvgFile(char * _path);
	int AddFctToSvgFile(char * _path, FCT_TRACER * CurFct, CL_FCT_SUB_ARGS * ActiveRec, std::vector<STBenchValue*> & dataVector, int start, int end);
	void EnableConsole();
	void DisableConsole();
	bool GetConsoleStatus();
	void CleanMaxTime();
};
//================================================================

CL_APPLI_TRACER * AppliTracer();//int _ModeNbr=TRC_MAX_PROFILE_MODE, int _FctNbr=TRC_MAX_NBR_FCT);

/** @} */ // end of GPUCV_MACRO_BENCH_GRP

/** @addtogroup GPUCV_MACRO_BENCH_GRP */
/** @{*/

#if _GPUCV_PROFILE
#	define _BENCH(FctName, Type, Fct, Params, SizeX, SizeY){\
	FCT_TRACER * FctTrc = AppliTracer()->AddRecord(FctName, Type, Params,SizeX, SizeY);\
	FctTrc->StartTrc();\
	Fct;\
	FctTrc->StopTrc();\
}
#else
#	ifdef _BENCH
#		undef _BENCH
#	endif
#	define _BENCH(FctName, Type, Fct, Params, SizeX, SizeY){Fct;}
#endif

#if (_GPUCV_PROFILE_GL==1)
#	define _BENCH_GL(FctName, Fct, Params, SizeX, SizeY){Fct;}
	/*\
	glFlush();\
	glFinish();\
	FCT_TRACER * FctTrc = AppliTracer()->AddRecord(FctName, "GL", Params,SizeX, SizeY);\
	FctTrc->StartTrc();\
	Fct;\
	glFlush();\
	glFinish();\
	FctTrc->StopTrc();\
	}
	*/
#else
#	ifdef _BENCH_GL
#		undef _BENCH_GL
#	endif	
#	define _BENCH_GL(FctName, Fct, Params, SizeX, SizeY){Fct;}
#endif
/** @} */ // end of GPUCV_MACRO_BENCH_GRP 

#endif

