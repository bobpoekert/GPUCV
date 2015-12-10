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
#include <GPUCVCore/ToolsTracer.h>


#if 0//DEPRECATED
/** @addtogroup GPUCV_MACRO_BENCH_GRP
*  @{*/

#define BENCH_COLOR_OPENCV "red"
#define BENCH_COLOR_IPP "pink"
#define BENCH_COLOR_GPUCV "green"
#define BENCH_COLOR_CUDA "blue"
#define BENCH_COLOR_HTML_OPENCV "FF0000"
#define BENCH_COLOR_HTML_GPUCV  "009900"
#define BENCH_COLOR_HTML_CUDA   "0000FF"
#define BENCH_COLOR_HTML_IPP    "FF00FF"

#define BENCH_COPYRIGHT_STRING_HTML    "&copy;Telecom & Management SudParis 2008"
#define BENCH_COPYRIGHT_STRING_SVG    "Copyright Telecom and Management SudParis 2008"


//==============================================

//Classical exports values

//Commented out by PH on 23/1/06:
//#ifdef _WINDOWS
//	//#define _GPUCV_CORE_EXPORT "C" __declspec(dllexport)
//
//	#ifdef _VCC2005
//		//#define _CRT_SECURE_NO_DEPRECATE // To disable deprecation (!)
//		#pragma warning(once : 4996) //some C++ functions were declared deprecated, show only once by function.
//	#endif
//#else
//	//#define _GPUCV_CORE_EXPORT
//#endif
////==============================================

#ifndef _WINDOWS
typedef long int LONGLONG;
#endif

#define MaxIndentSize  1024
char IndentString[MaxIndentSize];

#define __log(str1, str2) //printf("%s %s\n", str1, str2);


void  CL_TimerVal:: operator =( CL_TimerVal Time2)
{
	this->Time.tv_sec = Time2.Time.tv_sec;
	this->Time.tv_usec = Time2.Time.tv_usec;
}

bool  operator ==(CL_TimerVal& Time1, CL_TimerVal&  Time2)
{
	if ((Time1.Time.tv_sec == Time2.Time.tv_sec)
		&& (Time1.Time.tv_usec == Time2.Time.tv_usec))
		return true;
	return false;
}

void   CL_TimerVal::operator +=( CL_TimerVal  Time2)
{
	//micro seconde test
	this->Time.tv_usec += Time2.Time.tv_usec;
	while (this->Time.tv_usec > 1000000)
	{
		this->Time.tv_sec ++;
		this->Time.tv_usec -= 1000000;
	}
	//seconde test
	this->Time.tv_sec += Time2.Time.tv_sec;
}

void   CL_TimerVal::operator -=( CL_TimerVal Time2)
{
	//micro seconde test
	this->Time.tv_usec -= Time2.Time.tv_usec;
	while (this->Time.tv_usec < 0)
	{
		this->Time.tv_sec --;
		this->Time.tv_usec += 1000000;
	}
	//seconde test
	this->Time.tv_sec -= Time2.Time.tv_sec;
}

void   CL_TimerVal::operator /=(float a)
{
	long TempSec=0;
	this->Time.tv_usec = (long)(this->Time.tv_usec/a);
	TempSec = this->Time.tv_sec;
	this->Time.tv_sec = (long)(this->Time.tv_sec/a);
	TempSec -= this->Time.tv_sec;
	this->Time.tv_usec = (long)(this->Time.tv_usec+(TempSec/ a) * 1000000);
	while (this->Time.tv_usec > 1000000)
	{
		this->Time.tv_sec ++;
		this->Time.tv_usec -= 1000000;
	}
}

void   CL_TimerVal::operator *=(float a)
{
	this->Time.tv_sec = (long)(this->Time.tv_sec * a);
	this->Time.tv_usec = (long)(this->Time.tv_usec*a);
	while (this->Time.tv_usec > 1000000)
	{
		this->Time.tv_sec ++;
		this->Time.tv_usec -= 1000000;
	}
}

//void   CL_TimerVal::operator /=(CL_TimerVal Time2){}

//void   CL_TimerVal::operator *=(CL_TimerVal Time2){}

long   CL_TimerVal::GetTimeInUsec(void)
{  return (this->Time.tv_sec * 1000000+this->Time.tv_usec);}


bool   CL_TimerVal::operator <(CL_TimerVal Time2)
{
	if (this->Time.tv_sec < Time2.Time.tv_sec )
		return true;
	else if (this->Time.tv_sec > Time2.Time.tv_sec)
		return false;
	else if(this->Time.tv_usec < Time2.Time.tv_usec)
		return true;
	else if(this->Time.tv_usec > Time2.Time.tv_usec)
		return false;
	return false;
}

bool   CL_TimerVal::operator >(CL_TimerVal Time2)
{
	if (this->Time.tv_sec > Time2.Time.tv_sec )
		return true;
	else if (this->Time.tv_sec < Time2.Time.tv_sec)
		return false;
	else if(this->Time.tv_usec > Time2.Time.tv_usec)
		return true;
	else if(this->Time.tv_usec < Time2.Time.tv_usec)
		return false;
	return false;
}

void CL_TimerVal::PrintTime()
{
	printf("%d'%6.d", (int)Time.tv_sec, (int)Time.tv_usec);
}
void CL_TimerVal::Clear()
{
	Time.tv_sec = 0;
	Time.tv_usec = 0;
}

void CL_TimerVal::GetTime(void)
{
	Ygettimeofday(&this->Time);
}
//==============================================================================
//==============================================================================

CL_FCT_SUB_ARGS::CL_FCT_SUB_ARGS(string _Type, string _Params, unsigned int _widthIn, unsigned int _heightIn, unsigned int _widthOut, unsigned int _heightOut)
{
	Clear();
	Params = _Params;
	Type = _Type;
	SizeIn[0] =_widthIn;
	SizeIn[1] =_heightIn;
	SizeOut[0] =_widthOut;
	SizeOut[1] =_heightOut;
}

void CL_FCT_SUB_ARGS::AddRecord()
{

}

void CL_FCT_SUB_ARGS::Clear()
{
	for (unsigned int i=0; i< CLTimes.size(); i++)
		CLTimes[i]->Clear();

	StartTime.Clear();
	StopTime.Clear();
	TotalTime.Clear();
	MaxTime.Clear();
	MinTime.Clear();
	MinTime.Time.tv_sec=10000;

	NbrOfTime = 0;
	Type = "";
	//TimeID=-1;
}

//!<Clear the maximum value of this record, to remove image loading...
void CL_FCT_SUB_ARGS::CleanMaxTime()
{
	//printf("\nCL_FCT_SUB_ARGS::CleanMaxTime()");
	int MaxTimeID=0;
	if (CLTimes.size()<1) return;

	//get the ID of the MaxTime :
	for (unsigned int i=0; i< CLTimes.size(); i++)
		if ((CLTimes[i]->Time.tv_usec == MaxTime.Time.tv_usec)
			&& (CLTimes[i]->Time.tv_sec == MaxTime.Time.tv_sec))
			MaxTimeID = i;

	//clear MaxTime
	CLTimes[MaxTimeID]->Clear();
	MaxTime.Clear();
	TotalTime.Clear();
	//get the new max value and total time:
	for (unsigned int i=0; i< CLTimes.size(); i++)
	{
		if ((CL_TimerVal)(*CLTimes[i]) > MaxTime) MaxTime = *CLTimes[i];
		TotalTime+=*CLTimes[i];
	}
	NbrOfTime--;
}

CL_FCT_SUB_ARGS::~CL_FCT_SUB_ARGS()
{
	vector<CL_TimerVal * >::iterator it = CLTimes.begin();
	for (unsigned int i=0; i<this->CLTimes.size(); i++)
	{ it++;

	delete(CLTimes[i]);//(*it)->~CL_TimerVal();
	CLTimes.erase(it);
	}
}

CL_TimerVal *CL_FCT_SUB_ARGS:: GetLastTime()
{
	return	CLTimes[CLTimes.size()-1];
}

bool operator==(const CL_FCT_SUB_ARGS& x, const CL_FCT_SUB_ARGS& y)
{
	//GPUCV_NOTICE("\noperator==");
	return (x.Params == y.Params) && (x.Type == y.Type)\
		&& (x.SizeIn[0] == y.SizeIn[0])\
		&& (x.SizeIn[1] == y.SizeIn[1])\
		&& (x.SizeOut[0] == y.SizeOut[0])\
		&& (x.SizeOut[1] == y.SizeOut[1]);
}

bool operator<(const CL_FCT_SUB_ARGS& x, const CL_FCT_SUB_ARGS& y)
{
	//GPUCV_NOTICE("\noperator<");
	if (strcmp (x.Params.data(), y.Params.data())==-1)
		return true;
	else if (strcmp (x.Params.data(), y.Params.data())==1)
		return false;
	else if (x.SizeIn[0]*x.SizeIn[1] > y.SizeIn[0]*y.SizeIn[1]) return true;
	else if (x.SizeIn[0]*x.SizeIn[1] < y.SizeIn[0]*y.SizeIn[1]) return false;
	else if (x.SizeOut[0]*x.SizeOut[1] > y.SizeOut[0]*y.SizeOut[1]) return true;
	else if (x.SizeOut[0]*x.SizeOut[1] < y.SizeOut[0]*y.SizeOut[1]) return false;
	return false;
}

bool operator>(const CL_FCT_SUB_ARGS& x, const CL_FCT_SUB_ARGS& y)
{
	//GPUCV_NOTICE("\noperator>(CL_FCT_SUB_ARGS)");
	if (strcmp (x.Params.data(), y.Params.data())==1)
		return true;
	else if (strcmp (x.Params.data(), y.Params.data())==-1)
		return false;
	else if (x.SizeIn[0]*x.SizeIn[1] < y.SizeIn[0]*y.SizeIn[1]) return true;
	else if (x.SizeIn[0]*x.SizeIn[1] > y.SizeIn[0]*y.SizeIn[1]) return false;
	else if (x.SizeOut[0]*x.SizeOut[1] < y.SizeOut[0]*y.SizeOut[1]) return true;
	else if (x.SizeOut[0]*x.SizeOut[1] > y.SizeOut[0]*y.SizeOut[1]) return false;


	return false;
}


bool operator>(const FCT_TRACER& x, const FCT_TRACER& y)
{
	size_t Size = strlen(x.FCT_NAME.data())+1;
	char * buffx = new char [Size];
	char * buffy = new char [Size];
	strcpy(buffx,x.FCT_NAME.data());
	strcpy(buffy,y.FCT_NAME.data());
	size_t maxLen = (strlen(x.FCT_NAME.data()) < strlen(y.FCT_NAME.data()))? strlen(x.FCT_NAME.data()):strlen(y.FCT_NAME.data());
	bool result = false;
	for (int i =0; i< (int)maxLen; i++)
	{
		if (buffx[i]!=buffy[i])
		{
			if (buffx[i]>buffy[i])
			{
				result= true;
				break;
			}
			else if (buffx[i]<buffy[i])
			{
				result= false;
				break;
			}
		}
	}
	/*	if (strcmp (x.FCT_NAME.data(), y.FCT_NAME.data())>0)
	return true;
	else if (strcmp (x.FCT_NAME.data(), y.FCT_NAME.data())<0)
	return false;
	return false;
	*/
	delete  [] buffx;
	delete  [] buffy;

	return result;
}

bool operator<(const FCT_TRACER& x, const FCT_TRACER& y)
{
	if (strcmp (x.FCT_NAME.data(), y.FCT_NAME.data())<0)
		return true;
	else if (strcmp (x.FCT_NAME.data(), y.FCT_NAME.data())>0)
		return false;
	return false;
}

bool operator==(const FCT_TRACER& x, const FCT_TRACER& y)
{
	if (x.FCT_NAME==y.FCT_NAME)
		return true;

	return false;
}



void FCT_TRACER::Sort()
{
	// printf("\nFCT_TRACER::Sort");
	bool Change = true;
	if (Records.empty()) return;
	while (Change)
	{
		Change = false;
		for (unsigned int i=0; i< Records.size()-1; i++)
			if (*Records[i]>*Records[i+1])
			{
				swap(Records[i+1], Records[i]);
				//printf("\nSwap %s and %s",Records[i]->Params.data(),Records[i]->Params.data());
				Change = true;
			}
	}
}


FCT_TRACER::FCT_TRACER(string _FctName)//, bool _CVG, char *_Params, GLuint _widthIn=0, GLuint _heightIn=0, GLuint _widthOut=0, GLuint _heightIn=0)
{
	FCT_NAME = _FctName;
	Clear();
}

void FCT_TRACER::AddRecord(string _type, const char * _Params, unsigned int _widthIn/*=0*/, unsigned int _heightIn/*=0*/, unsigned int _widthOut/*=0*/, unsigned int _heightOut/*=0*/)
{
	//static bool FirstTime=true;

	CL_FCT_SUB_ARGS * ActRec = NULL;

	// if (FirstTime==false)
	//look into the existing records to find the right parameters
	for (unsigned int i=0; i< Records.size(); i++)
		if (Records[i]->Type==_type)
			if (Records[i]->SizeIn[0]==_widthIn)
				if (Records[i]->SizeIn[1]==_heightIn)
					if (Records[i]->SizeOut[0]==_widthOut)
						if (Records[i]->SizeOut[1]==_heightOut)
							if (Records[i]->Params == _Params)
							{//all parameters are corrects
								ActRec = Records[i];
								ActRecID = i;
								break;
					  }
							//----------------------


							//add Fct is required
							if (ActRec == NULL)
							{
								if (TRC_MAX_NBR_RECORDS > Records.size())
								{
									Records.push_back(new CL_FCT_SUB_ARGS(_type,_Params, _widthIn, _heightIn, _widthOut, _heightIn));
									ActRecID = (int)Records.size()-1;
									ActRec = Records[ActRecID];
								}
								else
								{
									std::cout << "Notice : ToolsTracer::FCT_TRACER=> Maximum function number reached, stopping function tracing to preserve memory\n";
									ActRecID = -1;
								}
							}


							//Add Record
							if (ActRec != NULL) ActRec->AddRecord();//_type,_Params, _widthIn,_heightIn, _widthOut, _heightIn);

}


void FCT_TRACER::StartTrc()
{
	if(ActRecID!=-1)
	{
		if(AppliTracer()->GetConsoleStatus())
			printf("\n%s         ->%s:%s", IndentString, Records[ActRecID]->Type.data(), FCT_NAME.data());

		if (strlen(IndentString) <MaxIndentSize-2)
			strcat(IndentString, "  ");
		Records[ActRecID]->StartTime.GetTime();
	}
}
void FCT_TRACER::StopTrc()
{
	//static bool FirstTime=true;

	/*	if (FirstTime)
	{//we don't store the first time...cause we load image...
	FirstTime=false;
	return;
	}
	*/
	if(ActRecID==-1) return;
	Records[ActRecID]->StopTime.GetTime();//???

	if (TRC_MAX_NBR_TIMES > Records[ActRecID]->CLTimes.size())
	{


		Records[ActRecID]->CLTimes.push_back(new CL_TimerVal());
		CL_TimerVal * CurTime=Records[ActRecID]->CLTimes[Records[ActRecID]->CLTimes.size()-1];

		//Records[ActRecID]->StopTime.GetTime();//???
		*CurTime = Records[ActRecID]->StopTime;
		*CurTime -= Records[ActRecID]->StartTime;
		Records[ActRecID]->TotalTime+=*CurTime;

		//local min and max depending on params
		if (*CurTime < Records[ActRecID]->MinTime)
			Records[ActRecID]->MinTime = *CurTime;

		if (*CurTime > Records[ActRecID]->MaxTime)
		{
			Records[ActRecID]->MaxTime = *CurTime;

		}
		//=========================================

		//Min and max of the function, not depending on params
		if (*CurTime < MinTime)
			MinTime = *CurTime;

		if (*CurTime > MaxTime)
		{
			MaxTime = *CurTime;
			//AppliTracer()->SetMax(CurTime);
		}
		//============================================
		if (strlen(IndentString) >2)
			IndentString[strlen(IndentString)-2]='\0';
		if(AppliTracer()->GetConsoleStatus())
		{
			if (Records[ActRecID]->Params.size()!=0)
			{
				if (Records[ActRecID]->SizeIn[0]!=0)
					printf("\n%d'%6.d:%s<-%s:%s (%s,%d,%d)", (int)CurTime->Time.tv_sec, (int)CurTime->Time.tv_usec, GetIndent(), Records[ActRecID]->Type.data(), FCT_NAME.data(), Records[ActRecID]->Params.data(), Records[ActRecID]->SizeIn[0], Records[ActRecID]->SizeIn[1]);
				else
					printf("\n%d'%6.d:%s<-%s:%s (%s)", (int)CurTime->Time.tv_sec, (int)CurTime->Time.tv_usec, GetIndent(), Records[ActRecID]->Type.data(), FCT_NAME.data(), Records[ActRecID]->Params.data());
			}
			else
			{
				if (Records[ActRecID]->SizeIn[0]!=0)
					printf("\n%d'%6.d:%s<-%s:%s (%d,%d)", (int) CurTime->Time.tv_sec, (int) CurTime->Time.tv_usec, GetIndent(), Records[ActRecID]->Type.data(), FCT_NAME.data(), Records[ActRecID]->SizeIn[0], Records[ActRecID]->SizeIn[1]);
				else
					printf("\n%d'%6.d:%s<-%s:%s", (int) CurTime->Time.tv_sec, (int) CurTime->Time.tv_usec, GetIndent(), Records[ActRecID]->Type.data(), FCT_NAME.data());
			}
		}
		Records[ActRecID]->NbrOfTime++;
	}
	else
	{
		//if(AppliTracer()->GetConsoleStatus())
		//GPUCV_NOTICE("\nNotice : ToolsTracer::FCT_TRACER=> Maximum times record number reached, stopping function tracing to preserve memory");
	}
}

void FCT_TRACER::Clear()
{
	for (unsigned int i=0; i< Records.size(); i++)
		Records[i]->Clear();

	TimeID=-1;
	TimeNbr=0;
	MaxTime.Clear();
	MinTime.Clear();

}

int FCT_TRACER::getNbrOfParams()
{
	string Actparams="";
	//int ParamsNbr=0;
	vector<string*> AllParams;
	bool found = false;
	for (unsigned int i=0; i< Records.size(); i++)
	{
		found = false;
		for( unsigned int j=1; j < AllParams.size(); j++)
		{
			//printf("\nFCT_TRACER::getNbrOfParams compare %s and %s", Records[i]->Params.data(),AllParams[j]->data());
			if( Records[i]->Params == *AllParams[j])
			{
				found = true;
			}
		}
		if (found==false) 	AllParams.push_back(&Records[i]->Params);
	}
	//printf("\nFCT_TRACER::getNbrOfParams %d", AllParams.size()-1);
	return (int)AllParams.size()-1;
}


CL_TimerVal *FCT_TRACER:: GetLastTime()
{
	return	Records[ActRecID]->GetLastTime();
}

FCT_TRACER::~FCT_TRACER(void)
{
	Records.clear();
}


void FCT_TRACER::CleanMaxTime()
{
	//	printf("\nFCT_TRACER::CleanMaxTime()");
	// 	int MaxTimeID;
	//get the ID of the MaxTime :
	for (unsigned int i=0; i< Records.size(); i++)
		Records[i]->CleanMaxTime();

	MaxTime.Clear();
	//get the new max value and total time:
	for (unsigned int i=0; i< Records.size(); i++)
	{
		if (Records[i]->MaxTime > MaxTime) MaxTime = Records[i]->MaxTime;
	}

}



char* FCT_TRACER::GetIndent()
{
	return IndentString;
}


CL_APPLI_TRACER * AppliTracer()//int _ModeNbr, int _FctNbr)
{
	static CL_APPLI_TRACER *AppTrc = new CL_APPLI_TRACER();//_ModeNbr,  _FctNbr) ;
	return AppTrc;
}

void CL_APPLI_TRACER::SetMax(CL_TimerVal * _Max)
{
	if( Max < *_Max) Max= *_Max;
}


CL_APPLI_TRACER::CL_APPLI_TRACER()
{
	Console=false;
}


FCT_TRACER* CL_APPLI_TRACER::AddRecord(string _FctName, string _type, float _Params, unsigned int _widthIn/*=0*/, unsigned int _heightIn/*=0*/, unsigned int _widthOut/*=0*/, unsigned int _heightOut/*=0*/)
{
	char Temp[256];
	sprintf(Temp,"%f", _Params);
	return AddRecord(_FctName, _type, Temp,  _widthIn, _heightIn,_widthOut,_heightOut);
}

FCT_TRACER* CL_APPLI_TRACER::AddRecord(string _FctName, string _type, std::string _Params, unsigned int _widthIn/*=0*/, unsigned int _heightIn/*=0*/, unsigned int _widthOut/*=0*/, unsigned int _heightOut/*=0*/)
{
	FCT_TRACER * ActRec = NULL;


	//look for the function
	for (unsigned int i=0; i< FctTrc.size(); i++)
		if (FctTrc[i]->FCT_NAME==_FctName)
		{
			ActRec = FctTrc[i];
			break;
		}
		//----------------------

		//add Fct is required
		if (ActRec == NULL)
		{
			if (TRC_MAX_NBR_FCT > FctTrc.size() )
			{
				FctTrc.push_back(new FCT_TRACER(_FctName));
				ActRec = FctTrc[FctTrc.size()-1];
		 }
			else
		 {
			 std::cout <<"\nNotice : ToolsTracer::CL_APPLI_TRACER=> Maximum function number reached, stopping function tracing to preserve memory\n";
		 }
		}

		//Add Record
		if (ActRec != NULL) ActRec->AddRecord(_type, _Params.data(), _widthIn,_heightIn, _widthOut, _heightIn);

		return ActRec;
}

CL_APPLI_TRACER::~CL_APPLI_TRACER()
{

}

void CL_APPLI_TRACER::EnableConsole()
{
	Console =true;
}
void CL_APPLI_TRACER::DisableConsole()
{
	Console =false;
}

bool CL_APPLI_TRACER::GetConsoleStatus()
{
	return Console;
}

void CL_APPLI_TRACER::Sort()
{
	// printf("\nCL_APPLI_TRACER::Sort");
	if (FctTrc.empty()) return;
	bool Change = true;
	while (Change)
	{
		Change = false;
		for (unsigned int i=0; i< FctTrc.size()-1; i++)
			if (FctTrc[i]>FctTrc[i+1])
			{
				swap(FctTrc[i+1], FctTrc[i]);
				printf("\nSwap %s and %s",FctTrc[i]->FCT_NAME.data(),FctTrc[i+1]->FCT_NAME.data());
				Change = true;
			}
	}
}



void CL_APPLI_TRACER::OpenFile(char * filname)
{//open file
	//	int i;
	File = fopen(filname, "w");
}


void CL_APPLI_TRACER::CloseFile()
{
	fclose(File );//CloseHandle(AppTracer.File);
}

void CL_APPLI_TRACER::CleanMaxTime()
{
	// 	int MaxTimeID;
	//get the ID of the MaxTime :
	for (unsigned int i=0; i< FctTrc.size(); i++)
		FctTrc[i]->CleanMaxTime();

	Max.Clear();
	//get the new max value  and total time:
	for (unsigned int i=0; i< FctTrc.size(); i++)
	{
		if (FctTrc[i]->MaxTime > Max) Max = FctTrc[i]->MaxTime;
	}
}




void CL_APPLI_TRACER::AddFctToFile(FCT_TRACER * CurFct, int MaxValues)
{
	if (CurFct)
	{
		char buff[512];

		sprintf(buff,"\n%s", CurFct->FCT_NAME.data());
		fwrite( buff, strlen(buff)*sizeof(unsigned char),1,File);

		CL_FCT_SUB_ARGS * ActiveRec=NULL;
		CL_TimerVal * CurTime =NULL;
		for (unsigned int i=0; i<CurFct->Records.size(); i++)
		{
			ActiveRec = CurFct->Records[i];
			//sprintf_s(buff,256, "\n%s\t(%d,%d,%s)", ActiveRec->Type.data(), ActiveRec->SizeIn[0],ActiveRec->SizeIn[1], ActiveRec->Params.data());
			sprintf(buff,"\n%s\t(%d,%d,%s)", ActiveRec->Type.data(), ActiveRec->SizeIn[0],ActiveRec->SizeIn[1], ActiveRec->Params.data());
			fwrite( buff, strlen(buff)*sizeof(unsigned char),1,File);

			for (unsigned int j =0; j< ActiveRec->CLTimes.size() ; j++)
			{
				CurTime = ActiveRec->CLTimes[j];
				//sprintf_s(buff,256, "\t%d", CurTime->Time.tv_sec*1000000 + CurTime->Time.tv_usec);//, CurTime->NbrOfTime);
				sprintf(buff,"\t%d", int(CurTime->Time.tv_sec*1000000 + CurTime->Time.tv_usec));//, CurTime->NbrOfTime);
				fwrite( buff, strlen(buff)*sizeof(unsigned char),1,File);
			}
		}
	}
	else
	{
		printf("\nFonction not found...");
	}

}

int CL_APPLI_TRACER::GenerateTextFile(char * Filename)
{
	printf("\nGenerating TEXT file...");
	OpenFile(Filename);
	if (!File) return -1;

	//	this->Sort();
	//sort(FctTrc.begin(), FctTrc.end());
	//scan all functions and print them to file
	for (unsigned int i = 0; i<this->FctTrc.size(); i++)
		AddFctToFile(FctTrc[i], 1000);

	//start writing end of html file:
	CloseFile();

	printf("Done!");
	if (File) return -1;
	return 1;
}



void RetrieveBenchData(std::vector<STBenchValue*> & dataVector, FCT_TRACER * CurFct, size_t start, size_t end)
{
	CL_FCT_SUB_ARGS * ActiveRec=NULL;

	std::string Params="";
	STBenchValue* CurrBenchValue;
	for (size_t i=start; i < end; i++)
	{
		if(i>=CurFct->Records.size())
			break;
		ActiveRec = CurFct->Records[i];
		if(!ActiveRec)
			continue;

		ActiveRec->Params;
		CurrBenchValue = new STBenchValue();
		dataVector.push_back(CurrBenchValue);
		CurrBenchValue->nbrOfTime = ActiveRec->NbrOfTime;
		CurrBenchValue->min   = ActiveRec->MinTime.GetTimeInUsec();
		CurrBenchValue->max   = ActiveRec->MaxTime.GetTimeInUsec();
		CurrBenchValue->total = ActiveRec->TotalTime.GetTimeInUsec();
		CurrBenchValue->width = ActiveRec->SizeIn[0];
		CurrBenchValue->height = ActiveRec->SizeIn[1];
		CurrBenchValue->type = ActiveRec->Type;
		if (ActiveRec->NbrOfTime)
		{
			CurrBenchValue->avg = CurrBenchValue->total;
			CurrBenchValue->avg/= float(CurrBenchValue->nbrOfTime);
		}
		else
			CurrBenchValue->avg=0;
	}
}

#define ExportHtmlColumnValues(STRING, DATA_VECTOR, VALUE)\
{\
	STRING="";\
	char TempBuff[512];\
	STBenchValue* CurrBenchValue;\
	std::string Color2;\
	for (unsigned int j=0; j< DATA_VECTOR.size(); j+=1)\
{\
	CurrBenchValue = DATA_VECTOR[j];\
	if (CurrBenchValue->type==GPUCV_IMPL_GLSL_STR)\
	Color2 = BENCH_COLOR_HTML_GPUCV;\
		else if(CurrBenchValue->type==_BENCH_OPENCV_TYPE)\
		Color2 = BENCH_COLOR_HTML_OPENCV;\
		else if(CurrBenchValue->type==_BENCH_CUDA_TYPE)\
		Color2 = "BENCH_COLOR_HTML_CUDA";\
		else if(CurrBenchValue->type=="GL")\
		Color2 = "00FFFF";\
		else\
		Color2 = "555555";\
		sprintf(TempBuff,"<span style='color:%s'>%d</span>", Color2.data(),(int)(CurrBenchValue->VALUE));\
		STRING += TempBuff;\
		STRING += ((j!=DATA_VECTOR.size()-1) && (DATA_VECTOR.size()>1))?"<br>":"";\
}\
}


#define ExportHtmlRowValuesAll(STRING, DATA_VECTOR, VALUE)\
{\
	STRING="";\
	char TempBuff[512];\
	STBenchValue* CurrBenchValue;\
	std::string Color2;\
	std::string Buff[4];\
	std::string * CurrentBuff = Buff;\
	sprintf(TempBuff, "Average processing time in �s for %d iterations", DATA_VECTOR[0]->nbrOfTime);\
	STRING +=TempBuff;\
	STRING += "<table border=1 align='center'><tr>";\
	STRING +="<td width='20%'>Image Size</td>";\
	STRING +="<td width='20%'>";\
	STRING +=_BENCH_OPENCV_TYPE;\
	STRING += "</td>";\
	STRING +="<td width='20%'>";\
	STRING +=_BENCH_IPP_TYPE;\
	STRING += "</td>";\
	STRING +="<td width='20%'>";\
	STRING +=GPUCV_IMPL_GLSL_STR;\
	STRING += "</td>";\
	STRING +="<td width='20%'>";\
	STRING +=_BENCH_CUDA_TYPE;\
	STRING += "</td>";\
	long int PreviousSize = 0;\
	for (unsigned int j=0; j< DATA_VECTOR.size(); j+=1)\
{\
	CurrBenchValue = DATA_VECTOR[j];\
	if(PreviousSize==0 || PreviousSize != CurrBenchValue->width*CurrBenchValue->height)\
{\
	sprintf(TempBuff, "</tr><tr><td>%dx%d</td>", CurrBenchValue->width, CurrBenchValue->height);\
	STRING += Buff[0]+Buff[1]+Buff[2]+Buff[3]+TempBuff;\
	Buff[0]=Buff[1]=Buff[2]=Buff[3] = "<td>.</td>";\
	PreviousSize = CurrBenchValue->width*CurrBenchValue->height;\
}\
	\
	if(CurrBenchValue->type==_BENCH_OPENCV_TYPE)\
{\
	Color2 = BENCH_COLOR_HTML_OPENCV;\
	CurrentBuff = &Buff[0];\
}\
		else if (CurrBenchValue->type==_BENCH_IPP_TYPE)\
{\
	Color2 = BENCH_COLOR_HTML_IPP;\
	CurrentBuff = &Buff[1];\
}\
		else if (CurrBenchValue->type==GPUCV_IMPL_GLSL_STR)\
{\
	Color2 = BENCH_COLOR_HTML_GPUCV;\
	CurrentBuff = &Buff[2];\
}\
		else if (CurrBenchValue->type==_BENCH_CUDA_TYPE)\
{\
	Color2 = BENCH_COLOR_HTML_CUDA;\
	CurrentBuff = &Buff[3];\
}\
		else if(CurrBenchValue->type=="GL")\
		Color2 = "00FFFF";\
		else\
		Color2 = "555555";\
		sprintf(TempBuff,"<td><span style='color:%s'>%d</span></td>", Color2.data(),(int)(CurrBenchValue->VALUE));\
		*CurrentBuff = TempBuff;\
}\
	STRING += Buff[0]+Buff[1]+Buff[2]+Buff[3];\
	STRING +="</tr></table>";\
}

#define TRACER_BENCH_ROW_HTML 1

void CL_APPLI_TRACER::AddFctToHtmlFile(FCT_TRACER * CurFct, long int MaxValue, std::string SvgPath)
{
	static bool paire=true;
	if (CurFct)
	{
		//start Line
		string Html;
		char TempBuff[512];
		string Color, Color2;

		if (paire)
			Color="EEEECC";
		else
			Color="CCEEEE";

		CurFct->Sort();
		sprintf(TempBuff,"<tr bgcolor='#%s'><td rowspan=%d><a name='%s'>%s</a><br><a href='#Top'> Top</a></td>", Color.data() , CurFct->getNbrOfParams(),CurFct->FCT_NAME.data(), CurFct->FCT_NAME.data());
		fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);

		CL_FCT_SUB_ARGS * ActiveRec=NULL;
		//CL_TimerVal * CurTime =NULL;
		CL_TimerVal Average;
		//	int AverageInt;
		int NParamsEqual=1;
		bool NbParamsFirst=true;
		//	int Max, Min;
		string Params="", ImgFile="";
		std::vector<STBenchValue*> DataVector;

		for (unsigned int i=0; i<CurFct->Records.size(); i+=NParamsEqual)
		{
			ActiveRec = CurFct->Records[i];
			Params = ActiveRec->Params;
			if (NParamsEqual==1)
			{
				for (unsigned int j=i+1; j<CurFct->Records.size(); j++)
					if (ActiveRec->Params == CurFct->Records[j]->Params)NParamsEqual++;
				NbParamsFirst=true;
			}

			//clear existing data
			if(DataVector.size()>0)
			{
				for (size_t x=0; x < DataVector.size(); x++)
					delete DataVector[x];
				DataVector.clear();
			}
			//retriev new one
			RetrieveBenchData(DataVector, CurFct, i,i+NParamsEqual);


			if (i!=0)
				Html="<tr bgcolor='#"+Color+"'>";

			//params
			if (NbParamsFirst==true)
			{
				Html+="<td rowspan=1 align=center>";
				//Html+=(ActiveRec->Params!="")?ActiveRec->Params + "</td>":"</td>";
				Html+= ActiveRec->Params;
				fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
			}

			Html="<td valign=center>";
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

#if _GPUCV_SVG_CONVERT_TO_PNG
			ImgFile= _GPUCV_SVG_EXPORT_PATH;
			//ImgFile+= "/";
			ImgFile+= CurFct->FCT_NAME;
			ImgFile+= "_";
			ImgFile+= ActiveRec->Params;
			ImgFile+= ".png";
			//		Html = "<a href='"+ImgFile+"'><img src='"+ImgFile+"'>";
			Html = "<img src='"+ImgFile+"'>";
			Html +="</td><td>";
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
#endif

#if !TRACER_BENCH_ROW_HTML
			//image size
			char LastImgSize[32]="";
			STBenchValue* CurrBenchValue;
			for (unsigned int j=0; j< DataVector.size(); j+=1)
			{
				CurrBenchValue = DataVector[j];
				if (CurrBenchValue->nbrOfTime)
				{
					sprintf(TempBuff, "%dx%d", CurrBenchValue->width, CurrBenchValue->height);
					if (strcmp(LastImgSize, TempBuff)==0)
					{//we don't write image size again

					}
					else
					{
						fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
						if (j < i+NParamsEqual-2)
						{
							Html = ((j!=i+NParamsEqual) && (NParamsEqual>1))?"<br><br>":"";
							Html = ((Html=="<br><br>") && (CurrBenchValue->type!=_BENCH_OPENCV_TYPE) && (CurrBenchValue->type!=GPUCV_IMPL_GLSL_STR))?"<br>":Html;
							fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
						}
					}
					//strcpy_s(LastImgSize, 256,TempBuff);
					strcpy(LastImgSize,TempBuff);
				}
			}

			Html = "</td><td>";
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
#endif
#if !_GPUCV_SVG_CONVERT_TO_PNG
			//Values
			for (unsigned int j=i; j< i+NParamsEqual; j++)
			{
				ActiveRec = CurFct->Records[j];
				if (ActiveRec->NbrOfTime)
				{
					//statistics values
					Max = (ActiveRec->MaxTime.Time.tv_sec*1000000 +  ActiveRec->MaxTime.Time.tv_usec);
					Min = (ActiveRec->MinTime.Time.tv_sec*1000000 +  ActiveRec->MinTime.Time.tv_usec);

					if (MaxValue==0) MaxValue=10000;
					int MinPorcent = (Min * 100) /MaxValue;
					int MaxPorcent = ((Max - Min) * 100) /MaxValue;

					//average time :
					/*			Average = ActiveRec->TotalTime;
					Average/= ActiveRec->NbrOfTime;

					AverageInt = (Average.Time.tv_sec*1000000 +  Average.Time.tv_usec);
					//AverageInt = AverageInt;
					*/

					//show min value
					if (ActiveRec->Type==GPUCV_IMPL_GLSL_STR)
						Color2 = "009900";//rgb(255, 51, 255);";//"FFCC33";
					else if(ActiveRec->Type==_BENCH_OPENCV_TYPE)
						Color2 = "FF0000";//rgb(51, 102, 255);";//"3366FF";
					/*
					if (ActiveRec->Type==_BENCH_OPENCV_TYPE)
					Color2 = "FFCC33";//rgb(255, 51, 255);";//"FFCC33";
					else if(ActiveRec->Type==GPUCV_IMPL_GLSL_STR)
					Color2 = "0000FF";//rgb(51, 102, 255);";//"3366FF";
					*/
					else if(ActiveRec->Type=="GL")
						Color2 = "00FFFF";
					else
						Color2 = "555555";

					//Html = (j!=i)?"<br>":"";
					Html = "<table width='100%' border=0 cellspacing=0 cellpadding=0><tr height=15><td style='background-color:"+Color2+"' width='";
					fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);


					//	if(Min/MaxValue*100 < 1)

					if(MinPorcent <1 )
					{	//sprintf_s(TempBuff,256, "1" );
						sprintf(TempBuff,"1");
						Html=TempBuff;
					}
					else
					{	//sprintf_s(TempBuff,256, "%d",MaxPorcent);//(int)(((Max-Min)/MaxValue)*100));
						sprintf(TempBuff,"%d",MinPorcent);//(int)(((Max-Min)/MaxValue)*100));
						Html = TempBuff;
						Html += "%";
					}
					//fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);

					Html += "' align=center label='Minimum value:";
					Html += TempBuff;
					Html += "'></td>";
					fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);


					//show max value
					if (ActiveRec->Type==_BENCH_OPENCV_TYPE)
						Color2 = "FFFF66";//rgb(51, 204, 255);";//"FFDDD";
					else if(ActiveRec->Type==GPUCV_IMPL_GLSL_STR)
						Color2 = "33CCFF";//rgb(102, 255, 153);";//"FFDDFF";
					else if(ActiveRec->Type=="GL")
						Color2 = "55FFFF";
					else
						Color2 = "CCCCCC";

					/*
					if (ActiveRec->Type==_BENCH_OPENCV_TYPE)
					Color2 = "FF0000";//rgb(51, 204, 255);";//"FFDDD";
					else if(ActiveRec->Type==GPUCV_IMPL_GLSL_STR)
					Color2 = "009900";//rgb(102, 255, 153);";//"FFDDFF";
					else if(ActiveRec->Type=="GL")
					*/
					Html = "<td style='background-color:"+Color2+"' width='";
					fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

					//if(((Max-Min)/MaxValue)*100 < 1)
					if(MaxPorcent <1 )
					{	//sprintf_s(TempBuff,256, "1" );
						sprintf(TempBuff,"1");
						Html=TempBuff;
					}
					else
					{	//sprintf_s(TempBuff,256, "%d",MaxPorcent);//(int)(((Max-Min)/MaxValue)*100));
						sprintf(TempBuff,"%d",MaxPorcent);//(int)(((Max-Min)/MaxValue)*100));
						Html = TempBuff;
						Html += "%";
					}
					//fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
					Html += "' align=center label='Maximum value:";
					Html += TempBuff;
					Html += "'></td><td>&nbsp;</td></tr></table>";
					/*
					fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
					//sprintf_s(TempBuff,256, "<td>%d</td></tr></table>",AverageInt);
					sprintf(TempBuff,"</tr></table>",AverageInt);
					*/
					fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
				}
			}

			Html = "</td><td>";
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
#endif


#if TRACER_BENCH_ROW_HTML
			//All values
			ExportHtmlRowValuesAll(Html , DataVector, avg);
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
			//Html = "</td><td>";
			//fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
#else
			//Average values
			ExportHtmlColumnValues(Html , DataVector, avg);
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
			Html = "</td><td>";
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

			//Min values
			ExportHtmlColumnValues(Html, DataVector, min);
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
			Html = "</td><td>";
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);


			//Max values
			ExportHtmlColumnValues(Html, DataVector, max);
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

			Html = "</td><td>";
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

			//Nbr of time values
			ExportHtmlColumnValues(Html, DataVector, nbrOfTime);
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

			Html = "</td></tr>";
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
#endif
			//ask to export SVG file:
			if(SvgPath!="")
				AddFctToSvgFile((char*)SvgPath.data(), CurFct, ActiveRec,DataVector, 0, 0);
		}
	}
	paire = !paire;
}

int CL_APPLI_TRACER::GenerateHtmlFile2(char * Filename,
									   char * Title,
									   char * Start,
									   char * svgPath
									   )
{
	printf("\nGenerating HTML file...");
	OpenFile(Filename);
	if (!File) return -1;


	//start writing header of html file:
	std::string Html;
	//Title
	Html =  "<html><head><title>";
	Html +=  Title;
	Html += "</title>";
#if 0
	Html += "<link href='../css/cube.css' rel='stylesheet' type='text/css'>"
#endif
		Html += "</head><body>";
	Html += "<H1>";
	Html += Title;
	Html += "</H1>";
	Html += "<table width='100%'><tbody><tr><td>";
	//Start
	Html += Start;
	Html += "</td>";

	/*
	"<h1>GPUCV library benchmarks</h1> "\
	"Performance comparison between OpenCV operators and GPUCV operators for various image size and parameters."\
	"<br><h3>Benchmark computer description:</h3>"\
	"<ul>"\
	"<li>Graphic card:...</li>"\
	"<li>Processor: ...</li>"\
	"<li>Operating system : Windows XP SP2.</li>"\
	"<li>GPUCV library version: v0.0.1.</li>"\
	"<li>Date:" __DATE__ "</li></ul>"\
	"</td>";
	*/
	fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

	//list all functions and create links
	Html  ="</tr><tr><td><h3><a name='Top'>List of the functions:</a></h3><ul>";
	fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
	for (unsigned int i = 0; i<this->FctTrc.size(); i++)
	{
		//if(FctTrc[i]->TimeNbr!=0)
		{
			Html = "<li><a href='#"+FctTrc[i]->FCT_NAME +"'>"+FctTrc[i]->FCT_NAME +"</a></li>";
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
		}
	}
	Html ="</ul><br></td></tr></table>";
	fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

	Html  = "<table width='100%' border=1>";
	Html +="<tr><td width='5%'>Functions</td>";
	Html +="<td width='5%'>Parameters</td>";
#if _GPUCV_SVG_CONVERT_TO_PNG
	Html +="<td>Graphs</td>";
#endif

#if !_GPUCV_SVG_CONVERT_TO_PNG
	Html +="<td width='70%'>Execution time in �s<br>"\
		"<table border='0'><tbody>"\
		"<tr><td>OpenCV minimum:</td><td style='width: 20px; background-color: FFCC33;'><br></td>"\
		"<td>maximum:</td><td style='width: 20px; background-color: FFFF66;'><br></td>"\
		"<tr><td>GPUCV minimum:</td><td style='width: 10px; background-color: 0000FF;'><br></td>"\
		"<td>maximum:</td><td style='width: 20px; background-color: 33CCFF;'><br></td>"\
		"</tr></tbody></table></td>";
#endif

	Html +="<td width='20%'>Results</td>";

	fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

	//int MaxValue = Max.GetTimeInUsec();
	//scan all functions and print them to html file
	for (unsigned int i = 0; i<this->FctTrc.size(); i++)
	{
		//if(FctTrc[i]->TimeNbr!=0)
		AddFctToHtmlFile(FctTrc[i], FctTrc[i]->MaxTime.GetTimeInUsec(),svgPath);//MaxValue);
	}

	//start writing end of html file:

	Html ="</table><table><tr><td width=100><div style='clear: both;'><span style='float: left;'>";
	Html +=BENCH_COPYRIGHT_STRING_HTML;
	Html +=".</span></div>";
	Html +="</tr></table>"\
		"</body></html>";
	fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

	CloseFile();
	printf("Done!");
	if (File)
		return -1;
	else
		return 1;
}

int CL_APPLI_TRACER::GenerateHtmlFile(char * Filename)
{
	printf("\nGenerating HTML file...");
	OpenFile(Filename);
	if (!File) return -1;

	//this->Sort();
	//sort(FctTrc.begin(), FctTrc.end());
	//start writing header of html file:
	string Html;
	Html =  "<html><head><title>GPUCV library benchmarks</title>"\
		"<link href='../css/cube.css' rel='stylesheet' type='text/css'>"\
		"</head><body><table width='100%'><tbody><tr><td>"\
		"<h1>GPUCV library benchmarks</h1> "\
		"Performance comparison between OpenCV operators and GPUCV operators for various image size and parameters."\
		"<br><h3>Benchmark computer description:</h3>"\
		"<ul>"\
		"<li>Graphic card:...</li>"\
		"<li>Processor: ...</li>"\
		"<li>Operating system : Windows XP SP2.</li>"\
		"<li>GPUCV library version: v0.0.1.</li>"\
		"<li>Date:" __DATE__ "</li></ul>"\
		"</td>";
	fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

	//list all functions and create links
	Html  ="</tr><tr><td><h3><a name='Top'>List of the functions:</a></h3><ul>";
	fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
	for (unsigned int i = 0; i<this->FctTrc.size(); i++)
	{
		//if(FctTrc[i]->TimeNbr!=0)
		{
			Html = "<li><a href='#"+FctTrc[i]->FCT_NAME +"'>"+FctTrc[i]->FCT_NAME +"</a></li>";
			fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);
		}
	}
	Html ="</ul><br></td></tr></table>";
	fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

	Html  = "<table width='100%' border=1>";
	Html +="<tr><td width='5%'>Functions</td>";
	Html +="<td width='5%'>Parameters</td>";
#if _GPUCV_SVG_CONVERT_TO_PNG
	Html +="<td>Graphs</td>";
#endif
	Html +="<td width='5%'>Image Size</td>";
#if !_GPUCV_SVG_CONVERT_TO_PNG
	Html +="<td width='70%'>Execution time in �s<br>"\
		"<table border='0'><tbody>"\
		"<tr><td>OpenCV minimum:</td><td style='width: 20px; background-color: FFCC33;'><br></td>"\
		"<td>maximum:</td><td style='width: 20px; background-color: FFFF66;'><br></td>"\
		"<tr><td>GPUCV minimum:</td><td style='width: 10px; background-color: 0000FF;'><br></td>"\
		"<td>maximum:</td><td style='width: 20px; background-color: 33CCFF;'><br></td>"\
		"</tr></tbody></table></td>";
#endif
	Html +="<td width='5%'>OpenCV Avg (�s)</td>";
	Html +="<td width='5%'>IPP Avg (�s)</td>";
	Html +="<td width='5%'>GpuCV Avg (�s)</td>";
	Html +="<td width='5%'>CUDA Avg (�s)</td>";
	Html +="<td width='5%'>Records Number</td>";
	fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

	//int MaxValue = Max.GetTimeInUsec();
	//scan all functions and print them to html file
	for (unsigned int i = 0; i<this->FctTrc.size(); i++)
	{
		//if(FctTrc[i]->TimeNbr!=0)
		AddFctToHtmlFile(FctTrc[i], FctTrc[i]->MaxTime.GetTimeInUsec());//MaxValue);
	}

	//start writing end of html file:
	Html ="</table><table><tr><td width=100><div style='clear: both;'><span style='float: left;'>";
	Html +=BENCH_COPYRIGHT_STRING_HTML;
	Html +=".</span></div>"\
		"<td align='center'><script language='javascript'>compte='285308GPUCV' couleur_lib='rouge' </script>"\
		"<script language='javascript'src='http://lib3.libstat.com/private/stat.js'></script>"\
		"<a href='http://www.libstat.com' target='_Blank'>mesure d'audience</a></td></tr></table>"\
		"</body></html>";
	fwrite( Html.data(), strlen(Html.data())*sizeof(unsigned char),1,File);

	CloseFile();
	printf("Done!");
	if (File) return -1;

	return 1;
}

#define SVG_GENERATE_HEADER(STRING, WIDTH, HEIGHT)\
{\
	STRING="<svg width='550' height='425'>\n";\
}

#define SVG_GENERATE_X_AXE(STRING)\
{\
	STRING="<line x1='35' x2='480' y1='350' y2='350'  style='stroke-width:3 ; fill:none; stroke:black'/>\n";\
	STRING+="<line x1='470' x2='480' y1='347' y2='350'  style='stroke-width:3 ; fill:none; stroke:black'/>\n";\
	STRING+="<line x1='470' x2='480' y1='353' y2='350'  style='stroke-width:3 ; fill:none; stroke:black'/>\n";\
}

#define SVG_GENERATE_Y_AXE(STRING)\
{\
	char TempBuff[256];\
	STRING="<line x1='35' x2='32' y1='25' y2='35'  style='stroke-width:3 ; fill:none; stroke:black'/>\n";\
	STRING+="<line x1='35' x2='38' y1='25' y2='35'  style='stroke-width:3 ; fill:none; stroke:black'/>\n";\
	STRING+="<line x1='35' x2='35' y1='351' y2='25'  style='stroke-width:3 ; fill:none; stroke:black'/>\n";\
	for(unsigned int j=0;j<5;j++)\
{\
	sprintf(TempBuff, "<line x1='35' x2='38' y1='%d' y2='%d'  style='stroke-width:2 ; fill:none; stroke:black'/>\n",50+60*j,50+60*j);\
	STRING+=TempBuff;\
}\
}

#define DRAW_LINES(STRING, DATA_VECTOR,TYPE,COLOR,NAME)\
{\
	char TempBuff[512];\
	sprintf(TempBuff, "<g style='stroke:%s'>\n", COLOR);\
	STRING=TempBuff;\
	for (unsigned int j=0; j< DATA_VECTOR.size(); j+=1)\
{\
	CurrBenchValue = DATA_VECTOR[j];\
	if(!CurrBenchValue)\
	continue;\
	if (CurrBenchValue->type!=TYPE) \
	continue;\
	\
	ImgSize = CurrBenchValue->width * CurrBenchValue->height;\
	/*x axe*/\
	ImgSizePos = ((float)ImgSize)/ImgSizeMax * AxeLength + AxeStart[1];\
	Pt1[0] = ImgSizePos;\
	/*y axe*/\
	Pt1[1] = 350-300*((float)CurrBenchValue->avg/maxvalue);\
	if(!(Pt2[0] == Pt2[1] && Pt2[0] == 0))\
{\
	sprintf(TempBuff, "<polyline points='%d,%d,%d,%d' style='stroke-width:3 ; fill:none;'/>\n",Pt1[0]+5,Pt1[1],Pt2[0]+5,Pt2[1]);\
	STRING+=TempBuff;\
}\
			else\
{\
	sprintf(TempBuff, "<text x='450'  y='%d' style='font-family:Verdana; font-size:15;stroke:none;fill:%s'>%s</text>\n",Pt1[1],COLOR,NAME);\
	STRING+=TempBuff;\
}\
	/*draw current point*/\
	if(Pt1[0]>(50))\
{\
	sprintf(TempBuff, "<circle cx='%d'  cy='%d' r='2' style='stroke-width:2 ; fill:none;'/>\n",Pt1[0]+5,Pt1[1]);\
	STRING+=TempBuff;\
}\
	Pt2[0] =Pt1[0];\
	Pt2[1] =Pt1[1];\
}\
	STRING+="</g>\n";\
	Pt2[0] = Pt2[1] =0;\
}

int CL_APPLI_TRACER::AddFctToSvgFile(char * _path, FCT_TRACER * CurFct, CL_FCT_SUB_ARGS * ActiveRec, std::vector<STBenchValue*> & dataVector, int start, int end)
{

#if _GPUCV_SVG_CREATE_FILES
	string StrOpencvColor="";
	if (CurFct)
	{
		string Svg;
		string Filename;
		char TempBuff[256];
		CurFct->Sort();
		unsigned long int maxvalue;
		unsigned int NParamsEqual=1;
		bool NbParamsFirst=true;

		string ActiveParams="";
		string Params="";

		STBenchValue* CurrBenchValue=NULL;

		//prepare to write in the export_path
		Filename = _path;
		//Filename += "\\"+CurFct->FCT_NAME+"_"+ActiveRec->Params+".svg";
		Filename += CurFct->FCT_NAME+"_"+ActiveRec->Params+".svg";

		FILE * File = fopen((char *)Filename.data(), "w");
		if (!File)
			return -1;

		SVG_GENERATE_HEADER(Svg,/*550*/550,/*425*/450);
		fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);


		SVG_GENERATE_X_AXE(Svg);
		fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);
		SVG_GENERATE_Y_AXE(Svg);
		fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);


		//legends
		Svg="<g style='font-size:16;fill:black'>\n";
		Svg+="<text x='20'  y='20'>Time(ms)</text>\n";
		Svg+="<text x='460' y='375'>Size(pixels)</text>\n";
		Svg+="<text x='5' y='420'>";
		Svg+=BENCH_COPYRIGHT_STRING_SVG;
		Svg+="</text>\n";
		Svg+="</g>\n";
		fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);


		//draw image size
		Svg="<g style='font-size:13'>\n";
		Svg+="<text x='10' y='355'>0</text>\n";
		Svg+="<g transform='rotate(90)'>\n";
		fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);

		int ImgSizePixDistMin = 15;
		int ImgSizePreviousPos = 0;
		int ImgSizePos = 0;
		int ImgSizeMax = 0;
		int ImgSize = 0;
		int AxeLength = 400;
		int AxeStart[2] = {350, 30};
		for (unsigned int j=0; j< dataVector.size(); j+=1)
		{
			CurrBenchValue = dataVector[j];
			if(!CurrBenchValue)
				continue;

			if(ImgSizeMax==0)
				ImgSize = ImgSizeMax = CurrBenchValue->width * CurrBenchValue->height;
			else
				ImgSize = CurrBenchValue->width * CurrBenchValue->height;

			//size position:
			ImgSizePos = (int)((float)ImgSize*AxeLength)/ImgSizeMax + AxeStart[1];
			if(ImgSizePos == ImgSizePreviousPos)
				continue;
			if(ImgSizePreviousPos==0 || (ImgSizePreviousPos - ImgSizePixDistMin > ImgSizePos))
			{//we draw image size
				//		Svg+="<g transform='rotate(45)'>\n";
				sprintf(TempBuff, "<text x='%d' y='%d'>%d*%d</text>\n",AxeStart[0], (-1)*(ImgSizePos),CurrBenchValue->width,CurrBenchValue->height);
				fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
				sprintf(TempBuff, "<line x1='347' x2='350' y1='%d' y2='%d'  style='stroke-width:2 ; fill:none; stroke:black'/>\n",(-1)*(ImgSizePos+5),(-1)*(ImgSizePos+5));\
					fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
				//		Svg+="</g>\n";
			}
			ImgSizePreviousPos = ImgSizePos;
		}
		Svg="</g>\n";
		Svg+="</g>\n";
		fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);

		//look for the maxvalue of the average time
		maxvalue=0;
		for (size_t j=0; j< dataVector.size(); j+=1)
		{
			CurrBenchValue = dataVector[j];
			if(!CurrBenchValue)
				continue;
			if (CurrBenchValue->avg > maxvalue)
				maxvalue=CurrBenchValue->avg;
		}

		//define the max value of the ordinate
		if (maxvalue<=2500) maxvalue=2500;
		else if (maxvalue<=5000) maxvalue=5000;
		else if (maxvalue%10000!=0) maxvalue=(1+(maxvalue/10000))*10000;


		//graph name and params
		sprintf(TempBuff, "<text x='225'  y='25' style='font-family:Verdana; font-size:20; fill:black'>%s</text>\n",CurFct->FCT_NAME.data());
		fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
		sprintf(TempBuff, "<text x='225'  y='45' style='font-family:Verdana; font-size:13; fill:black'>%s</text>\n",ActiveRec->Params.data());
		fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);


		//Draw GPUCV line
		int Pt1[2]={0,0};
		int Pt2[2]={0,0};

		DRAW_LINES(Svg, dataVector, GPUCV_IMPL_GLSL_STR, BENCH_COLOR_GPUCV, _BENCH_GLSL_STR);
		fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);
		DRAW_LINES(Svg, dataVector, _BENCH_OPENCV_TYPE, BENCH_COLOR_OPENCV, _BENCH_OPENCV_TYPE);
		fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);
		DRAW_LINES(Svg, dataVector, _BENCH_CUDA_TYPE, BENCH_COLOR_CUDA, _BENCH_CUDA_TYPE);
		fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);

		DRAW_LINES(Svg, dataVector, _BENCH_IPP_TYPE, BENCH_COLOR_IPP, _BENCH_IPP_TYPE);
		fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);


		Svg="<g style='font-size:13'>\n";
		fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);
		if (maxvalue>3000)
		{
			for (unsigned int j=1,yy=295; j<6;j++)
			{
				sprintf(TempBuff, "<text x='10' y='%d'>%d</text>\n",yy,(int)(j*maxvalue/5000));
				fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
				yy-=60;
			}
		}
		else
		{
			for (unsigned int j=1,yy=295; j<6;j++)
			{
				float m = (float)maxvalue;
				sprintf(TempBuff, "<text x='10' y='%d'>%.1f</text>\n",yy,j*m/5000);
				fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
				yy-=60;
			}
		}
		Svg="</g>\n";
		Svg+="<!-- Your SVG elements -->\n";
		Svg+="</svg>\n";
		fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);

		fclose(File );
		//}//end for
	}//end  if (CurFct)
	return 1;
#else
	printf("\n_GPUCV_SVG_CREATE_FILES flag is not defined in definitions.h, SVG files not created...");
	return 0;
#endif
}//end  this function

int CL_APPLI_TRACER::GenerateSvgFile(char*_path)
{
#if _GPUCV_SVG_CREATE_FILES
	printf("\nGenerating SVG files...");
	for (unsigned int i = 0; i<this->FctTrc.size(); i++)
	{
		//if(FctTrc[i]->TimeNbr!=0)
		{
			// AddFctToSvgFile(_path, FctTrc[i]);
		}
	}
	printf("Done!");
#if 0//_GPUCV_SVG_CONVERT_TO_PNG
	printf("\nGenerating PNG files...");
	//"/k java.exe -jar D:\\batik-1.6\\batik-rasterizer.jar -m image/png SVG\\*.svg "
	string Command = "/k java.exe -jar ";
	Command += _GPUCV_SVG_BATIK_RASTERIZER_PATH;
	Command += " -m image/png ";
#ifdef _GPUCV_SVG_BATIK_RASTERIZER_SIZE
	Command += _GPUCV_SVG_BATIK_RASTERIZER_SIZE;
#else
	Command += "-w 640 -h 480";
#endif
	Command += _GPUCV_SVG_EXPORT_PATH;
	Command += "\\*.svg";

	//PH 20/01/06:
	//				#ifdef _VCC2005
	//					printf("Not implemented on Visual C++ binaries...");
	//				#else
	//				ShellExecute(0,0,"cmd.exe",Command.data(),0,SW_HIDE);
	//				printf("Done!");
	//				#endif
	string cmdline = "cmd ";
	cmdline += Command.data();
	system(cmdline.data());
	printf("Done!");

#endif
	return 1;
#else
	printf("\n_GPUCV_SVG_CREATE_FILES flag is not defined in definitions.h, SVG files not created...");
	return 0;
#endif
}


/*
int CL_APPLI_TRACER::AddFctToSvgFile(FCT_TRACER * CurFct)
{
#if _GPUCV_SVG_CREATE_FILES
if (CurFct)
{
string Svg;
string Filename;
char TempBuff[256];
CurFct->Sort();
vector <CL_FCT_SUB_ARGS*> Records;
CL_FCT_SUB_ARGS * ActiveRec=NULL;
CL_TimerVal * CurTime =NULL;
CL_TimerVal Average;
unsigned long int maxvalue;
unsigned int NParamsEqual=1;
int NbParamsFirst=true;
int pst[12];
struct Pts {unsigned int ImgSize[2];
unsigned long int time;
}ChartPts[128];
string ActiveParams="";
string Params="";


for (unsigned int i=0; i<CurFct->Records.size(); i+=NParamsEqual)
{
//if (ActiveParams!=CurFct->Records[i]->Params)ActiveParams = CurFct->Records[i]->Params;


if (NParamsEqual==1)
{ActiveRec = CurFct->Records[i];
for (unsigned int j=i+1; j<CurFct->Records.size(); j++)
if (ActiveRec->Params == CurFct->Records[j]->Params)NParamsEqual++;
NbParamsFirst=true;

}

for (unsigned int t=i; t<i+NParamsEqual; t++)
{
ActiveRec = CurFct->Records[t];
//printf("\n%s - %s - %s %d",CurFct->FCT_NAME.data(),  ActiveRec->Params.data(), ActiveRec->Type.data(), ActiveRec->SizeIn[0]);
Average = ActiveRec->TotalTime;
Average/= ActiveRec->NbrOfTime;
ChartPts[t-i].time = Average.GetTimeInUsec();
}


Filename = _GPUCV_SVG_EXPORT_PATH;
Filename += "\\"+CurFct->FCT_NAME+"_"+ActiveRec->Params+".svg";

OpenFile((char *)Filename.data());
if (!File) return -1;
//this->Sort();

Svg = "<svg width='550' height='425'>\n";

Svg+="<line x1='49' x2='480' y1='350' y2='350'  style='stroke-width:3 ; fill:none; stroke:blue'/>\n";
Svg+="<line x1='470' x2='480' y1='347' y2='350'  style='stroke-width:3 ; fill:none; stroke:blue'/>\n";
Svg+="<line x1='470' x2='480' y1='353' y2='350'  style='stroke-width:3 ; fill:none; stroke:blue'/>\n";
fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);

for(unsigned int j=0,ps=400;j<6;j++)
{sprintf(TempBuff, "<line x1='%d' x2='%d' y1='347' y2='350'  style='stroke-width:2 ; fill:none; stroke:blue'/>\n",50+ps,50+ps);
fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
ps/=4;
}

Svg="<line x1='50' x2='47' y1='25' y2='35'  style='stroke-width:3 ; fill:none; stroke:blue'/>\n";
Svg+="<line x1='50' x2='53' y1='25' y2='35'  style='stroke-width:3 ; fill:none; stroke:blue'/>\n";
Svg+="<line x1='50' x2='50' y1='351' y2='25'  style='stroke-width:3 ; fill:none; stroke:blue'/>\n";
fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);

for(int j=0;j<6;j++)
{sprintf(TempBuff, "<line x1='50' x2='53' y1='%d' y2='%d'  style='stroke-width:2 ; fill:none; stroke:blue'/>\n",80+45*j,80+45*j);
fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);

}
Svg="<g style='font-size:16'>\n";
Svg+="<text x='20'  y='25'>Time(ms)</text>\n";
Svg+="<text x='460' y='365'>Size(pixels)</text>\n";
Svg+="</g>\n";
fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);

Svg="<g style='font-size:13'>\n";
Svg+="<text x='25' y='350'>0</text>\n";
Svg+="<g transform='rotate(90)'>\n";
fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);

for(unsigned int j=0,pp=2048,ps=400;j<4;j++)
{
sprintf(TempBuff, "<text x='355' y='%d'>%d*%d</text>\n",(-1)*(ps+46),pp,pp);
fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
pp/=2;
ps/=4;
}
Svg="</g>\n";
Svg+="</g>\n";
fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);

maxvalue=0;
for (unsigned int j=0;j<NParamsEqual;j++)
{if (ChartPts[j].time>maxvalue) maxvalue=ChartPts[j].time;
}
for (unsigned int j=0;j<NParamsEqual;j++){
pst[j]=350-270*ChartPts[j].time/maxvalue;}

sprintf(TempBuff, "<text x='225'  y='15' style='font-family:Verdana; font-size:15; fill:black'>%s</text>\n",CurFct->FCT_NAME.data());
fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
sprintf(TempBuff, "<text x='225'  y='35' style='font-family:Verdana; font-size:15; fill:black'>%s</text>\n",ActiveRec->Params.data());
fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);

Svg="<g style='stroke:black; fill:black' >\n";
fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);

for (unsigned int j=0,ps=400;j<6;j++)
{
sprintf(TempBuff, "<circle cx='%d'  cy='%d' r='3'  />\n",ps+50,pst[2*j]);
fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
sprintf(TempBuff, "<circle cx='%d'  cy='%d' r='3'  />\n",ps+50,pst[2*j+1]);
fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
ps/=4;
}

Svg="</g>\n";
fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);

sprintf(TempBuff, "<polyline points='51,%d,52,%d,56,%d,75,%d,150,%d,450,%d' style='stroke-width:3 ; fill:none; stroke:red'/>\n",pst[10],pst[8],pst[6],pst[4],pst[2],pst[0]);
fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
sprintf(TempBuff, "<polyline points='51,%d,52,%d,56,%d,75,%d,150,%d,450,%d' style='stroke-width:3 ; fill:none; stroke:green'/>\n",pst[11],pst[9],pst[7],pst[5],pst[3],pst[1]);
fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);

sprintf(TempBuff, "<text x='460'  y='%d' style='font-family:Verdana; font-size:15; fill:red'>OpenCV</text>\n",pst[0]);
fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
sprintf(TempBuff, "<text x='460'  y='%d' style='font-family:Verdana; font-size:15; fill:green'>GPUCV</text>\n",pst[1]);
fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);

Svg="<g style='font-size:13'>\n";
fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);
for (int j=1,yy=305; j<7;j++)
{
sprintf(TempBuff, "<text x='20' y='%d'>%d</text>\n",yy,j*maxvalue/6000);
fwrite( TempBuff, strlen(TempBuff)*sizeof(unsigned char),1,File);
yy-=45;
}

Svg="</g>\n";

Svg+="<!-- Your SVG elements -->\n";
Svg+="</svg>\n";
fwrite( Svg.data(), strlen(Svg.data())*sizeof(unsigned char),1,File);

CloseFile();
//if (File) return -1;

}//end for
}//end  if (CurFct)
return 1;
#else
printf("\n_GPUCV_SVG_CREATE_FILES flag is not defined in definitions.h, SVG files not created...");
return 0;
#endif

}//end  this function


int CL_APPLI_TRACER::GenerateSvgFile()
{
#if _GPUCV_SVG_CREATE_FILES
printf("\nGenerating SVG files...");
for (unsigned int i = 0; i<this->FctTrc.size(); i++)
AddFctToSvgFile(FctTrc[i]);
printf("Done!");
#if _GPUCV_SVG_CONVERT_TO_PNG
#ifndef _VCC2005
printf("\nGenerating PNG files...");
//"/k java.exe -jar D:\\batik-1.6\\batik-rasterizer.jar -m image/png SVG\\*.svg "
string Command = "/k java.exe -jar ";
Command += _GPUCV_SVG_BATIK_RASTERIZER_PATH;
Command += "-m image/png";
Command += _GPUCV_SVG_EXPORT_PATH;
Command += "\\*.svg";
ShellExecute(0,0,"cmd.exe",Command.data(),0,SW_HIDE);
printf("Done!");
#endif
#endif
return 1;
#else
printf("\n_GPUCV_SVG_CREATE_FILES flag is not defined in definitions.h, SVG files not created...");
return 0;
#endif
}
*/

void CL_APPLI_TRACER::Clear()
{
	for (unsigned int i=0; i< FctTrc.size(); i++)
		FctTrc[i]->Clear();
	FctTrc.clear();
	this->FctNbr = 0;
	this->Max.Clear();
}

/**\brief Read time from the system until it started, and put it into tp. Time is in second and micro seconds.
*
* int Ygettimeofday(struct timeval * tp) fill tp struct with the time elapsed until the computer started, in seconds and micro seconds.
* Should work on different systems like MS Windows, Linux and MacOS.
*
* tp		 a pointer to struct timeval that will contain the system time
*\return 1 when the function completes normally, -1 otherwise
*/
int Ygettimeofday(struct timeval * tp)
{
#if defined (_WINDOWS)
	LARGE_INTEGER ticksPerSecond;
	LARGE_INTEGER tick;   // A point in time
	LARGE_INTEGER time;   // For converting tick into real time

	// get the high resolution counter's accuracy
	QueryPerformanceFrequency(&ticksPerSecond);

	// what time is it?
	QueryPerformanceCounter(&tick);

	// convert the tick number into the number of seconds
	// since the system was started...
	time.QuadPart = tick.QuadPart/ticksPerSecond.QuadPart;
	tp->tv_sec = (long)time.QuadPart;

	//get the number of hours
	LONGLONG hours = time.QuadPart/3600;

	//get the number of minutes
	time.QuadPart = time.QuadPart - (hours * 3600);
	LONGLONG minutes = time.QuadPart/60;

	//get the number of seconds
	LONGLONG  seconds = time.QuadPart - (minutes * 60);

	tp->tv_usec = long(tick.QuadPart % ticksPerSecond.QuadPart / (float)ticksPerSecond.QuadPart * 1000000);

	return 0;//normal return
#elif defined(__linux) || defined(__APPLE__)
	return gettimeofday(tp, NULL );
#else
	return gettimeofday(tp );
#endif
}
//==================================================================================


//! gettimediff() get the time difference between t1 and t2 and put it into tres.
//! Notes : tres handler can be the same handler as t1 or t2.
void gettimediff(struct timeval * t1, struct timeval * t2, struct timeval * tres)
{
	struct timeval res;
	cleartimeval(&res);


	//micro seconde test
	res.tv_usec = t1->tv_usec - t2->tv_usec;
	if (res.tv_usec < 0)
	{
		res.tv_sec --;
		res.tv_usec += 1000000;//-res.tv_usec;
	}
	//seconde test
	res.tv_sec += t1->tv_sec - t2->tv_sec;


	tres->tv_sec = res.tv_sec;
	tres->tv_usec = res.tv_usec;
}

//! gettimesum() get the time addition of t1 and t2, then put it into tres.
//! Notes : tres handler can be the same handler as t1 or t2.
void gettimesum(struct timeval * t1, struct timeval * t2, struct timeval * tres)
{
	struct timeval res;
	cleartimeval(&res);

	if (!(t1 && t2 && tres)) return;

	//micro seconde test
	res.tv_usec = t1->tv_usec + t2->tv_usec;
	while (res.tv_usec > 1000000)
	{
		res.tv_sec ++;
		res.tv_usec -= 1000000;
	}
	//seconde test
	res.tv_sec += t1->tv_sec + t2->tv_sec;

	tres->tv_sec = res.tv_sec;
	tres->tv_usec = res.tv_usec;
}


//! gettimedivide() get the division of t1 by 'divide' , then put it into tres.
//! Notes : tres handler can be the same handler as t1.
void gettimedivide(struct timeval * t1, int divide, struct timeval * tres)
{
	struct timeval res;
	cleartimeval(&res);
	LONGLONG sum = 0;

	if (!(t1 && tres)) return;
	if (divide == 0) return;

	sum = t1->tv_sec * 1000000 + t1->tv_usec;
	sum /= divide;

	//seconde test
	res.tv_sec = (long)sum / 1000000;


	//micro seconde test
	res.tv_usec = t1->tv_usec / divide;
	res.tv_usec += (long)(t1->tv_sec % divide / (float)divide * 1000000);


	tres->tv_sec = res.tv_sec;
	tres->tv_usec = res.tv_usec;
}

//! cleartimeval() set the timeval struct to 0.
void cleartimeval(struct timeval * t1)
{
	if (t1)
		t1->tv_sec = t1->tv_usec = 0;
}

/** @} */ // end of GPUCV_MACRO_BENCH_GRP

#endif//0 deprecated
