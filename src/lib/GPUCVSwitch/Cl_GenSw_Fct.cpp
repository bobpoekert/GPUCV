#include "StdAfx.h"
#include <GPUCVSwitch/macro.h>
#include <GPUCVSwitch/Cl_GenSw_Fct.h>
#include <GPUCVCore/ToolsTracer.h>
#include <iostream>
#include <fstream>
#include <string>

namespace GCV{

#define SW_ARR_GPU_IMPLS ",\n{GPUCV_IMPL_GLSL, NULL, NULL, true, GenericGPU::HRD_PRF_2},\n\
{GPUCV_IMPL_CUDA, NULL, NULL, true, GenericGPU::HRD_PRF_CUDA}\n}; \n "

using namespace std;
using namespace GCV;

/*static*/ SGE::CL_TEMPLATE_OBJECT_MANAGER<CL_GenSwFn, std::string> * CL_GenSwFn::m_FctObjMngr = NULL;

CL_GenSwFn::CL_GenSwFn(const std::string &_name):
SGE::CL_BASE_OBJ<std::string>(_name)
,m_Fntype("")
,m_FnName("")
,m_Argstr("")
,m_SwFnType("")
,m_Dst_Arr()
,m_Src_Arr()
,m_requireSwitching(false)
//,m_Args_Type()
//,m_Args_Name()
//,m_Args_DefltVal()
{

};

bool CL_GenSwFn::MatchBrackets(std::string &currArg, size_t &StartBracketPos, size_t &StopBracketPos)
{

	std::string TempStr=currArg;

	size_t Start	= 0;
	size_t End		= 0;
	bool loop		= true;

	int StartBrkts	= 0;
	int EndBrkts	= 0;
	StartBracketPos = StopBracketPos = 0;

	do 
	{
		Start			= TempStr.find_first_of("(");

		if (Start!=std::string::npos)
		{
			if(Start>1)
			{
				//Start			= TempStr.find_first_of("(");
				//if (Start!=std::string::npos)
				//	break;
				StartBracketPos = Start;
				//StartBrkts++;
				for(size_t i =Start; i < TempStr.size(); i++)
				{
					if(TempStr[i]==')')
						EndBrkts++;
					else if(TempStr[i]=='(')
						StartBrkts++;
					if(EndBrkts==StartBrkts)
					{
						StopBracketPos = i;
						loop=false;
						break;
					}
				}
				//StartBrkts++;
			}
			else
				TempStr = TempStr.substr(Start+1, TempStr.size()-Start-1);
			/*End				= TempStr.find_first_of(")");
			if (End!=std::string::npos)
			{
				EndBrkts++;
			}
			else loop=false;
			*/
		}
		else loop=false;
	}
	while (loop);

	if(StartBracketPos!= StopBracketPos && StopBracketPos!=0)
		return true;//>1 && StartBrkts==EndBrkts) return true;
	else
		return false;


}
//================================================
bool CL_GenSwFn::ParseLine(const std::string &line)
{
	std::string args, currArg, subarg, func_name_args;

	size_t Start	= 0;
	size_t End		= 0;
	size_t Start_sp = 0;
	size_t Start_st = 0;
	size_t Start_am = 0;

	//read function type, name and args...
	Start			= line.find_first_of(" ");
	if(Start==std::string::npos)
		return false;
	m_Fntype = line.substr(0, Start);
	func_name_args	= line.substr(Start, line.size());
	SGE::StrTrimLR(func_name_args);
	//check the const califier
	if(m_Fntype=="const")
	{
		Start = func_name_args.find_first_of(" ");
		m_Fntype += " ";
		m_Fntype += func_name_args.substr(0, Start);
		func_name_args = func_name_args.substr(Start, line.size());
	}
	//get function name
	SGE::StrTrimLR(func_name_args);
	Start			= func_name_args.find_first_of("(");
	if(Start==std::string::npos)
		return false;
	m_FnName		= func_name_args.substr(0, Start);
	if(m_FnName == "cvDilate")
	{
		m_FnName = m_FnName;
	}
	//get function short name without cv
	size_t Pos	=	m_FnName.find_first_of("cv");
	if (Pos!=std::string::npos)
		m_FnShortName =	m_FnName.substr(Pos+2, m_FnName.size());
	else
		m_FnShortName =	m_FnName;
	//get args
	End				= func_name_args.find_first_of(";");
	if(Start==std::string::npos)
		return false;
	args			= func_name_args.substr(Start, End);
	m_Argstr		= args;
	SGE::StrTrimLR(m_FnName);
	//SGE::StrTrimLR(m_Argstr);

	//====================
	bool loop=true;

	size_t BracketPosStart=0;
	size_t BracketPosStop=0;
	//parse function args.
	do 
	{
		if(args=="")
			break;
		Start			= args.find_first_of(","); // write a function that check the number of open nd close brackets then include ot disclude accordingly the arguement

		if(MatchBrackets(args, BracketPosStart, BracketPosStop))
		{
			if(BracketPosStart<Start)
			{
				currArg = args.substr(1, BracketPosStop/*BracketPosStart-1*/);
				//??? bug when having several optional arguments? Start = -1;
			}
			else
				currArg			= args.substr(1, Start-1);
		}
		else
			currArg			= args.substr(1, Start-1);
/*
		if (MatchBrackets(currArg))
		{
			Start			= currArg.find_first_of("(");
			subarg			= currArg.substr(1,Start);
			//if(currArg.find_first_of(" "))
			m_Args_Type.push_back(subarg.substr(1,currArg.find_first_of(" ")));
			m_Args_Name.push_back(subarg.substr(currArg.find_first_of(" "),currArg.size()));

			currArg		= currArg.substr(currArg.find_last_of(")")+1, currArg.size());
		}
*/		if (currArg.find_last_of(";")!=std::string::npos)
			loop=false;
		else if (Start==std::string::npos)
			loop=false;

		args			= args.substr(Start+1,args.size());
		if(args!="")
		{
			FctAgrs NewArgs = ParseArgument(currArg);
			m_ArgsList.push_back(NewArgs);
		}
	}
	while(loop);//End==std::string::npos);

	//Start			= func_type.find_last_of(" ");
	//m_FnName		= currArg.substr(Start+1,func_type.size()-1);
	//m_Fntype		= currArg.substr(0, Start);

	m_SwFnType		= GenSwFnTypeDef();
	SetID(m_FnName);
	return true;
};

CL_GenSwFn::~CL_GenSwFn()
{
};



#define GPUCV_SWITCH__LICENSE_FILE	GetGpuCVSettings()->GetShaderPath() + std::string ("../etc/sed/license.h")
#define GPUCV_SWITCH__TOP_H_FILE	GetGpuCVSettings()->GetShaderPath() + std::string ("../etc/sed/cv.h/cv.top.h")
#define GPUCV_SWITCH__TOP_CPP_FILE	GetGpuCVSettings()->GetShaderPath() + std::string ("../etc/sed/cv.h/cv.top.cpp")
#define GPUCV_SWITCH__BOTTOM_H_FILE GetGpuCVSettings()->GetShaderPath() + std::string ("../etc/sed/cv.h/cv.bottom.h")
#define GPUCV_SWITCH__BOTTOM_CPP_FILE	GetGpuCVSettings()->GetShaderPath() + std::string ("../etc/sed/cv.h/cv.bottom.cpp")


void CL_GenSwFn::AppendFileToFnsFile(ofstream & fileswfns, std::string FileName)
{
	SG_AssertFile(SGE::CL_BASE_OBJ_FILE::FileExists(FileName),FileName, "File does not exist");

	fstream NewFile(FileName.data());
	char str[2000];
	while (!NewFile.eof())
	{
		NewFile.getline(str, 2000);
		fileswfns << str << std::endl;
	}
	NewFile.close();
}

void CL_GenSwFn::AddObjsToFns_H(std::string Outfilename, std::string OUT_FILEPATH) //as argument
{
	GPUCV_NOTICE("\n\tGenerating "<< OUT_FILEPATH << "/" <<Outfilename);

	//SG_Assert(CL_GenSwFn::GetFctObjMngr()->GetCount()==0, "CL_GenSwFn::AddObjsToFns_H() must be called after funtcion CL_GenSwFn::AddObjsToFns_CPP()");

	ifstream funcfile;
	ofstream fileswfns;
	string FileOut = OUT_FILEPATH;
	FileOut += "/";
	FileOut += Outfilename;
	fileswfns.open(FileOut.data());
	std::string  swfns;

	std::string BaseFileName = Outfilename.substr(0, Outfilename.size()-2);//remove ".h"
	std::string HeaderFileMacro = "__";
	HeaderFileMacro += BaseFileName;
	HeaderFileMacro += "_H";
	HeaderFileMacro = SGE::StringToUpper(HeaderFileMacro );



	//Create an output file
		AppendFileToFnsFile(fileswfns, GPUCV_SWITCH__LICENSE_FILE);
		AppendFileToFnsFile(fileswfns, GPUCV_SWITCH__TOP_H_FILE);

		swfns	= "#ifndef ";
		swfns	+= HeaderFileMacro;
		swfns	+= "\n#define ";
		swfns	+= HeaderFileMacro;
		swfns	+= "\n";
		fileswfns << swfns ;
		swfns="";

		fileswfns << "\n#include <" << BaseFileName<< "/config.h>\n";
		int FctNbr =0;
		CL_GenSwFn *LocalFctGen = NULL;
		std::string SpareLine;
		
		//export macro
		std::string ExportStr = "_";
		ExportStr += SGE::StringToUpper(BaseFileName);
		ExportStr += "_EXPORT_C ";

		std::string ExportStrCPP = "_";
		ExportStrCPP += SGE::StringToUpper(BaseFileName);
		ExportStrCPP += "_EXPORT ";
		
		int iPos=0;
		fileswfns << "#ifdef __cplusplus\n";
		fileswfns << ExportStrCPP << " void cvg_"<< BaseFileName<<"_RegisterTracerSingletons(SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> * _pAppliTracer, SG_TRC::CL_TRACING_EVENT_LIST *_pEventList);\n";
		fileswfns << "#endif\n";
		


		//start iterating functions
		SGE::CL_TEMPLATE_OBJECT_MANAGER<CL_GenSwFn, std::string>::iterator itFunct;

		
		for(itFunct=CL_GenSwFn::GetFctObjMngr()->GetFirstIter();
			itFunct!=CL_GenSwFn::GetFctObjMngr()->GetLastIter();
			itFunct++)
		{
			//if(!itFunct)
			//	continue;
			swfns += ExportStr;
			swfns	+=	(*itFunct).second->GenSwFnDeclaration(FILETYPE_H);
		}
		fileswfns << swfns;
		fileswfns << "/*........End Declaration.............*/" << "\n";
		AppendFileToFnsFile(fileswfns, GPUCV_SWITCH__BOTTOM_H_FILE);
		swfns	= "\n#endif //";
		swfns	+= HeaderFileMacro;
		fileswfns << swfns;
		fileswfns.close();
}


void CL_GenSwFn::AddObjsToFns_H_WRAPPER(std::string Outfilename, std::string OUT_FILEPATH) //as argument
{
	GPUCV_NOTICE("\n\tGenerating "<< OUT_FILEPATH << "/" <<Outfilename);
	//SG_Assert(CL_GenSwFn::GetFctObjMngr()->GetCount()==0, "CL_GenSwFn::AddObjsToFns_H() must be called after funtcion CL_GenSwFn::AddObjsToFns_CPP()");

	ifstream funcfile;
	ofstream fileswfns;
	string FileOut = OUT_FILEPATH;
	FileOut += "/";
	FileOut += Outfilename;
	FileOut += "_wrapper.h";
	fileswfns.open(FileOut.data());
	std::string  swfns;

	std::string BaseFileName = Outfilename.substr(0, Outfilename.size()-2);//remove ".h"
	std::string HeaderFileMacro = "__";
	HeaderFileMacro += BaseFileName;
	HeaderFileMacro += "_H";
	HeaderFileMacro = SGE::StringToUpper(HeaderFileMacro );



	//Create an output file
	AppendFileToFnsFile(fileswfns, GPUCV_SWITCH__LICENSE_FILE);
	AppendFileToFnsFile(fileswfns, GPUCV_SWITCH__TOP_H_FILE);

	int FctNbr =0;
	CL_GenSwFn *LocalFctGen = NULL;
	std::string SpareLine;
	int iPos=0;;


	SGE::CL_TEMPLATE_OBJECT_MANAGER<CL_GenSwFn, std::string>::iterator itFunct;

	swfns	+= "#ifndef ";
	swfns	+= HeaderFileMacro;
	swfns	+= "\n#define ";
	swfns	+= HeaderFileMacro;
	swfns	+= "\n";

	fileswfns << swfns;
	
	//include file containing the cvgws* functions...
	swfns	= "\n\n#include <";
	swfns	+= Outfilename;
	swfns	+= "/";
	swfns	+= 	Outfilename;
	swfns	+= 	".h";
	swfns	+= ">\n";
	fileswfns << swfns;

	std::string BaseName;
	std::string NewName;
	size_t Start=0;

	for(itFunct=CL_GenSwFn::GetFctObjMngr()->GetFirstIter();
		itFunct!=CL_GenSwFn::GetFctObjMngr()->GetLastIter();
		itFunct++)
	{
		//get function names
		BaseName = (*itFunct).second->GetIDStr();
		NewName = _GCV_SWITCH_FCT_PREFIX;//"cvgSw";
		NewName	+=	(*itFunct).second->m_FnShortName;

		//write definitions
		//swfns ="\n#ifdef ";
		//swfns +=	BaseName;
		//swfns +="\n#undef "	;
		//swfns +=	BaseName;
		swfns ="\n#define ";
		swfns +=	BaseName;
		swfns +="\t ";
		swfns +=	NewName;
		//swfns +="\n;//#endif\n";
		fileswfns << swfns;
	}
	fileswfns << "\n/*........End Declaration.............*/" << "\n";
	AppendFileToFnsFile(fileswfns, GPUCV_SWITCH__BOTTOM_H_FILE);
	swfns	= "\n#endif //";
	swfns	+= HeaderFileMacro;
	fileswfns << swfns;
	fileswfns.close();
}


void CL_GenSwFn::AddObjsToFns_CPP(	std::string Infilename, std::string IN_FILEPATH, \
				std::string Outfilename, std::string OUT_FILEPATH) //as argument
{
	GPUCV_NOTICE("\n\tGenerating "<< OUT_FILEPATH << "/" <<Outfilename);
	GPUCV_NOTICE("\tFrom "<< IN_FILEPATH << "/" <<Infilename);
	
	ifstream funcfile;
	ofstream fileswfns;
	string FileIn = IN_FILEPATH;
	FileIn += "/";
	FileIn += Infilename;
	string FileOut = OUT_FILEPATH;
	FileOut += "/";
	FileOut += Outfilename;
	funcfile.open(FileIn.data(),  ifstream::in);
	fileswfns.open(FileOut.data());
	std::string line, swfns;

	//Create an output file
	if(funcfile.is_open())
	{
		AppendFileToFnsFile(fileswfns, GPUCV_SWITCH__LICENSE_FILE);
		AppendFileToFnsFile(fileswfns, GPUCV_SWITCH__TOP_CPP_FILE);

		
		size_t DotPos = Outfilename.find_last_of(".");
		if(DotPos != std::string::npos)
			swfns	= Outfilename.substr(0, DotPos);
		else
			swfns	= Outfilename;



		fileswfns << "\n#include <" << swfns << "/" << swfns << ".h>\n";
		fileswfns << "#include <GPUCVSwitch/switch.h>";
		

		//add a first common function to register profiling singletons
		std::string BaseFileName = Outfilename.substr(0, Outfilename.size()-4);//remove ".h"
		fileswfns << std::endl << "/*====================================*/" <<std::endl;
		fileswfns << "void cvg_"<< BaseFileName<<"_RegisterTracerSingletons(SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> * _pAppliTracer, SG_TRC::CL_TRACING_EVENT_LIST *_pEventList)\n";
		fileswfns << "{\nSG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance().RegisterNewSingleton(_pAppliTracer);\n";
		fileswfns << "SG_TRC::CL_TRACING_EVENT_LIST::Instance().RegisterNewSingleton(_pEventList);\n";
		fileswfns << "}\n/*====================================*/\n";


		int FctNbr =0;
		CL_GenSwFn *LocalFctGen = NULL;
		std::string SpareLine;
		size_t iPos=0;;
		do
		{
			//we check that we do not have several function definition on the same line
				if(SpareLine=="")
					getline (funcfile,line);
				else
				{
					getline (funcfile,line);
					line = SpareLine + line;
					SpareLine="";
				}

				iPos=line.find_first_of(";");

				if(iPos==std::string::npos)
				{//we grab also next line
					std::string nextline ;
					getline (funcfile,nextline);
					line+=nextline;
				}
			
				//get only one function definition
				if(iPos!=std::string::npos)
				{//we have a ';'
					if(iPos!=line.size()-1)
					{//it is not the end of the line
						line = line.substr(0, iPos);
						SpareLine = line.substr(iPos, line.size());
					}
				}
			//=====================

			if(line=="")
				continue;
			//funcfile.getline(line);
			if(!line.empty())
			{
				LocalFctGen = new CL_GenSwFn("");
				if(LocalFctGen->ParseLine(line))
				{
					if(CL_GenSwFn::GetFctObjMngr()->Find(LocalFctGen->GetID()))
					{
						GPUCV_WARNING("A function has been found twice in the header file: \n "<< LocalFctGen->GetID());
						delete LocalFctGen;
						continue;
					}
					else if (LocalFctGen->GetID().substr(0, 2) != "cv")
					{
						GPUCV_WARNING("A function has not a cv* like name, exiting wrapping: \n "<< LocalFctGen->GetID());
						delete LocalFctGen;
						continue;
					}
					CL_GenSwFn::GetFctObjMngr()->AddObj(LocalFctGen);

					//call generate Funct and concat result into a string

					//swfns    =  LocalFctGen->m_Fntype;
					LocalFctGen->ParseArgsType();
					swfns	= "\n/*====================================*/\n";
					swfns	+=	LocalFctGen->GenSwFnDeclaration(FILETYPE_CPP);
					swfns	+= "{\n";

					if(LocalFctGen->m_Src_Arr.empty() 
						&& LocalFctGen->m_Dst_Arr.empty() 
						&& !LocalFctGen->m_requireSwitching
						)
					{//no input or output images, no need to switch!
						GPUCV_NOTICE("Function do not have input image:" << LocalFctGen->GetID());
						GPUCV_NOTICE("Function do not have output image:" << LocalFctGen->GetID());

						swfns	+=	LocalFctGen->GenNoSwitch();
					}
					else
					{//we switch
						swfns	+=	LocalFctGen->GenSwFnTypeDef();
						//swfns	+=	LocalFctGen->GenSwchImpnsArr(); no need any more...
						swfns	+=	LocalFctGen->GenSwchFnName();
						swfns	+=	LocalFctGen->GenSwchInOutArr();
						swfns	+=	LocalFctGen->GenStrtRunStop();
					}
					swfns	+= "}\n";
					fileswfns << swfns << "\n";
					FctNbr++;
				}
				else
				{
					GPUCV_ERROR("Error parsing line: \n=>" << line);
				}
			}
			//if (FctNbr>10)
			//	break;//just for debugging....
		}
		while (!funcfile.eof());
		fileswfns << "/*........End Code.............*/" << "\n";
		AppendFileToFnsFile(fileswfns, GPUCV_SWITCH__BOTTOM_CPP_FILE);
		funcfile.close();
		fileswfns.close();
		//save output file/...
		/*		string outfile = OUT_FILEPATH;
		outfile += "/";
		outfile += Outfilename;
		*/
		
	}

	else 

		GPUCV_NOTICE("Empty File");
}

/* To generate the following string -

void cvswSub(CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask)

*/
std::string CL_GenSwFn::GenSwFnDeclaration(ENUM_SrcFileType _fileType)
{		
	std::string SwitchFnDec;

	SwitchFnDec		= m_Fntype;
	SwitchFnDec		+=	" ";
	SwitchFnDec		+=	_GCV_SWITCH_FCT_PREFIX;//"cvgSw"
	SwitchFnDec		+=	m_FnShortName;
	SwitchFnDec		+=	"(";

	if(m_FnShortName=="DrawContours")
		int i=0;
	for (size_t i = 0; i < m_ArgsList.size(); i++)
	{
		if(m_ArgsList[i].m_name=="")
			continue;
		if(i!=0)
			SwitchFnDec	+=  ", ";
		if(m_ArgsList[i].m_objType==Obj_Param && m_ArgsList[i].m_const==true)
			SwitchFnDec	+=  "const ";
		SwitchFnDec		+=	m_ArgsList[i].m_type;
		SwitchFnDec		+=	" ";
		SwitchFnDec		+=	m_ArgsList[i].m_name;
		if(_fileType==FILETYPE_H && m_ArgsList[i].m_defaultVal!="")
		{
			SwitchFnDec		+=	" ";
			SwitchFnDec		+=	m_ArgsList[i].m_defaultVal;
		}
	}
	if(_fileType==FILETYPE_H)
		SwitchFnDec		+= ");\n";
	else
		SwitchFnDec		+= ")\n";
	return SwitchFnDec;
}




/* To generate the following -

typedef void(*CVGType_Sub)(CvArr*,CvArr*,CvArr*,CvArr*);
*/
std::string CL_GenSwFn::GenSwFnTypeDef()
{
	std::string SwfnType;
	std::string Type, LastPart;
	std::string TempStr;
	
	int SwfnTypeNbr = 0;
	size_t Start = 0;
	
	TempStr		 = m_Argstr; //( const CvArr* src, CvArr* dst, CvPoint offset, int bordertype, CvScalar value CV_DEFAULT(cvScalarAll(0)));

	//generate function typedef name
	m_FnTypeDef = "TypeDef";
	m_FnTypeDef += _GCV_SWITCH_FCT_PREFIX;
	m_FnTypeDef = "_";
	m_FnTypeDef += m_FnShortName;
	//==========================

	SwfnType	 = "\ttypedef ";
	SwfnType	+= m_Fntype;
	SwfnType	+= "(*";
	SwfnType	+= m_FnTypeDef;
	SwfnType	+= ") (";

	//if(!m_Args_Type.empty())
	//	SwfnType	+= m_Args_Type[0];

	for (size_t i = 0; i < m_ArgsList.size(); i++ )
	{
		if(m_ArgsList[i].m_name=="")
			continue;

		if(SwfnTypeNbr!=0)
			SwfnType	+= ", ";
		//if(!m_Args_Type.empty())
		if(m_ArgsList[i].m_objType==Obj_Param && m_ArgsList[i].m_const==true)
			SwfnType	+=  "const ";
		SwfnType	+= m_ArgsList[i].m_type;
		SwfnTypeNbr++;
	}


	SwfnType	+= " ); \n";

	return SwfnType;
}


/*
To generate : GPUCV_FUNCNAME("Erode");

*/
std::string CL_GenSwFn::GenSwchFnName()
{
	std::string FnName;

	FnName = "\tGPUCV_FUNCNAME(\"";
	FnName += m_FnName;
	FnName += "\");";
	FnName += "\n";

	return FnName;

}



/*	To generate the following string -
static switchFctStruct SwitchImplementations[]={
{GPUCV_IMPL_OPENCV, (PROC)cvSub, NULL, false, GenericGPU::HRD_PRF_0}
,{GPUCV_IMPL_GLSL, (PROC)cvgSub, NULL, true, GenericGPU::HRD_PRF_2}
//,{GPUCV_IMPL_CUDA, (PROC)cvgCudaAdd, NULL, true, GenericGPU::HRD_PRF_CUDA}
};
*/
std::string CL_GenSwFn::GenSwchImpnsArr()
{
#if 0
	std::string SwOp;
	SwOp = "\tstatic switchFctStruct SwitchImplementations[]={";
	SwOp += "\n";
	SwOp += "\t{GPUCV_IMPL_OPENCV, (PROC)";
	SwOp += m_FnName;
	SwOp +=	", NULL, false, GenericGPU::HRD_PRF_0}";
	SwOp += SW_ARR_GPU_IMPLS;
	SwOp += "\n\t";


	return SwOp;
#endif
	return "";
}

void CL_GenSwFn::ParseArgsType()
{
	std::string StrngsTableSrc[] = {
		"src"
		,"mask"
		//cvIntegral
		,"image"
		,"arr"
	};
	std::string StrngsTableDst[] = {
		"dst"
		//cvIntegral
		,"sum"
		,"sqsum"
		,"tilted_sum"
		//cvWatershed
		,"markers"
	};

	std::string StrngsTableImage[] = {
		"const CvArr*"
		,"CvArr*"
		,"const CvMat*"
		,"CvMat*"
		,"const IplImage*"
		,"IplImage*"
	};

	GPUCV_NOTICE("Fct " << GetID() << " Parsing " << m_ArgsList.size() << "arguments");
	for (size_t i = 0; i < m_ArgsList.size(); i++ )
	{
		GPUCV_NOTICE("\t" << i << " args: " << m_ArgsList[i].m_type << ","<< m_ArgsList[i].m_name);
		if (CheckImageStrngs(m_ArgsList[i].m_type, StrngsTableImage, sizeof(StrngsTableImage)/sizeof(std::string) ))
		{
			if(m_ArgsList[i].m_type.find("**")!=std::string::npos)
			{//when it is a pointer to pointer we do not manage it ... 
				m_ArgsList[i].m_objType = Obj_Mngd_GpuCV;
				m_requireSwitching = true;
			}
			else if(m_ArgsList[i].m_const==true)//we assume const object are source object ;-)
			{
				m_Src_Arr.push_back(m_ArgsList[i].m_name);
				m_ArgsList[i].m_objType = Obj_Input_Arr;
			}
			else if (CheckImageStrngs(m_ArgsList[i].m_name, StrngsTableSrc, sizeof(StrngsTableSrc)/sizeof(std::string)))
			{
				m_Src_Arr.push_back(m_ArgsList[i].m_name);
				m_ArgsList[i].m_objType = Obj_Input_Arr;
			}
			else if (CheckImageStrngs(m_ArgsList[i].m_name, StrngsTableDst, sizeof(StrngsTableDst)/sizeof(std::string)))
			{
				m_Dst_Arr.push_back(m_ArgsList[i].m_name);
				m_ArgsList[i].m_objType = Obj_Output_Arr;
			}
			
			//check mask
			if(m_ArgsList[i].m_name=="mask")
			{
				m_ArgsList[i].m_objType = (ObjType)(m_ArgsList[i].m_objType | Obj_Mask_Arr);
			}
		}
		else
		{
			m_ArgsList[i].m_objType = Obj_Param;
		}
	}
	//parsing return type:
	if(CheckImageStrngs(this->m_Fntype, StrngsTableImage, sizeof(StrngsTableImage)/sizeof(std::string)))
		m_requireSwitching = true;
		
}
//CvArr* SrcARR[]={src1, src2, mask};
//CvArr* DstARR[]={dst};
std::string CL_GenSwFn::GenSwchInOutArr()
{
	std::string InArr, OutArr;

	if(m_Src_Arr.size())
		InArr = "\tCvArr* SrcARR[] = {";
	else
		InArr = "\tCvArr** SrcARR = NULL;\n";
	if(m_Dst_Arr.size())
		OutArr = "\tCvArr* DstARR[] = {";
	else
		OutArr = "\tCvArr** DstARR = NULL;\n";

	for (size_t i = 0; i < m_Src_Arr.size(); i++ )
	{
		if(i!=0)
			InArr+= ", ";
		InArr+=" (CvArr*) ";
		InArr+= m_Src_Arr[i];
	}

	for (size_t i = 0; i < m_Dst_Arr.size(); i++ )
	{
		if(i!=0)
			OutArr+= ", ";
		OutArr+=" (CvArr*) ";
		OutArr+= m_Dst_Arr[i];
	}

	if(m_Src_Arr.size()!=0)
		InArr +="};\n";
	if(m_Dst_Arr.size()!=0)
		OutArr +="};\n";

	InArr += OutArr;

	return InArr;
}





/*
SWITCH_START_OPR(dst);
RUNOP((src1,src2,dst,mask), CVGType_AddS);
SWITCH_STOP_OPR();
*/

std::string CL_GenSwFn::GenNoSwitch()
{
	std::string StRunSp="\t";
	//	size_t Start, End;
	int Nbr=0;

	if(m_Fntype!="void")
		StRunSp += "return ";

	StRunSp += m_FnName;

	StRunSp += "(";		
	for (size_t i = 0; i < m_ArgsList.size(); i++ )
	{
		if(m_ArgsList[i].m_name=="")
			continue;
		if(Nbr!=0)
			StRunSp+= ", ";
		StRunSp	+= "(";
		if(m_ArgsList[i].m_objType==Obj_Param && m_ArgsList[i].m_const==true)
			StRunSp	+=  "const ";
		StRunSp	+=  m_ArgsList[i].m_type;
		StRunSp	+=	") ";		
		StRunSp	+=	m_ArgsList[i].m_name;
		Nbr++;
	}
	StRunSp += ");\n";
	return StRunSp;
}

std::string CL_GenSwFn::GenStrtRunStop()
{
	std::string StRunSp;
//	size_t Start, End;
	int Nbr=0;

	if(m_Fntype!="void")
	{
		StRunSp += "\t";
		StRunSp += m_Fntype;
		StRunSp += " ReturnObj;";
	}
	StRunSp += "\tSWITCH_START_OPR(";
	if (!m_Dst_Arr.empty())
		StRunSp += m_Dst_Arr[0];
	else
		StRunSp += "NULL";
	StRunSp += "); \n";

	//check if we have a mask image...
	for (size_t i = 0; i < m_ArgsList.size(); i++ )
	{
		if(m_ArgsList[i].m_objType & Obj_Mask_Arr)
		{
			StRunSp += "//Mask has been found, add it to params.\n";
			StRunSp += "\t if(paramsobj &&";
			StRunSp += m_ArgsList[i].m_name+")paramsobj->AddParam(\"option\", \"MASK\");\n";
		}
	}
	
	StRunSp += "\tRUNOP((";		
	//list of args
	for (size_t i = 0; i < m_ArgsList.size(); i++ )
	{
		if(m_ArgsList[i].m_name=="")
			continue;
		if(Nbr!=0)
			StRunSp+= ", ";
		StRunSp	+= "(";
		if(m_ArgsList[i].m_objType==Obj_Param && m_ArgsList[i].m_const==true)
			StRunSp	+=  "const ";
		StRunSp	+=  m_ArgsList[i].m_type;
		StRunSp	+=	") ";		
		StRunSp	+=	m_ArgsList[i].m_name;
		Nbr++;
	}
	StRunSp += "), ";
	//==================
	//Start	=	m_SwFnType.find_first_of("*");
	//End		=	m_SwFnType.find_first_of(")");
	StRunSp += m_FnTypeDef;
	
	//return option...part 1
	StRunSp += ", ";
	if(m_Fntype!="void")
	{
//		StRunSp += m_Fntype;
		StRunSp += " ReturnObj =";
	}
	//=======================
	StRunSp += "); \n";
	StRunSp += "\tSWITCH_STOP_OPR();\n";

	//return option...part 2
	if(m_Fntype!="void")
	{
		StRunSp += "\treturn ReturnObj;\n\n";
	}
	//==================

	return StRunSp;
}



bool CL_GenSwFn::CheckImageStrngs(std::string inputstr, std::string * _patterns,
								  int _patternsNbr
								  )
{
	size_t Start = 0;
	bool Loop = true;

	for(int i =0; i < _patternsNbr; i++)
	{
		Start = inputstr.find(_patterns[i]);
		if (Start!=std::string::npos)
			return true;
		/*DstBuffer += tempSource.substr(0,Start);
		tempSource =
		tempSource.substr(Start+_patterns[i].size(),tempSource.size());
		}
		DstBuffer += tempSource;
		tempSource = DstBuffer;*/
	}
	return false;//..??
}



FctAgrs ParseArgument(std::string &arg)
{
	FctAgrs CurArg;
	string LocalArgStr = arg;
	CurArg.m_fullStr = arg;

	size_t Start, Start_sp;//, Start_st, Start_am;
	std::string subarg;

	//check default value:
	Start = LocalArgStr.find_first_of("=");
	if(Start==std::string::npos)
	{
		Start = LocalArgStr.find("CV_DEFAULT");
	}
	if(Start!=std::string::npos)
	{
		CurArg.m_defaultVal = LocalArgStr.substr(Start, LocalArgStr.size());
		LocalArgStr = LocalArgStr.substr(0, Start-1);
		if(CurArg.m_defaultVal[CurArg.m_defaultVal.size()-1]!=')')
			CurArg.m_defaultVal+=")";
	}
	//===============================

	SGE::StrTrimLR(LocalArgStr);
	SGE::StrTrimLastChar(LocalArgStr, ';');
	SGE::StrTrimLastChar(LocalArgStr, ')');
	SGE::StrTrimLR(LocalArgStr);


	//check for type, 
	//it is normally every thing on the left of * or &
	Start		= LocalArgStr.find_last_of("*");
	if(Start==std::string::npos)
	{
		Start		= LocalArgStr.find_last_of("&");
	}

	if(Start!=std::string::npos)
	{
		CurArg.m_type = LocalArgStr.substr(0, Start+1);
		CurArg.m_name = LocalArgStr.substr(Start+1, LocalArgStr.size());
	}
	else//we check for a space...
	{
		Start_sp		= LocalArgStr.find_last_of(" ");
		if(Start_sp!=std::string::npos)
		{
			CurArg.m_type = LocalArgStr.substr(0, Start_sp);
			CurArg.m_name = LocalArgStr.substr(Start_sp+1, LocalArgStr.size());
		}
	}

	//trim all space that might be there...
	SGE::StrTrimLR(CurArg.m_type);
	SGE::StrTrimLR(CurArg.m_name);
	SGE::StrTrimLR(CurArg.m_defaultVal);

	//check const values
	Start = CurArg.m_type.find("const");
	if(Start!=std::string::npos)
	{
		CurArg.m_const = true;
		CurArg.m_type = CurArg.m_type.substr(Start+strlen("const"), CurArg.m_type.size());
	}
	else
		CurArg.m_const = false;

	//check that argument is not an array of type 'int x[4]'
	Start =CurArg.m_name.find_first_of("[");
	if(Start!=std::string::npos)
	{
		CurArg.m_name = CurArg.m_name.substr(0, Start);
		CurArg.m_type += " * ";
	}

	if(CurArg.m_name=="" || CurArg.m_type=="")
	{
		int a =0;
		a++;
	}
#if 0	
	Start			= min(Start_sp, Start_st);
	Start			= min(Start,Start_am);

	if (Start==Start_sp)
	{
		if (arg.substr(0,Start).find_last_of(" ")!=std::string::npos)
		{
			subarg			= arg.substr(0,Start);
			Start			= subarg.find_last_of(" ");
			if (Start!=std::string::npos)
				subarg			= subarg.substr(Start+1,arg.size());
		}
		else 
		{
			if (arg.find_last_of(";")!=std::string::npos)
				subarg			= arg.substr(Start+1,arg.substr(Start+1,arg.size()).size()-2);
			else 
				subarg			= arg.substr(Start+1,arg.size());
		}
	}
	/*else if (Start==Start_st)	//		= arg.find_last_of("*");
	{
	if (arg.substr(0,Start).find_last_of(" ")!=std::string::npos)
	{
	subarg			= arg.substr(0,Start);
	Start			= subarg.find_last_of(" ");
	if (Start!=std::string::npos)
	subarg			= subarg.substr(Start+1,arg.size());
	}
	}*/
	else
		subarg			= arg.substr(Start+1,arg.size());
#endif
	return CurArg;
}


}//namespace GCV
