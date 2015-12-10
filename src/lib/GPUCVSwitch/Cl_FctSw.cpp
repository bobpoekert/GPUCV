//CVG_LicenseBegin==============================================================
//
//	Copyright@ Institut TELECOM 2005
//		http://www.institut-telecom.fr/en_accueil.html
//	
//	This software is a GPU accelerated library for computer-vision. It 
//	supports an OPENCV-like extensible interface for easily porting OPENCV 
//	applications.
//
//
//	Contacts :
//				patrick.horain@it-sudparis.eu		
//				gpucv-developers@picoforge.int-evry.fr
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
*\brief Switch Manager to manage all switch operations
*\author Darshan
*/

#include "StdAfx.h"
#include <GPUCVSwitch/Cl_FctSw.h>
#include <GPUCVSwitch/Cl_FctSw_Mngr.h>
#include <GPUCVSwitch/macro.h>
#include <GPUCV/misc.h>
#include <SugoiTracer/TracerRecord.h>
#include <SugoiTools/xml.h>
#include <GPUCV/cxtypesg.h>
using namespace std;

namespace GCV{

//SGE::CL_TEMPLATE_OBJECT_MANAGER<CL_FctSw, std::string> *
//CL_FctSw::m_GlobFctSwMngr = NULL;//init static obj
//==================================================================================================
CL_FctSw::CL_FctSw()
:SGE::CL_XML_BASE_OBJ<std::string>("","fctSw")
//,m_switchFct(NULL)
//,m_switchNbr(0)
,m_ForcedImpl(NULL)//auto
,m_LastCalledImpl(NULL)
,m_ImagesImpl(NULL)
,m_ProfileMode(false)
,m_InArray(0)
,m_OutArray(0)
,m_InArrsize(0)
,m_OutArrsize(0)
,m_CurTracerRecord(NULL)
{
};
//==================================================================================================
CL_FctSw::CL_FctSw(const std::string & _ID)
:SGE::CL_XML_BASE_OBJ<std::string>(_ID,"fctSw")
//,m_switchFct(NULL)
//,m_switchNbr(0)
,m_ForcedImpl(NULL)//auto
,m_LastCalledImpl(NULL)
,m_ImagesImpl(NULL)
,m_ProfileMode(false)
,m_InArray(0)
,m_OutArray(0)
,m_InArrsize(0)
,m_OutArrsize(0)
,m_CurTracerRecord(NULL)
{
};
//==================================================================================================
CL_FctSw::~CL_FctSw()
{
#if 1//OPTIMIZED_SYNC_ALL
	if(m_InArray!=NULL)
		delete []m_InArray;
	if(m_OutArray!=NULL)
		delete []m_OutArray;
#endif
};

//==================================================================================================
#if 0
void CL_FctSw::SetSwitchOperators(switchFctStruct * _sw, int nbr)
{
	m_switchFct = _sw;
	m_switchNbr = nbr;
	std::string ClassImplName;
	for(int i = 0; i < m_switchNbr /*(sizeof(_sw)/sizeof(switchFctStruct))*/; i++)
	{
		//<!get ptr to class Impl
		ClassImplName = GetStrImplemtation(m_switchFct[i].m_ImplID);
		SG_TRC::CL_CLASS_TRACER<SG_TRC::CL_TimerVal>* ImplClass = MainTagTracer->ClassManager->Add(ClassImplName);//Add(---) here firstly finds the class "opencv","glsl".. and adds the the class if it is not present
		//<!get ptr to fct tracer
		if(m_switchFct[i].m_FctTracer==NULL && ImplClass)
			m_switchFct[i].m_FctTracer = ImplClass->AddFunct(GetID());
	}	
}
#endif
//==================================================================================================
void CL_FctSw::AddImplementation(switchFctStruct * _newImpl)
{
	SG_Assert(_newImpl, "CL_FctSw::AddImplementation()=> obj is empty");
	m_switchFct.push_back(_newImpl);
}
//==================================================================================================
void CL_FctSw::SetInputArray(CvArr * _Img[], int innbr)
{
	if(innbr>0 && _Img!=NULL)
	{
		int iLocalImageImplID =0;
		if(m_InArray==NULL)
			m_InArray = new IMAGE_TYPE*[innbr];

		for(int i=0; i < innbr; i++)
		{
			if(_Img[i]==NULL)
				m_InArray[i] = NULL;
			else
		#if OPTIMIZED_SYNC_ALL
				m_InArray[i] = (IMAGE_TYPE*)GPUCV_GET_TEX(_Img[i]);
		#else
				//m_InArray = _Img;
				m_InArray[i] = (IMAGE_TYPE*)_Img[i];
		#endif
#if 0
			//ask if there is any restrictions on the image processing mode.
		iLocalImageImplID = (IMAGE_TYPE*)GPUCV_GET_TEX(_Img[i])->GetSwitchLockImplementationID();
		if(i==0)
			ImagesImplID = iLocalImageImplID != 
#endif
		}	
		m_InArrsize = innbr;
	}
}
//==================================================================================================
void CL_FctSw::SetOutputArray(CvArr * _Img[], int outnbr)
{
	if(outnbr>0 && _Img!=NULL)
	{
	
		if(m_OutArray==NULL)
			m_OutArray = new IMAGE_TYPE*[outnbr];
		for(int i =0; i < outnbr; i ++)
		{
			if(_Img[i]==NULL)
				m_OutArray[i] = NULL;
			else
#if OPTIMIZED_SYNC_ALL
				m_OutArray[i] = (IMAGE_TYPE*)GPUCV_GET_TEX(_Img[i]);
#else
				m_OutArray[i] = (IMAGE_TYPE*)_Img[i];
				//m_OutArray = _Img;
#endif

		}
		
		m_OutArrsize = outnbr;
	}
}
//==================================================================================================
void CL_FctSw::SynchronizeAll(IMAGE_TYPE_ARRAY _array, int Nbr, bool output)
{
	CvgArr *tempImg = NULL;
	for(int i=0; i< Nbr; i++)
	{
		if(_array[i]!=NULL)
		{
#if OPTIMIZED_SYNC_ALL
				tempImg = _array[i];
				tempImg->SetOption(DataContainer::DEST_IMG, output);
				if(output)
				{
					tempImg->PushOptions();
					tempImg->SetLocation<DataDsc_CPU>(false);
					tempImg->SetDataFlag<DataDsc_CPU>(true, true);
				}
				else
				{
					tempImg->PushSetOptions(DataContainer::UBIQUITY, !output);
					tempImg->SetLocation<DataDsc_CPU>(true);
				}
				tempImg->PopOptions();
#else
				cvgSetOptions(_array[i], DataContainer::DEST_IMG, output);

				//cvgPushSetOptions(m_InArray[i], DataContainer::CPU_RETURN, false);
				if(output)
				{
					cvgPushSetOptions(_array[i],0,true);
					cvgSetLocation<DataDsc_CPU>(_array[i], false);
					cvgSetDataFlag<DataDsc_CPU>(_array[i], true, true);//data will be here later...
				}
				else
				{
					cvgPushSetOptions(_array[i], DataContainer::UBIQUITY, true);
					cvgSynchronize(_array[i]);	
				}
				//cvgSetDataFlag<DataDsc_IplImage>(m_InArray[i], true, false);
				cvgPopOptions(_array[i]);
#endif
			}
		}
}
#if 0
{
	CvgArr *tempImg = NULL;
	for(int i=0; i< m_InArrsize; i++)
	{
		if(m_InArray[i]!=NULL)
		{
#if OPTIMIZED_SYNC_ALL
			tempImg = m_InArray[i];
			tempImg->SetOption(DataContainer::DEST_IMG, false);
			tempImg->PushSetOptions(DataContainer::UBIQUITY, true);
			tempImg->SetLocation<DataDsc_CPU>(true);
			tempImg->PopOptions();
#else
			cvgSetOptions(m_InArray[i], DataContainer::DEST_IMG, false);

			//cvgPushSetOptions(m_InArray[i], DataContainer::CPU_RETURN, false);
			cvgPushSetOptions(m_InArray[i], DataContainer::UBIQUITY, true);
			cvgSynchronize(m_InArray[i]);	
			//cvgSetDataFlag<DataDsc_IplImage>(m_InArray[i], true, false);
			cvgPopOptions(m_InArray[i]);
#endif
		}
	}

	for(int i=0; i< m_OutArrsize; i++)
	{
		if(m_OutArray[i]!=NULL)
		{
#if OPTIMIZED_SYNC_ALL
			tempImg = m_OutArray[i];
			tempImg->SetOption(DataContainer::DEST_IMG, true);
			tempImg->PushOptions();
			tempImg->SetLocation<DataDsc_CPU>(false);
			tempImg->SetDataFlag<DataDsc_CPU>(true, true);
			tempImg->PopOptions();
#else
			cvgSetOptions(m_OutArray[i], DataContainer::DEST_IMG, true);
			//cvgPushSetOptions(m_OutArray[i], DataContainer::CPU_RETURN, false);
			//cvgSetOptions(m_OutArray[i], DataContainer::UBIQUITY, true);
			cvgPushSetOptions(m_OutArray[i],0,true);
			cvgSetLocation<DataDsc_CPU>(m_OutArray[i], false);
			cvgSetDataFlag<DataDsc_CPU>(m_OutArray[i], true, true);//data will be here later...
			cvgPopOptions(m_OutArray[i]);
#endif
		}
	}
}
#endif
//==================================================================================================
switchFctStruct * CL_FctSw::GetImplSwitchByID(const ImplementationDescriptor* _pImpl)
{
	if(_pImpl==NULL)
	return NULL;

	for(unsigned int i = 0; i < m_switchFct.size(); i++)
	{
		if(m_switchFct[i]->m_Implementation == _pImpl)
			return m_switchFct[i];
	}
	return NULL;
}
//==================================================================================================
switchFctStruct *CL_FctSw::GetImplSwitch(size_t i)
{
	SG_Assert(m_switchFct.size()>i, "Index is out of bound");
	return m_switchFct[i];
}
//==================================================================================================
size_t	 CL_FctSw::GetImplSwitchNbr(void)const
{
	return m_switchFct.size();
}
//==================================================================================================
int CL_FctSw::GetTransferTime(IMAGE_TYPE* _Img, const ImplementationDescriptor * _pImplID)
{
	long int TransferTime =0;
	char BestSrcID = -1;
	if(_pImplID==NULL)
		return 0;
	switch(_pImplID->m_baseImplID)
	{
#if OPTIMIZED_SYNC_ALL
		case GPUCV_IMPL_OPENCV:

			TransferTime = _Img->SetLocationEstimation<DataDsc_CPU>(BestSrcID, true);break;
		case GPUCV_IMPL_GLSL:
			TransferTime = _Img->SetLocationEstimation<DataDsc_GLTex>(BestSrcID, true);	break;
#else
		case GPUCV_IMPL_OPENCV:
			TransferTime = cvgSetLocationEstimation<DataDsc_CPU>(_Img, BestSrcID, true);break;
		case GPUCV_IMPL_GLSL:
			TransferTime = cvgSetLocationEstimation<DataDsc_GLTex>(_Img, BestSrcID, true);	break;
		case GPUCV_IMPL_OTHER:
			TransferTime = cvgSetLocationEstimation<DataDsc_CPU>(_Img, BestSrcID, true);break;
#endif

			//	case GPUCV_IMPL_CUDA:
			//		TransferTime = cvgSetLocationEstimation<DataDsc_CUDA_BUFFER>(_Img, true);
			//		break;
		default:
			break;
	}
	return TransferTime;
}
//==================================================================================================
int CL_FctSw::GetTotalTime(SG_TRC::CL_TRACER_RECORD<SG_TRC::CL_TimerVal> * _TracerPtr,  const ImplementationDescriptor * _pImplID)
{
	long int TotalTime = _TracerPtr->GetStats().GetAvgTime()->GetValue();//StdDevTime->GetValue();
	long int TotalTransferTime =0;
	for(int i=0; i< m_InArrsize; i++)
	{
		if (TotalTransferTime<=-1) return TotalTransferTime;
		else 
		{	
			if (m_InArray[i]!= NULL)
				TotalTransferTime += GetTransferTime(m_InArray[i], _pImplID);
			else continue;
		}
	}

	if (TotalTransferTime<=-1) return TotalTime;
	else
	{
		GPUCV_SWITCH_LOG("TotalTransferTime Time :"<<TotalTransferTime);
		TotalTime += TotalTransferTime;
		return TotalTime;
	}
}
//==================================================================================================
SG_TRC::CL_FUNCT_TRC<SG_TRC::CL_TimerVal>* GetFctTracer(std::string _Fctname)
{
	GET_FCTMNGR_MUTEX(MutexFctMngr);
	return MutexFctMngr->Find(_Fctname);
}
//==================================================================================================
SG_TRC::CL_TRACE_BASE_PARAMS * CL_FctSw::GetParamsObj(CvArr* _Img)
{
	SG_TRC::CL_TRACE_BASE_PARAMS * _ParamsPtr = NULL;

	//we need it all the time...
	//if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_PROFILING))
	{
		_ParamsPtr = new SG_TRC::CL_TRACE_BASE_PARAMS();
		if(_Img)
		{
			_ParamsPtr->AddParam("width", GetWidth(_Img));\
			_ParamsPtr->AddParam("height", GetHeight(_Img));\
			_ParamsPtr->AddParam("channels", GetnChannels(_Img));\
			_ParamsPtr->AddParam("format", GetStrCVPixelType(GetCVDepth(_Img)));\
		}
	}
	return _ParamsPtr;
}
//==================================================================================================

//==================================================================================================
SG_TRC::CL_TRACER_RECORD<SG_TRC::CL_TimerVal>* 
CL_FctSw::GetFctRecord(SG_TRC::CL_FUNCT_TRC<SG_TRC::CL_TimerVal>* _FctPtr, CvArr* _Img, SG_TRC::CL_TRACE_BASE_PARAMS * _ParamsPtr)
{	
	SG_Assert(_FctPtr, "No function ptr");
	//static CL_FctSw *	LocalFctSw1	=	NULL;\
	
	//SG_TRC::CL_TRACE_BASE_PARAMS * _ParamsPtr = _ParamsPtr;

	if(_FctPtr->TracerRecorded)
		return _FctPtr->TracerRecorded->Find(_ParamsPtr->GetParamsValue()); 
	else
		return NULL;
}
//==================================================================================================
SG_TRC::ExecState CL_FctSw::ControlImplementation( switchFctStruct* _FctSwitchImpl, CvArr* _Img, SG_TRC::CL_TRACE_BASE_PARAMS * _ParamsPtr, SG_TRC::CL_TRACER_RECORD<SG_TRC::CL_TimerVal> ** _pTracerRecord)
{
	SG_Assert(_FctSwitchImpl,"Empty implementation pointer");
	SG_Assert(_ParamsPtr,"Empty parameter pointer");

	if(!GetHardProfile()->IsCompatible(_FctSwitchImpl->m_HrdPrf))
	{//hardware not compatible
		return SG_TRC::EXEC_FAILED_COMPAT_ISSUE;
	}
#if 1
	else
	{//check last execution state
		//update parameter for the request
		SG_TRC::CL_TRACE_BASE_PARAMS LocalParamSet = *_ParamsPtr;
		LocalParamSet.AddParamAtStart("type", _FctSwitchImpl->m_Implementation->m_strImplName);

		//request benchmarking results
		*_pTracerRecord = GetFctRecord(_FctSwitchImpl->m_FctTracer, _Img, &LocalParamSet);
//		GPUCV_SWITCH_LOG("Parameter set: "<< LocalParamSet.GetParamsValue());
		
		if(!*_pTracerRecord)
		{//!<Ask to flush and process the record list.
			SG_TRC::CL_TEMP_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::GetProfilerInstance().ParseEventQueue(SG_TRC::CL_TEMP_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::GetEventListInstance());
			SG_TRC::CL_TEMP_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::GetProfilerInstance().Process();
			_FctSwitchImpl->m_FctTracer->Process();
			*_pTracerRecord = GetFctRecord(_FctSwitchImpl->m_FctTracer, _Img, &LocalParamSet);
		}
		
		if(*_pTracerRecord)
		{//return known state
			GPUCV_SWITCH_LOG("Last execution state: "<<SGE::ToCharStr((*_pTracerRecord)->GetExecutionState()));
			GPUCV_SWITCH_LOG("Record in base: " << (*_pTracerRecord)->GetStats().GetTotalRecords());
			return (*_pTracerRecord)->GetExecutionState();
		}
	}
#endif

	return SG_TRC::EXEC_UNKNOWN;
}
/*
if(m_CurTracerRecord && m_CurTracerRecord->GetExecutionState() < SG_TRC::EXEC_UNKNOWN)
		{
			//this funcion generated exception in the past for this set of parameters
			//we won't call it again
			//set ForceImplID to auto:	
			GPUCV_SWITCH_LOG("Impl '"<<GetStrImplemtation(_FctSwitchImpl->m_Implementation)<< "', exception occured in the past, reseting forced implementations to auto.");
			SetForcedImpl(NULL);
			TempFctSt=NULL;//reset current Impl selection
			GCV_OPER_COMPAT_ASSERT(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_PLUG_AUTO_RESOLVE_ERROR)==true,
				"Current implementation is known to fail, flag GpuCVSettings::GPUCV_SETTINGS_PLUG_AUTO_RESOLVE_ERROR is set to false so the switch mechanism will not try to find another valid implementation.");
		}
		*/
//==================================================================================================
switchFctStruct * CL_FctSw::GetImplIDToCall(CvArr* _Img, SG_TRC::CL_TRACE_BASE_PARAMS * _ParamsPtr)
{
//	SG_TRC::CL_TRACE_BASE_PARAMS LocalParamSet = *_ParamsPtr;
//	GPUCV_SWITCH_LOG("function call -> GetImplIDToCall()");
	GPUCV_SWITCH_LOG("Local switch flag: "<< GetStrImplemtation(m_ForcedImpl));
	GPUCV_SWITCH_LOG("Global switch flag: "<< GetStrImplemtationID(CL_FctSw_Mngr::GetSingleton()->GetGlobalForcedImplemID()));
	GPUCV_SWITCH_LOG("Nbr of implementation :" << m_switchFct.size());
	
	//init local variables
	switchFctStruct*	TempFctSt = NULL;
	enum BaseImplementation GlobalImplementationFilter = GPUCV_IMPL_AUTO;
	unsigned int uiImplNbr = m_switchFct.size();
	switchFctStruct * funcToCall = NULL;
	SG_TRC::ExecState currentImplExecState =  SG_TRC::EXEC_UNKNOWN;
	

	 if (uiImplNbr > 1)
	 {//most cases
		if(m_ForcedImpl==NULL)
		{//we use global implementation flag
			GlobalImplementationFilter = CL_FctSw_Mngr::GetSingleton()->GetGlobalForcedImplemID();
			TempFctSt = NULL;//will be find later in the for loop
		}
		else
		{//we have a local implementation forced
			TempFctSt = GetImplSwitchByID(m_ForcedImpl);
		}
	 }
	 else if (uiImplNbr == 1)
	 {
		TempFctSt = m_switchFct[0];
	 }
	 else//no implementation found
	 {
		GPUCV_ERROR("CL_FctSw::GetImplIDToCall()=> No implemantation found for function " << GetIDStr());
		GPUCV_ERROR("CL_FctSw::GetImplIDToCall()=> CRITICAL, one fonction is missing, could not continue!");
		exit(-1);	
	 }


	 if(TempFctSt)
	 {
		//Check that chosen implementation is valid, Hardware profile + last execution state
		currentImplExecState = ControlImplementation(TempFctSt, _Img, _ParamsPtr,&m_CurTracerRecord);
	 
		if(currentImplExecState < SG_TRC::EXEC_UNKNOWN)
		{
			GPUCV_SWITCH_LOG("Impl '"<<GetStrImplemtation(TempFctSt->m_Implementation)<< "', is known not to be supported on this hardware or exception occured in the past, reseting forced implementations to auto.");
			TempFctSt=NULL;
			SetForcedImpl(NULL);
			//we will scan all available implementations
		}
	 }
	
	if(TempFctSt!=NULL)
	{
		funcToCall=TempFctSt;
	}
	else//we do not have any implementation to call yet, try to find one.
	{
		long double min=999999999;int minID=-1;long int TotalTime=0;long int OpenCVTime=0;
		SG_TRC::CL_TRACER_RECORD<SG_TRC::CL_TimerVal> * pLocalTracerRecord=NULL;
		GPUCV_SWITCH_LOG("Parameter set: "<< _ParamsPtr->GetParamsValue());
		for(unsigned int i = 0; i < m_switchFct.size(); i++)
		{
			TempFctSt = m_switchFct[i];
			GPUCV_SWITCH_LOG("Implementation Class : "<< GetStrImplemtation(TempFctSt->m_Implementation));
			SGE::LoggerAutoIndent indent;//manage indentation
			
			if(GlobalImplementationFilter != GPUCV_IMPL_AUTO)
			{//check that current function match the switch mode
			 //we will scan only the requested implementation (all GLSL, all CUDA, etc...)
				if(TempFctSt->m_Implementation->m_baseImplID != GlobalImplementationFilter)
					continue;
			}

			 //Check that chosen implementation is valid, Hardware profile + last execution state
			currentImplExecState = ControlImplementation(TempFctSt, _Img, _ParamsPtr,&pLocalTracerRecord);
			 
			if(currentImplExecState < SG_TRC::EXEC_UNKNOWN)
			{//not compatible
				GPUCV_SWITCH_LOG("Last execution state: "<< SGE::ToCharStr(currentImplExecState));
				continue;
			}
			else if ((pLocalTracerRecord==NULL || currentImplExecState == SG_TRC::EXEC_UNKNOWN)
				|| (pLocalTracerRecord && 
					(pLocalTracerRecord->GetStats().GetTotalRecords()>CL_FctSw_Mngr::GetSingleton()->GetMinBenchNbr())
					)
				)
			{//ask for more benchmarks
				//return the pointer to the record 
				GPUCV_SWITCH_LOG("Not enough records...");
				if(pLocalTracerRecord)
					GPUCV_SWITCH_LOG("Record in base: " << pLocalTracerRecord->GetStats().GetTotalRecords() << ", required: " << CL_FctSw_Mngr::GetSingleton()->GetMinBenchNbr());
				SetProfileMode(true);
				funcToCall=TempFctSt;	
				m_CurTracerRecord = pLocalTracerRecord;
				break;
			}
			else
			{//we have enought benchmarks records
				GPUCV_SWITCH_LOG("Record in base: " << pLocalTracerRecord->GetStats().GetTotalRecords() << ", required: " << CL_FctSw_Mngr::GetSingleton()->GetMinBenchNbr());
	
				//get corresponding timing,
				GPUCV_SWITCH_LOG("Processing Time : "<<pLocalTracerRecord->GetStats().GetAvgTime()->GetValue());		
				TotalTime=GetTotalTime(pLocalTracerRecord, TempFctSt->m_Implementation);
				GPUCV_SWITCH_LOG("Total Time(Processing + Transfer) : '"<<GetStrImplemtation(TempFctSt->m_Implementation)<< "':" << TotalTime);

				if (TotalTime<=-1)
				{
					GPUCV_SWITCH_LOG("No transfer time records...");
				}

				if(min>TotalTime)
				{
					min = TotalTime;//pLocalTracerRecord->AvgTime->GetValue();
					minID=i;
					funcToCall=TempFctSt;
					//affect selected tracer record, corresponding to the impl that will be called.
					m_CurTracerRecord = pLocalTracerRecord;

					//GPUCV_SWITCH_LOG("Min Total Time impl function number of records '"<<GetStrImplemtation(m_switchFct[minID].m_ImplID)<< "':" << pLocalTracerRecord->GetTotalRecords());
					GPUCV_SWITCH_LOG("Min Total Time ImplID '"<<GetStrImplemtation(m_switchFct[minID]->m_Implementation)<< "':" << min);
					GPUCV_SWITCH_LOG("'"<<GetStrImplemtation(m_switchFct[minID]->m_Implementation)<< "' become best choice.");
				}
			}

			//for calculating gain in total time
			if (funcToCall->m_Implementation->m_baseImplID == GPUCV_IMPL_OPENCV)
			{
				OpenCVTime = TotalTime;
			}
		}//for loop end!!


		//GPUCV_SWITCH_LOG("Average Time ImplID "<<GetStrImplemtation(m_switchFct[minID].m_ImplID)<<min);
		if(!GetProfileMode())
		{
			funcToCall->m_counter++;
			if (funcToCall->m_Implementation->m_baseImplID !=GPUCV_IMPL_OPENCV)
			{//calculating gain in total time
				funcToCall->m_AmntOfTimeSaved += OpenCVTime - TotalTime;
			}
		}
	}// first else end 

	if (funcToCall->m_UseGpu==false)
	{//make sure to send the source image data back to RAM
		SynchronizeAll(m_InArray, m_InArrsize, false);
		SynchronizeAll(m_OutArray, m_OutArrsize, true);
	}

	//function not called yet but will be called soon after
	SetLastCalledImpl(funcToCall->m_Implementation);

	return funcToCall;
}	
//==================================================================================================
/*virtual */
const std::string CL_FctSw::GetValStr()
{	
	std::string strob;
	switchFctStruct*	TempFctSt;
	for(unsigned int i = 0; i < m_switchFct.size(); i++)
	{
		TempFctSt=m_switchFct[i];
		//implementation name
		strob+="\n\t";
		strob+=GetStrImplemtation(TempFctSt->m_Implementation) ;
		strob+="(";
		strob+= SGE::ToCharStr(TempFctSt->m_FctTracer->GetStats().GetTotalRecords());
		strob+=")";
		//all times
		strob+="	:		";
		strob+= SGE::ToCharStr(TempFctSt->m_counter);
		strob+="	|      ";
		//time saved...
		strob+= SGE::ToCharStr(TempFctSt->m_AmntOfTimeSaved);

		SGE::CL_XML_MNGR<SG_TRC::CL_TRACER_RECORD<SG_TRC::CL_TimerVal> , std::string>::iterator itFctParam;
		SG_TRC::CL_TRACER_RECORD<SG_TRC::CL_TimerVal> * pTrcRcrd;
		for(itFctParam = TempFctSt->m_FctTracer->TracerRecorded->GetFirstIter();
			itFctParam != TempFctSt->m_FctTracer->TracerRecorded->GetLastIter();
			itFctParam++
			)
		{
			pTrcRcrd=(*itFctParam).second;

			strob+="\n\t\t";	
			strob+=pTrcRcrd->GetParams()->GetParamsValue();
			//strob+=pTrcRcrd->TotalRecords;
			//strob+="\t";
			strob+="\t(min)";
			strob+=pTrcRcrd->GetStats().GetMinTime()->GetStr();
			strob+="\t(avg)";
			strob+=pTrcRcrd->GetStats().GetAvgTime()->GetStr();
			strob+="\t(max)";
			strob+=pTrcRcrd->GetStats().GetMaxTime()->GetStr();
			strob+="\t(total)";
			strob+=pTrcRcrd->GetStats().GetTotalTime()->GetStr();
		}
		//nbr of bench()
	}
	return strob;
} 
//======================================================
TiXmlElement* CL_FctSw::XMLLoad(TiXmlElement* _XML_Root)
{
	SG_Assert(_XML_Root, "No XML root");
	std::string implName;
	SGE::XMLReadVal(_XML_Root,"ForcedImplID", implName);
	m_ForcedImpl = GetGpuCVSettings()->GetImplementation(implName);
	return _XML_Root;
}
//======================================================
TiXmlElement* CL_FctSw::XMLSave(TiXmlElement* _XML_Root)
{
	SG_Assert(_XML_Root, "No XML root");
	SGE::XMLWriteVal(_XML_Root,"ForcedImplID", m_ForcedImpl->m_strImplName);
	return _XML_Root;
}
//======================================================

}//namespace GCV
