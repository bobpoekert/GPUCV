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

/* 
 * \brief Run time switch to use the processor that gives faster results in image processing. 
 * \author C S Priyadarshan

 Dynamically switch enclosed inside each operator calls faster implemention among OpenCV, GpuCV-GLSL and GpuCV-CUDA to run the operator.
*/
#ifndef CL_FCT_SWITCH_H_
#define CL_FCT_SWITCH_H_

#include <includecv.h>
#include <GPUCVSwitch/config.h>

#include <SugoiTracer/appli.h>
#include <SugoiTracer/timer.h>
#include <SugoiTracer/function.h>
#define _GPUCV_PROFILE_DARSHAN 1

namespace GCV{

/**
	*\brief get a pointer to the function
	*\param _Fctname => string function name.
	*\return : pointer to the function.
	*\author : Cindula Saipriyadarshan
	*/
_GPUCV_SWITCH_EXPORT SG_TRC::CL_FUNCT_TRC<SG_TRC::CL_TimerVal>* GetFctTracer(std::string _Fctname);


//#define MainTagTracer SG_TRC::CreateMainTracer<SG_TRC::CL_TimerVal>()//CL_Profiler::GetTimeTracer()
#define MainTagTracer CL_Profiler::GetTimeTracer()


/**
	*\brief get a pointer to the record of a specified function
	*\param _FctPtr => function pointer.
	*\param _Img => Image.
	*\return : pointer to the function record.
	*\author : C S Priyadarshan.
	*/

_GPUCV_SWITCH_EXPORT SG_TRC::CL_TRACER_RECORD<SG_TRC::CL_TimerVal>* GetFctRecord(SG_TRC::CL_FUNCT_TRC<SG_TRC::CL_TimerVal>* _FctPtr, CvArr* _Img);




/** Store informations about one implementation.
*/
struct switchFctStruct
{
	const ImplementationDescriptor *m_Implementation;	//!< Implementation ID of the current implementation.
	const LibraryDescriptor*		m_Library;			//!< Pointer to the library descriptor the function belongs to.
	//!< pointer to current implementation function.
#ifdef _WINDOWS	
	PROC			m_ImplPtr;								
#else
	void*			m_ImplPtr;
#endif
	SG_TRC::CL_FUNCT_TRC<SG_TRC::CL_TimerVal>*	m_FctTracer;//!< Pointer the benchmarks corresponding to current implementation.
	bool			m_UseGpu;								//!< Current implementation use GPU, this flag affect the synchronisation.
	enum			GenericGPU::HardwareProfile m_HrdPrf;	//!< Hardware profile required to run the function
	int				m_counter;								//!< Count the number of time the function has been called
	int				m_AmntOfTimeSaved;						//!< Sum the amount of time saved compared to OpenCV implementation.
};


#define OPTIMIZED_SYNC_ALL 0
	
/** CL_FctSw is a static object created into each function wrapper. It stores benchmarks informations and available implementation use.
*/
class _GPUCV_SWITCH_EXPORT CL_FctSw
	: public SGE::CL_XML_BASE_OBJ<std::string>
{
#if OPTIMIZED_SYNC_ALL 
	typedef CvgArr				IMAGE_TYPE;
	typedef IMAGE_TYPE**		IMAGE_TYPE_ARRAY;
#else
	typedef CvArr				IMAGE_TYPE;
	typedef IMAGE_TYPE**		IMAGE_TYPE_ARRAY;
#endif

//members
	//! enum value of the Implementation we want to switch to and run.
	_DECLARE_MEMBER(const ImplementationDescriptor*, ForcedImpl);
	//! enum value of the Implementation we have runned, maybe the m_ForcedImplID was not available.
	_DECLARE_MEMBER(const ImplementationDescriptor*, LastCalledImpl);
	//! enum value of the Implementation the input/output images are allowed to run, see DataContainer::m_SwitchLockImplementationID. Is calculated on each run.
	_DECLARE_MEMBER(const ImplementationDescriptor*, ImagesImpl);


	//! boolean value flag used to get the mode profile/processing of operator.
	_DECLARE_MEMBER(bool,								ProfileMode);
	//! input array pointer
	_DECLARE_MEMBER_BYREF__NOSET(IMAGE_TYPE_ARRAY,		InArray);
	//! output array pointer
	_DECLARE_MEMBER_BYREF__NOSET(IMAGE_TYPE_ARRAY,		OutArray);	
	//! Number of elements in the SwitchImplementations array.
	_DECLARE_MEMBER(int,								InArrsize);	
	//! Number of elements in the SwitchImplementations array.	
	_DECLARE_MEMBER(int,								OutArrsize);	
	//! Pointer to the current/last SugoiTracer record for the current/last set of parameters. It is mainly used to check last exectution state(exception/Success).
	_DECLARE_MEMBER(SG_TRC::CL_TRACER_RECORD<SG_TRC::CL_TimerVal> *, CurTracerRecord); 

protected:
	std::vector<switchFctStruct *>			m_switchFct;	//!< pointer to all the implementations.
	
//functions
public:
	CL_FctSw();
	CL_FctSw(const std::string & _ID);
	~CL_FctSw();
/*!
	*\brief initializes the implementation class and traces function pointer.
	*\param _sw => pointer to the SwitchFctStruct (for passing SwitchImplementations).
	*\param nbr => number of array elements in the SwitchImplementations[].
	*\return ImplID.
	*/
	//void SetSwitchOperators(switchFctStruct * _sw, int nbr);

	void AddImplementation(switchFctStruct*_newImpl);


	/**
	*\brief get a pointer to the switchFctStruct.
	*\return : pointer to the "switchFctStruct" data structure.
	*/
	//switchFctStruct * GetSwitchOperators();

	/*!
	*\brief sets the input array of source images to the members of class "CL_FctSw".
	*\param _Img[] => input array of source images.
	*\param innbr => number of array elements in the _Img[].

	*/
	void SetInputArray(CvArr * _Img[], int innbr);

	/*!
	*\brief sets the output array of destination images to the members of class "CL_FctSw".
	*\param _Img[] => input array of destination images.
	*\param outnbr => number of array elements in the _Img[].
	*\author : C S Priyadarshan.
	*/

	void SetOutputArray(CvArr * _Img[], int outnbr);

	/*!
	*\brief synchronizes all the source and destination images to be found in cpu.
	*\note: sets all the required flags like SetLocation, SetOption automatically.
	*\author : C S Priyadarshan.
	*/

	void SynchronizeAll(IMAGE_TYPE_ARRAY _array, int Nbr, bool output=false);

	/*!
	*\brief to get the pointer to the implementation switch.
	*\param _ID => pointer to Implementation descriptor.
	*\return pointer to the implementation of data sructure.
	*\author : C S Priyadarshan.
	*/
	switchFctStruct *GetImplSwitchByID(const ImplementationDescriptor *_ID);
	switchFctStruct *GetImplSwitch(size_t i);
	size_t	 GetImplSwitchNbr(void)const;


	/*!
	*\brief Return a CL_TRACE_BASE_PARAMS * object containing imformations about the given image.
	*\param img => image given as an input to the function
	*\note The params are common to all the implementations.
	*\return a new object pointer or NULL if no input image is given or global profiling flag GpuCVSettings::GPUCV_SETTINGS_PROFILING is disabled.
	*/	
	SG_TRC::CL_TRACE_BASE_PARAMS * GetParamsObj(CvArr* _Img);

	/*!
	*\brief Find the implementation processor taking least avg time and return its ImplD
	*\param _Img => image given as an input to the function
	*\param _ParamsPtr => params object pointer
	*\note Having an ImplID forced return directly its value.
	*\return ImplID
	*\author : C S Priyadarshan.
	*/	

	switchFctStruct * GetImplIDToCall(CvArr* _Img, SG_TRC::CL_TRACE_BASE_PARAMS * _ParamsPtr);


	/*!
	*\brief Calculates the total time = transfer time + processing time.
	*\param _Img => image given as an input to the function
	*\param _TracerPtr => Function Record pointer
	*\param m_ImplID => Implimentation ID
	*\note Having transfer time -1 returns transfer time.
	*\return TotalTime
	*\author : Cindula Saipriyadarshan
	*/
	int GetTotalTime(SG_TRC::CL_TRACER_RECORD<SG_TRC::CL_TimerVal> * _TracerPtr, const ImplementationDescriptor* _pImpl);
									
	/*!
	*\brief Calculates the transfer time.
	*\param _Img => image given as an input to the function
	*\param m_ImplID => Implimentation ID
	*\note Having transfer time -1 returns transfer time.
	*\return TotalTime.
	*\author : C S Priyadarshan.
	*/
	int GetTransferTime(IMAGE_TYPE* _Img, const ImplementationDescriptor* _pImpl);

	/*! 
	*	\brief Return a string describing the CvgArr object with folowing format : "Counter:%m_counter% | AmntOfTimeSaved:%m_AmntOfTimeSaved%".
	*	\return std::string : texture short description.
	*	\sa CL_FctSw::GetValStr().
	*/
	virtual const std::string GetValStr();


/*!
	*\brief Calculates the transfer time.
	*\param _FctPtr => pointer to a function in the function class.
	*\param _Img => Image (usually dst image).
	*\param _ParamsPtr => params object pointer.
	*\return pointer to the function record.
	*\author : C S Priyadarshan.
	*/

SG_TRC::CL_TRACER_RECORD<SG_TRC::CL_TimerVal>* GetFctRecord(SG_TRC::CL_FUNCT_TRC<SG_TRC::CL_TimerVal>* _FctPtr, CvArr* _Img, SG_TRC::CL_TRACE_BASE_PARAMS * _ParamsPtr);

	/** \brief Check that the corresponding implementation is compatible with current Hardware/OS and that it is known to run correclty.
	*/
	SG_TRC::ExecState ControlImplementation(switchFctStruct* _FctSwitchImpl, CvArr* _Img, SG_TRC::CL_TRACE_BASE_PARAMS * _ParamsPtr, SG_TRC::CL_TRACER_RECORD<SG_TRC::CL_TimerVal> ** _pTracerRecord);


	//XML
	virtual TiXmlElement* XMLLoad(TiXmlElement* _XML_Root);
	virtual TiXmlElement* XMLSave(TiXmlElement* _XML_Root);
};

}//namespace GCV

#endif
