#ifndef CL_GEN_SWITCH_FNS_H_
#define CL_GEN_SWITCH_FNS_H_



namespace GCV{
	/*!
	* \brief Matches the number of start and end brackets ("(",")") and returns true if equal. 
	*\code 
	CVAPI(void) cvgSwCopyMakeBorder (const CvArr* src, CvArr* dst, CvPoint offset, int bordertype, CvScalar value CV_DEFAULT(cvScalarAll(0)))

	{ 
		typedef void (*cvgSwType_cvCopyMakeBorder) (const CvArr*, CvArr*, CvPoint , int , CvScalar  ); 
		static switchFctStruct SwitchImplementations[]={
		{GPUCV_IMPL_OPENCV, (PROC)cvCopyMakeBorder, NULL, false, GenericGPU::HRD_PRF_0},
		{GPUCV_IMPL_GLSL, NULL, NULL, true, GenericGPU::HRD_PRF_2},
		{GPUCV_IMPL_CUDA, NULL, NULL, true, GenericGPU::HRD_PRF_CUDA}
		}; 
		 
		GPUCV_FUNCNAME('cvCopyMakeBorder');
		CVArr* SrcARR[] = { src};
		CVArr* DstARR[] = { dst};
		SWITCH_START_OPR( dst); 
		RUNOP(( src,  dst, offset, bordertype, value), cvgSwType_cvCopyMakeBorder); 
		SWITCH_STOP_OPR();
	}
	\endcode
	*\author : Cindula Saipriyadarshan 
	*/

enum ObjType
{
	Obj_Mngd_GpuCV	=	0x01
	,Obj_Input_Arr	=	0x02 | Obj_Mngd_GpuCV
	,Obj_Output_Arr	=	0x04 | Obj_Mngd_GpuCV
	,Obj_Mask_Arr	=	0x08 //! Arr is a mask
	,Obj_Param		=	0x10
};

struct FctAgrs
{
	bool m_const;
	std::string m_type;
	std::string m_name;
	std::string m_defaultVal;
	std::string m_fullStr;
	ObjType  m_objType;

};

FctAgrs ParseArgument(std::string &arg);

#define _GCV_SWITCH_FCT_PREFIX "cvgsw"

class _GPUCV_SWITCH_EXPORT CL_GenSwFn
	:public SGE::CL_BASE_OBJ<std::string>
{
	
	enum ENUM_SrcFileType
	{
		FILETYPE_H,
		FILETYPE_CPP
	};
	
//members
protected:
	std::string m_Fntype;		//!< to store the function type.
	std::string m_FnName;		//!< to store the function name.
	std::string m_FnShortName;	//!< to store the function short name, without the 'cv'.
	std::string m_FnTypeDef;	//!< to store the function typedef.
	std::string m_Argstr;		//!< to store the argument string.
	std::string m_SwFnType;		//!< to store the newly generated switch function type.
	vector <string> m_Dst_Arr;	//!< to store the destination array.
	vector <string> m_Src_Arr;	//!< to store the destination array.
	vector <FctAgrs> m_ArgsList;//!< to store the arguements' types.
	bool	m_requireSwitching;

	static SGE::CL_TEMPLATE_OBJECT_MANAGER<CL_GenSwFn, std::string> *
	m_FctObjMngr;				//!< to store the function pointer of each function.

	
	
//functions
public:

	CL_GenSwFn(const std::string &_name);//for compatibility with manager
	~CL_GenSwFn();


	/*!
	* \brief Matches the number of start and end brackets ("(",")") and returns true if equal. 
	*\param currArg => current arguement string.
	*\author : Cindula Saipriyadarshan 
	*/
	bool MatchBrackets(std::string &currArg, size_t &StartBracketPos, size_t &StopBracketPos);

	/*!
	* \brief Parsing the input function declaration into function type, function name, function arguements.
	* For example:
	\code
	CVAPI(void) cvCopyMakeBorder(const CvArr* src, CvArr* dst, CvPoint offset, int bordertype, CvScalar value CV_DEFAULT(cvScalarAll(0)));
	//into function type - 
	CVAPI(void), function name- cvCopyMakeBorder, and arguements types and arguement names into two vectors.
	\endcode
	*\param line => any OpenCV function declaration.
	*\author : Cindula Saipriyadarshan 
	*/
	bool ParseLine(const std::string &line);


/*!
	* \brief Adds objects to each line read so that line can be accessed using the corresponding object. 
	*\author : Cindula Saipriyadarshan 
	*/
	static void AddObjsToFns_CPP(std::string Infilename, std::string IN_FILEPATH, 
					   std::string Outfilename, std::string OUT_FILEPATH);

	static void AddObjsToFns_H(std::string Outfilename, std::string OUT_FILEPATH);
	static void AddObjsToFns_H_WRAPPER(std::string Outfilename, std::string OUT_FILEPATH);

	static void AppendFileToFnsFile(ofstream & fileswfns, std::string FileName);
		
/*!
* \brief Generates the Switch function declaration.
* For example-"void cvswSub(CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask)".
*\author : C S Priyadarshan.
*/
	std::string GenSwFnDeclaration(ENUM_SrcFileType _fileType);

/*!
* \brief Generates the required type definition of the Switch function.
* For example-"typedef void(*CVGType_Sub)(CvArr*,CvArr*,CvArr*,CvArr*);".
*\author : C S Priyadarshan.
*/
	std::string GenSwFnTypeDef();

	/*!
* \brief Generates the switchFctStruct SwitchImplementations[] array of the Switch function.
* For example:
\code 
	static switchFctStruct SwitchImplementations[]={
	{GPUCV_IMPL_OPENCV, (PROC)cvSub, NULL, false, GenericGPU::HRD_PRF_0}
	,{GPUCV_IMPL_GLSL, (PROC)cvgSub, NULL, true, GenericGPU::HRD_PRF_2}
	,{GPUCV_IMPL_CUDA, (PROC)cvgCudaAdd, NULL, true, GenericGPU::HRD_PRF_CUDA}"
	};
\endcode
*\author : C S Priyadarshan.
*/
	std::string GenSwchImpnsArr();

	/*!
* \brief Generates GPUCV_FUNCNAME("Function name");
* For example:
\code 
GPUCV_FUNCNAME("Erode");
\endcode
*\author : C S Priyadarshan.
*/
	std::string GenSwchFnName();

	/*!
* \brief Generates the required arrays CvArr* SrcARR[]={src1, src2, ....,mask, ...}; CvArr* DstARR[]={dst, ...};
* For example:
\code 
CvArr* SrcARR[]={src1, src2, mask};
CvArr* DstARR[]={dst};
\endcode
*\author : C S Priyadarshan.
*/
	std::string GenSwchInOutArr();
	void ParseArgsType();

	/*!
* \brief Generates the following 
* For example:
\code 
SWITCH_START_OPR(dst);
RUNOP((src1,src2,dst,mask), CVGType_Function);
SWITCH_STOP_OPR();
\endcode
*\author : C S Priyadarshan.
*/
	std::string GenStrtRunStop();
	std::string GenNoSwitch();

	/*!
* \brief Generates the required type definition of the Switch function.
*\param inputstr => input string to check the defined patterns present in it.
*\param _patterns => string patterns table or array.
*\param _patternsNbr => number of patterns in the array.
*\author : C S Priyadarshan.
*/
	bool CheckImageStrngs(std::string inputstr, std::string * _patterns, int _patternsNbr);


	/*!
*\brief sets the global pointer to the object manager in SGE.
*\return m_GetGenFctObjMngr.
*\author : C S Priyadarshan.
\todo [YCK] extract from this class
*/
	 static SGE::CL_TEMPLATE_OBJECT_MANAGER<CL_GenSwFn, std::string> *
		 GetFctObjMngr(void)
		{
			if(m_FctObjMngr==NULL)
				m_FctObjMngr = new SGE::CL_TEMPLATE_OBJECT_MANAGER<CL_GenSwFn, std::string>(NULL);
			return m_FctObjMngr;
		};	

};

}//namespace GCV
#endif
