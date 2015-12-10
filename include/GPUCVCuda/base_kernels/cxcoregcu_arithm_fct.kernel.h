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



/** \brief Contains some MACRO and structures to performs arithmetic and logics operator using NVIDIA CUDA.
\author Yannick Allusse
*/
#ifndef GPUCV_CUDA_ARITHM_FCT_KERNEL_H
#define GPUCV_CUDA_ARITHM_FCT_KERNEL_H

#include <GPUCVCuda/base_kernels/cuda_macro.kernel.cu>



__GCU_FCT_DEVICE_INLINE
unsigned int abs(unsigned int _val)
{
	return _val;
}
//KEYTAGS: TUTO_CUDA_ARITHM__STP1__DEFINE_OP
// Define macro for all kind of 2 parameters operators
//arithmetic
#define _ARITHM_OPER_ADD(A, B) ((A)+(B))
#define _ARITHM_OPER_SUB(A, B) ((A)-(B))
#define _ARITHM_OPER_SUBR(A, B) ((B)-(A))	//reverse subtraction
#define _ARITHM_OPER_MUL(A, B) ((A)*(B))
#define _ARITHM_OPER_DIV(A, B) (((float)(A))/((float)(B)))

#define _ARITHM_OPER_POWER(A, B) (powf((float)(A),(float)(B)))



//logic
#define _LOGIC_OPER_AND(A, B)	((A)&(B))
#define _LOGIC_OPER_OR(A, B)	((A)|(B))
#define _LOGIC_OPER_XOR(A, B)	((A)^(B))
#define _LOGIC_OPER_NOT(A)		((A)^0xFFFFFFFF)



#define _ARITHM_OPER_MIN(A, B) (min(A,B))
#define _ARITHM_OPER_MAX(A, B) (max(A,B))
//
#define _ARITHM_OPER_ABS(A) (abs(A))
#define _ARITHM_OPER_ABSDIFF(A, B) _ARITHM_OPER_ABS((A)-(B))
#define _ARITHM_OPER_CLAMP(A, B) (_Clamp(A,B))
#define _ARITHM_OPER_AFFECT(A, B) A=(B)
#define _ARITHM_OPER_CLEAR(A) 0
//
#define _ARITHM_OPER_EQUAL(A, B)			(((A) ==	(B))*0xFF)//Equal to
#define _ARITHM_OPER_GREATER(A, B)			(((A) >		(B))*0xFF)//Greater than
#define _ARITHM_OPER_GREATER_OR_EQUAL(A, B) (((A) >=	(B))*0xFF)//Greater or equal
#define _ARITHM_OPER_LESS(A, B)				(((A) <		(B))*0xFF)//Less than
#define _ARITHM_OPER_LESS_OR_EQUAL(A, B)	(((A) <=	(B))*0xFF)//Lesser than Equal to
#define _ARITHM_OPER_NOT_EQUAL(A, B)		(((A) !=	(B))*0xFF)//NOT Equal to

#define _ARITHM_OPER_MASK(A, B)		((A)?(B):0)//Mask..not tested..??

struct GCULogicStruct
{
};
struct GCUArithmStruct
{
};

/**	\brief This MACRO is used to generate arithmetics and logics structs operators that can be used into CUDA kernels, requires 2 values.
The operator supply must be compatible with the following synthaxe:
\code
Dst = OPER(Dst,A);
Dst = OPER(OPER(A,B),C);
Dst = OPER(OPER(OPER(A,B),C),D);
\endcode
Here is an example of operator definition that match the synthaxe:
\code
#define _ARITHM_OPER_ADD(A, B) ((A)+(B))
//this is translated by Compiler has
Dst = ((Dst)+(A));							=> Dst = Dst+A;
Dst = ((((A)+(B)))+(C));					=> Dst = A+B+C;
Dst = (((((A)+(B)))+(C))+(D));				=> Dst = A+B+C+D;
\endcode
\param OPER=> is a macro defining a 2 parameter operators(ex:CUDA_ARITHM_OPER_ADD)
\param NAME=> is the Macro for function name.
\note Special case: Do(Dest, A)=> the operation is Dst = OPER(Dst,A);
\sa CUDA_ARITHM_OPER_ADD
\author yannick Allusse
*/
#define DECLARE_TPL_ARITHM_OPER_2(NAME, OPER)				\
struct NAME											\
{													\
	template <typename TPLDst, typename TPL_A>		\
	__GCU_FCT_DEVICE_INLINE								\
	void Do(TPLDst & Dst, TPL_A & A)				\
{												\
	Dst = OPER(Dst,A);							\
}												\
	template <typename TPLDst, typename TPL_A, typename TPL_B>	\
	__GCU_FCT_DEVICE_INLINE								\
	void Do(TPLDst & Dst, TPL_A & A, TPL_B & B)					\
{															\
	Dst = (TPLDst)OPER(A,B);								\
}															\
	template <typename TPLDst, typename TPL_A, typename TPL_B, typename TPL_C>\
	__GCU_FCT_DEVICE_INLINE								\
	void Do(TPLDst & Dst, TPL_A & A, TPL_B & B, TPL_B & C)		\
{															\
	Dst = (TPLDst)OPER(OPER(A,B),C);						\
}															\
	template <typename TPLDst, typename TPL_A, typename TPL_B, typename TPL_C, typename TPL_D>\
	__GCU_FCT_DEVICE_INLINE								\
	void Do(TPLDst & Dst, TPL_A & A, TPL_B & B, TPL_C & C, TPL_D & D)\
{															\
	Dst = (TPLDst)OPER(OPER(OPER(A,B),C),D);				\
}															\
};

#define DECLARE_TPL_LOGIC_OPER_2(NAME, OPER)						\
struct NAME															\
{																	\
	template <typename TPLDst, typename TPL_A>						\
	__GCU_FCT_DEVICE_INLINE											\
	void Do(TPLDst & Dst, TPL_A & A)								\
{																	\
	Dst = (TPLDst)OPER((int)Dst,(int)A);							\
}																	\
	template <typename TPLDst, typename TPL_A, typename TPL_B>		\
	__GCU_FCT_DEVICE_INLINE											\
	void Do(TPLDst & Dst, TPL_A & A, TPL_B & B)						\
{																	\
	Dst = (TPLDst)OPER((int)A,(int)B);								\
}																	\
	template <typename TPLDst, typename TPL_A, typename TPL_B, typename TPL_C>\
	__GCU_FCT_DEVICE_INLINE											\
	void Do(TPLDst & Dst, TPL_A & A, TPL_B & B, TPL_B & C)			\
{																	\
	Dst = (TPLDst)OPER(OPER((int)A,(int)B),(int)C);					\
}																	\
	template <typename TPLDst, typename TPL_A, typename TPL_B, typename TPL_C, typename TPL_D>\
	__GCU_FCT_DEVICE_INLINE											\
	void Do(TPLDst & Dst, TPL_A & A, TPL_B & B, TPL_C & C, TPL_D & D)	\
{																		\
	Dst = (TPLDst)OPER(OPER(OPER((int)A,(int)B),(int)C),(int)D);		\
}																		\
};


/**	\brief This MACRO is used to generate arithmetics and logics structs operators that can be used into CUDA kernels, requires 1 values.
The operator supply must be compatible with the following syntaxes:
\code
Dst = OPER(A);
\endcode
\param NAME=> is the name for macro operations.
\param OPER=> is a macro defining a 1 parameter operators(ex:CUDA_ARITHM_OPER_SQUARE)
\note Special case: Do(Dest)=> the operation is Dst = OPER(Dst);
\sa CUDA_ARITHM_OPER_ADD
\author Yannick Allusse
*/
#define DECLARE_TPL_ARITHM_OPER_1(NAME, OPER)			\
struct NAME											\
{													\
	template <typename TPLDst>						\
	__GCU_FCT_DEVICE_INLINE							\
	void Do(TPLDst & Dst)							\
{												\
	Dst = OPER(Dst);							\
}												\
	template <typename TPLDst, typename TPL_A>		\
	__GCU_FCT_DEVICE_INLINE							\
	void Do(TPLDst & Dst, TPL_A & A)				\
{												\
	Dst = OPER(A);								\
}												\
	template <typename TPLDst, typename TPL_A, typename TPL_B>		\
	__GCU_FCT_DEVICE_INLINE							\
	void Do(TPLDst & Dst, TPL_A & A, TPL_B & B)				\
{												\
	Dst = OPER(A);								\
}												\
};

#define DECLARE_TPL_LOGIC_OPER_1(NAME, OPER)			\
struct NAME											\
{													\
	template <typename TPLDst>						\
	__GCU_FCT_DEVICE_INLINE							\
	void Do(TPLDst & Dst)							\
{												\
	Dst = (TPLDst)OPER((int)Dst);							\
}												\
	template <typename TPLDst, typename TPL_A>		\
	__GCU_FCT_DEVICE_INLINE							\
	void Do(TPLDst & Dst, TPL_A & A)				\
{												\
	Dst = (TPLDst)OPER((int)A);								\
}												\
	template <typename TPLDst, typename TPL_A, typename TPL_B>		\
	__GCU_FCT_DEVICE_INLINE							\
	void Do(TPLDst & Dst, TPL_A & A, TPL_B & B)				\
{												\
	Dst = (TPLDst)OPER((int)A);								\
}												\
};

//KEYTAGS: TUTO_CUDA_ARITHM__STP2__DECLARE_OP
//declare arithmetic structures for all arithmetic operators
DECLARE_TPL_ARITHM_OPER_2(KERNEL_ARITHM_OPER_ADD,		_ARITHM_OPER_ADD);
DECLARE_TPL_ARITHM_OPER_2(KERNEL_ARITHM_OPER_SUB,		_ARITHM_OPER_SUB);
DECLARE_TPL_ARITHM_OPER_2(KERNEL_ARITHM_OPER_SUBR,		_ARITHM_OPER_SUBR);
DECLARE_TPL_ARITHM_OPER_2(KERNEL_ARITHM_OPER_DIV,		_ARITHM_OPER_DIV);
DECLARE_TPL_ARITHM_OPER_2(KERNEL_ARITHM_OPER_MUL,		_ARITHM_OPER_MUL);

DECLARE_TPL_ARITHM_OPER_2(KERNEL_ARITHM_OPER_MIN,		_ARITHM_OPER_MIN);
DECLARE_TPL_ARITHM_OPER_2(KERNEL_ARITHM_OPER_MAX,		_ARITHM_OPER_MAX);
//...
DECLARE_TPL_ARITHM_OPER_2(KERNEL_ARITHM_OPER_CLAMP,		_ARITHM_OPER_CLAMP);

DECLARE_TPL_ARITHM_OPER_1(KERNEL_ARITHM_OPER_CLEAR,		_ARITHM_OPER_CLEAR);
DECLARE_TPL_ARITHM_OPER_1(KERNEL_ARITHM_OPER_ABS,		_ARITHM_OPER_ABS);
DECLARE_TPL_ARITHM_OPER_2(KERNEL_ARITHM_OPER_ABSDIFF,	_ARITHM_OPER_ABSDIFF);

DECLARE_TPL_ARITHM_OPER_2(KERNEL_ARITHM_OPER_POWER,		_ARITHM_OPER_POWER);

//logic
DECLARE_TPL_LOGIC_OPER_2(KERNEL_LOGIC_OPER_AND,		_LOGIC_OPER_AND);
DECLARE_TPL_LOGIC_OPER_2(KERNEL_LOGIC_OPER_OR,		_LOGIC_OPER_OR);
DECLARE_TPL_LOGIC_OPER_2(KERNEL_LOGIC_OPER_XOR,		_LOGIC_OPER_XOR);
DECLARE_TPL_LOGIC_OPER_1(KERNEL_LOGIC_OPER_NOT,		_LOGIC_OPER_NOT);

DECLARE_TPL_LOGIC_OPER_2(KERNEL_LOGIC_OPER_EQUAL,			_ARITHM_OPER_EQUAL);
DECLARE_TPL_LOGIC_OPER_2(KERNEL_LOGIC_OPER_GREATER,			_ARITHM_OPER_GREATER);
DECLARE_TPL_LOGIC_OPER_2(KERNEL_LOGIC_OPER_GREATER_OR_EQUAL,_ARITHM_OPER_GREATER_OR_EQUAL);
DECLARE_TPL_LOGIC_OPER_2(KERNEL_LOGIC_OPER_LESS,			_ARITHM_OPER_LESS);
DECLARE_TPL_LOGIC_OPER_2(KERNEL_LOGIC_OPER_LESS_OR_EQUAL,	_ARITHM_OPER_LESS_OR_EQUAL);
DECLARE_TPL_LOGIC_OPER_2(KERNEL_LOGIC_OPER_NOT_EQUAL,		_ARITHM_OPER_NOT_EQUAL);

//to avoid incoherent storage,
//we redefine it manually

//#define DECLARE_TPL_ARITHM_OPER_1(NAME, OPER)			\

DECLARE_TPL_ARITHM_OPER_2(KERNEL_ARITHM_OPER_AFFECT,	_ARITHM_OPER_AFFECT);


struct StAffectFilterTex					\
{													\
template <typename TPLDst>						\
__GCU_FCT_DEVICE_INLINE						\
void Do(TPLDst & Dst)							\
{												\
//Dst = OPER(Dst);							\

}												\
template <typename TPLDst, typename TPL_A>		\
__GCU_FCT_DEVICE_INLINE							\
void Do(TPLDst & Dst, TPL_A & A)				\
{
	TPL_A * TmpPointer = (TPL_A *)&Dst;\
		*TmpPointer = A;
}												\
template <typename TPLDst, typename TPL_A, typename TPL_B>		\
__GCU_FCT_DEVICE_INLINE							\
void Do(TPLDst & Dst, TPL_A & A, TPL_B & B)				\
{												\
TPL_A * TmpPointer = (TPL_A *)&Dst;\
*TmpPointer = A;
}												\
};



/** \brief Structs to process data of any type/channels and ouput to destination of any type/same channels using arithmetic operators given by FUNCTION.
\param channels -> Set the number of channels to process.
\param FUNCTION -> Is a struct declared by MACROS DECLARE_TPL_ARITHM_OPER_1 or DECLARE_TPL_ARITHM_OPER_2 that will perform the task required.
\author Yannick Allusse
When performing some arithmetic operators, results might be out of the destination type range (ex: uchar + uchar = [0..512]) and require to be clamped to the type range.
This structs performs the operator given by FUNCTION between A and B, apply a scale factor of fScale and clamp the result into the desination object.

It uses the struct GCU_OP_MULTIPLEXER to perform the FUNCTION on all channels.
\note This process is working for any kind of data (TPLDataDst, TPLDataSrc1, TPLDataSrc2) has long as they have the following format:
%type%%channels% whith type{uchar, char, uint, int, float} and channels {1,2,3,4}.
\note The clamp function is respectful with destination data type.
\sa GCU_OP_MULTIPLEXER, DECLARE_TPL_ARITHM_OPER_1, DECLARE_TPL_ARITHM_OPER_2
*/
template <int channels, typename FUNCTION>
struct StArithmFilterTex
{
	/**
	\param Dest -> Destination object.
	\param A -> First source.
	\param B -> Second source.
	\param fScale -> Scale factor.
	*/
	template <typename TPLDataDst, typename TPLDataSrc1, typename TPLDataSrc2>
	__GCU_FCT_DEVICE_INLINE
		void Do(TPLDataDst &Dest, TPLDataSrc1 & A, TPLDataSrc2 & B, float4 fScale)
	{
		float4 TempVal;//float is used cause we know the result may be out of the TPLDataDst range
		//GCUDA_KRNL_DBG(printf("\n=> Val1/2:%d + %d ", A.x, B.x);)
		//GCU_OP_MULTIPLEXER::Do<channels, FUNCTION, float4, TPLDataSrc1, TPLDataSrc2,TPLDataSrc2,TPLDataSrc2> (TempVal, A,B);
		GCU_MULTIPLEX_2(channels, FUNCTION, (TempVal, A,B), float4, TPLDataSrc1, TPLDataSrc2);
		//float4 TempScale;
		//TempScale.x = TempScale.y = TempScale.z = TempScale.w = 0.1;
		//GCU_OP_MULTIPLEXER::Do<channels, KERNEL_ARITHM_OPER_MUL, float4, float4, float4, float4, float4> (TempVal, fScale);//scale result
		GCU_MULTIPLEX_1(channels, KERNEL_ARITHM_OPER_MUL,(TempVal, fScale), float4, float4);
		//GCUDA_KRNL_DBG(printf("\nADDOPER=> Sum:%f", TempVal.x);)
		//clamp result into destination data

//write is coherent...
		TPLDataDst TempDest;
		//GCU_OP_MULTIPLEXER::Do<channels, KERNEL_ARITHM_OPER_CLAMP, TPLDataDst, float4, float4, float4, float4>(TempDest, TempVal);
		GCU_MULTIPLEX_1(channels, KERNEL_ARITHM_OPER_CLAMP,(TempDest, TempVal), TPLDataDst, float4);
		Dest = TempDest;
	}
};

template <int channels, typename FUNCTION>
struct StLogicFilterTex
{
	/**
	\param Dest -> Destination object.
	\param A -> First source.
	\param B -> Second source.
	\param fScale -> Scale factor, not used in logic operation, but we keep it to be compatible with main kernels
	*/
	template <typename TPLDataDst, typename TPLDataSrc>
	__GCU_FCT_DEVICE_INLINE
		void Do(TPLDataDst &Dest, TPLDataSrc & A, TPLDataSrc & B, float4 fScale)
	{
		int4 TempVal;
		GCU_MULTIPLEX_2(channels, FUNCTION,(TempVal, A,B), int4, TPLDataSrc, TPLDataSrc);
		TPLDataDst TempDest;
		GCU_MULTIPLEX_1(channels, KERNEL_ARITHM_OPER_AFFECT,(TempDest, TempVal), TPLDataDst, int4);
		Dest = TempDest;
	}
};
#endif
