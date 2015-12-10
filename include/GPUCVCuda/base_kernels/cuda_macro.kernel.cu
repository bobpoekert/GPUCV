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



/** \brief Contains macros to be used in CUDA kernels.
\author Yannick Allusse
*/
#ifndef  __GPUCV_CUDA_MACRO_KERNEL_CU
#define __GPUCV_CUDA_MACRO_KERNEL_CU
#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVCuda/config.h>
#include <cutil.h>


#define GCU_EMULATION_MODE 0
#if GCU_EMULATION_MODE
#define GCUDA_KRNL_DBG(FCT)FCT
#define GCUDA_KRNL_DBG_FIRST_THREAD(NAME, FCT)\
	if( threadIdx.x == 0 &&  threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)\
		{\
		printf("\n\n����������������������������������������������\n");\
		printf("%s()=>Start running kernel in EMULE mode!!\n",NAME);\
		printf("------------------------------------------------\n");\
		printf("Kernel parameters:\n");\
		printf("- File:\t %s\n", __FILE__);\
		printf("- Line:\t %d\n", __LINE__);\
		printf("- Compile date and time:\t %s - %s\n", __DATE__, __TIME__);\
		printf("- block dim:\t %d * %d * %d\n",blockDim.x, blockDim.y, blockDim.z);\
		printf("- grid dim:\t %d * %d * %d\n",gridDim.x, gridDim.y, gridDim.z);\
		printf("------------------------------------------------\n");\
		printf("Custom parameter:\n");\
		FCT\
		printf("------------------------------------------------\n");\
		printf("Debug infos...\n");\
		}

#define GCUDA_KRNL_DBG_LAST_THREAD(NAME, FCT)\
	if (threadIdx.x == blockDim.x-1 &&  threadIdx.y == blockDim.y-1 && blockIdx.x == gridDim.x-1 && blockIdx.y == gridDim.y-1)\
		{\
		FCT\
		printf("%s()=>Finished running kernel in EMULE mode!!\n",NAME);\
		printf("����������������������������������������������\n\n");\
		}

#define GCUDA_KRNL_DBG_SHOW_THREAD_POS()\
		{\
		printf("------------------------------------------------\n");\
		int BlockId = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;\
		int ThreadId = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;\
		printf("Block ID:%d{%d,%d,%d}\tThread ID:%d{%d,%d,%d}\n",\
		BlockId, blockIdx.x,blockIdx.y,blockIdx.z,\
		ThreadId, threadIdx.x,threadIdx.y,threadIdx.z);\
		printf("------------------------------------------------\n");\
		}
#else
#	define GCUDA_KRNL_DBG_SHOW_THREAD_POS()
#	define GCUDA_KRNL_DBG(FCT)
#	define GCUDA_KRNL_DBG_FIRST_THREAD(NAME, FCT)
#	define GCUDA_KRNL_DBG_LAST_THREAD(NAME, FCT)
#endif


//===================================================
/** Format support definitions. Can be used to get faster compilation when debugging or to disable support of some data types.
To enable/disable some data format, just comment the 'FCT' part of corresponding macro definition. 
*/
#ifdef _DEBUG
#	define _GPUCV_CUDA_SUPPORT_ALL_IMAGE_FORMAT 1
#else
#	define _GPUCV_CUDA_SUPPORT_ALL_IMAGE_FORMAT 1//compile all format in release mode
#endif

#define _GPUCV_CUDA_SUPPORT_UCHAR_FCT(FCT) FCT //Default format for testing

#ifdef _GPUCV_CUDA_SUPPORT_ALL_IMAGE_FORMAT
	#define _GPUCV_CUDA_SUPPORT_CHAR_FCT(FCT) FCT
	#define _GPUCV_CUDA_SUPPORT_INT_FCT(FCT) FCT
	#define _GPUCV_CUDA_SUPPORT_UINT_FCT(FCT) FCT
	#define _GPUCV_CUDA_SUPPORT_SHORT_FCT(FCT) FCT
	#define _GPUCV_CUDA_SUPPORT_USHORT_FCT(FCT) FCT
	#define _GPUCV_CUDA_SUPPORT_LONG_FCT(FCT) FCT
	#define _GPUCV_CUDA_SUPPORT_ULONG_FCT(FCT) FCT
	#define _GPUCV_CUDA_SUPPORT_FLOAT_FCT(FCT) FCT
	#if _GPUCV_CUDA_SUPPORT_DOUBLE_IMAGE_FORMAT
		#define _GPUCV_CUDA_SUPPORT_DOUBLE_FCT(FCT) FCT
	#else
		#define _GPUCV_CUDA_SUPPORT_DOUBLE_FCT(FCT)
	#endif
#else
	#define _GPUCV_CUDA_SUPPORT_CHAR_FCT(FCT)
	#define _GPUCV_CUDA_SUPPORT_INT_FCT(FCT)
	#define _GPUCV_CUDA_SUPPORT_UINT_FCT(FCT)
	#define _GPUCV_CUDA_SUPPORT_LONG_FCT(FCT)
	#define _GPUCV_CUDA_SUPPORT_ULONG_FCT(FCT) 
	#define _GPUCV_CUDA_SUPPORT_FLOAT_FCT(FCT) FCT
	#define _GPUCV_CUDA_SUPPORT_DOUBLE_FCT(FCT)
	#define _GPUCV_CUDA_SUPPORT_SHORT_FCT(FCT)
	#define _GPUCV_CUDA_SUPPORT_USHORT_FCT(FCT)
#endif
//Number of channel support definitions.
#define GCU_USE_CHANNEL_1_FCT(FCT) FCT
#define GCU_USE_CHANNEL_2_FCT(FCT) FCT
#define GCU_USE_CHANNEL_3_FCT(FCT) FCT
#define GCU_USE_CHANNEL_4_FCT(FCT) FCT
//===================================================


//===================================================
/** \brief Return the min or maximum value for each types
\sa http://home.att.net/~jackklein/c/inttypes.html#int for more details....
*/
#include <limits.h>
//template <typename TType>  __GCU_FCT_DEVICE_INLINE TType GetTypeMaxVal(TType &_val) {return ULONG_MAX;}
//template <typename TType>  __GCU_FCT_DEVICE_INLINE TType GetTypeMinVal(TType &_val) {return 0;}

//char
_GPUCV_CUDA_SUPPORT_CHAR_FCT(
	__GCU_FCT_DEVICE_INLINE char			GetTypeMaxVal			(char &_val){return CHAR_MAX;}
	__GCU_FCT_DEVICE_INLINE char			GetTypeMinVal			(char &_val){return CHAR_MIN;}
	__GCU_FCT_DEVICE_INLINE signed char		GetTypeMaxVal			(signed char &_val){return CHAR_MAX;}
	__GCU_FCT_DEVICE_INLINE signed char		GetTypeMinVal			(signed char &_val){return CHAR_MIN;}
	)
//unsigned char
_GPUCV_CUDA_SUPPORT_UCHAR_FCT(
	__GCU_FCT_DEVICE_INLINE unsigned char			GetTypeMaxVal			(unsigned char &_val){return UCHAR_MAX;}
	__GCU_FCT_DEVICE_INLINE unsigned char			GetTypeMinVal			(unsigned char &_val){return 0;}
	)
//short
_GPUCV_CUDA_SUPPORT_SHORT_FCT(
	__GCU_FCT_DEVICE_INLINE short			GetTypeMaxVal			(short &_val){return SHRT_MAX;}
	__GCU_FCT_DEVICE_INLINE short			GetTypeMinVal			(short &_val){return SHRT_MIN;}
	)
//unsigned short
_GPUCV_CUDA_SUPPORT_USHORT_FCT(
	__GCU_FCT_DEVICE_INLINE ushort			GetTypeMaxVal			(ushort &_val){return USHRT_MAX;}
	__GCU_FCT_DEVICE_INLINE ushort			GetTypeMinVal			(ushort &_val){return 0;}
	)
//int
_GPUCV_CUDA_SUPPORT_INT_FCT(
	__GCU_FCT_DEVICE_INLINE int				GetTypeMaxVal			(int &_val){return INT_MAX;}
	__GCU_FCT_DEVICE_INLINE int				GetTypeMinVal			(int &_val){return INT_MIN;}
	)
//unsigned int
_GPUCV_CUDA_SUPPORT_UINT_FCT(
	__GCU_FCT_DEVICE_INLINE uint			GetTypeMaxVal			(uint &_val){return UINT_MAX;}
	__GCU_FCT_DEVICE_INLINE uint			GetTypeMinVal			(uint &_val){return 0;}
	)
//long
_GPUCV_CUDA_SUPPORT_LONG_FCT(
	__GCU_FCT_DEVICE_INLINE long			GetTypeMaxVal			(long &_val){return LONG_MAX;}
	__GCU_FCT_DEVICE_INLINE long			GetTypeMinVal			(long &_val){return LONG_MIN;}
	)
//unsigned long
_GPUCV_CUDA_SUPPORT_ULONG_FCT(
	__GCU_FCT_DEVICE_INLINE unsigned long	GetTypeMaxVal			(unsigned long &_val){return ULONG_MAX;}
	__GCU_FCT_DEVICE_INLINE unsigned long	GetTypeMinVal			(unsigned long &_val){return 0;}
	)
//float and double
//no max/min values 
//===================================================

/** Clamping function that takes two template parameters to determine the clamping value.
\param Dst -> Src is clamped to destination max/min value and result is set to Dst.
\param Src -> input value to clamp.
\return Clamped value
*/
template <typename TTypeDst, typename TTypeSrc>
__GCU_FCT_DEVICE
TTypeDst& _Clamp(TTypeDst &Dst, TTypeSrc &Src)
{
	if(GetTypeMaxVal(Dst)  < Src)
		Dst = GetTypeMaxVal(Dst);
	else if (GetTypeMinVal(Dst)  > Src)
		Dst = GetTypeMinVal(Dst);
	else
		Dst = (TTypeDst)Src;
	return Dst;
}

__GCU_FCT_DEVICE_INLINE
	float& _Clamp(float &Dst, float &Src)
	{//no min max for float...
		return Dst=Src;
	}
__GCU_FCT_DEVICE_INLINE
	double& _Clamp(double &Dst, double &Src)
	{//no min max for double...
		return Dst=Src;
	}


#ifndef _LINUX
/** Preload some input data into shared memory
*/
template <typename TTypeShared, typename TTypeSrc>
__GCU_FCT_DEVICE
TTypeShared* _PreloadBlockData(TTypeShared *Shared, TTypeSrc *Src, unsigned int xIndex, unsigned int yIndex, unsigned int width, unsigned int height)
{
    // read the matrix tile into shared memory
   // if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        Shared[threadIdx.x + threadIdx.y*blockDim.x] = Src[index_in];
    }
	return Shared;
}
template <typename TTypeShared, typename TTypeDst>
__GCU_FCT_DEVICE
TTypeDst* _RestoreBlockData( TTypeDst *dst, TTypeShared *Shared, unsigned int xIndex, unsigned int yIndex, unsigned int width, unsigned int height)
{
    // read the matrix tile into shared memory
    {
        unsigned int index_out = yIndex * width + xIndex;
        dst[index_out] = Shared[threadIdx.x + threadIdx.y*blockDim.x];
    }
	return dst;
}
#endif
/*=================================================
*	MultiPlexer
*
==================================================*/
#define GCU_MULTIPLEX_0(CHANNELS, FUNCTION, PARAMS, DEST_TYPE)\
	GCU_OP_MULTIPLEXER<CHANNELS, FUNCTION, DEST_TYPE, DEST_TYPE, DEST_TYPE, DEST_TYPE, DEST_TYPE>::Do PARAMS;

#define GCU_MULTIPLEX_1(CHANNELS, FUNCTION, PARAMS, DEST_TYPE, SRC1_TYPE)\
	GCU_OP_MULTIPLEXER<CHANNELS, FUNCTION, DEST_TYPE, SRC1_TYPE, DEST_TYPE, DEST_TYPE, DEST_TYPE>::Do PARAMS;

#define GCU_MULTIPLEX_2(CHANNELS, FUNCTION, PARAMS, DEST_TYPE, SRC1_TYPE, SRC2_TYPE)\
	GCU_OP_MULTIPLEXER<CHANNELS, FUNCTION, DEST_TYPE, SRC1_TYPE, SRC2_TYPE, DEST_TYPE, DEST_TYPE>::Do PARAMS;

#define GCU_MULTIPLEX_3(CHANNELS, FUNCTION, PARAMS, DEST_TYPE, SRC1_TYPE, SRC2_TYPE, SRC3_TYPE)\
	GCU_OP_MULTIPLEXER<CHANNELS, FUNCTION, DEST_TYPE, SRC1_TYPE, SRC2_TYPE, SRC3_TYPE, DEST_TYPE>::Do PARAMS;

#define GCU_MULTIPLEX_4(CHANNELS, FUNCTION, PARAMS, DEST_TYPE, SRC1_TYPE, SRC2_TYPE, SRC3_TYPE, SRC4_TYPE)\
	GCU_OP_MULTIPLEXER<CHANNELS, FUNCTION, DEST_TYPE, SRC1_TYPE, SRC2_TYPE, SRC3_TYPE, SRC4_TYPE>::Do PARAMS;


template <int CHANNELS, typename TPL_Function, typename TPL_dest, typename TPL_A=float4, typename TPL_B=float4, typename TPL_C=float4, typename TPL_D=float4>
struct GCU_OP_MULTIPLEXER
{
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst)
	{
		//TPL_Function::Do(Dst.x);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A)
	{
		//TPL_Function::Do(Dst.x, A.x);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A, TPL_B & B)
	{
		//TPL_Function::Do(Dst.x, A.x, B.x);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A, TPL_B & B, TPL_B & C)
	{
		//TPL_Function::Do(Dst.x, A.x, B.x, C.x);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A, TPL_B & B, TPL_C & C, TPL_D & D)
	{
		//TPL_Function::Do(Dst.x, A.x, B.x, C.x, D.x);
	}
};

template <typename TPL_Function, typename TPL_dest, typename TPL_A, typename TPL_B, typename TPL_C, typename TPL_D>
struct GCU_OP_MULTIPLEXER<1, TPL_Function, TPL_dest, TPL_A, TPL_B, TPL_C, TPL_D>
{
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst)
	{
		TPL_Function::Do(Dst.x);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A)
	{
		TPL_Function::Do(Dst.x, A.x);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A, TPL_B & B)
	{
		TPL_Function::Do(Dst.x, A.x, B.x);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A, TPL_B & B, TPL_B & C)
	{
		TPL_Function::Do(Dst.x, A.x, B.x, C.x);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A, TPL_B & B, TPL_C & C, TPL_D & D)
	{
		TPL_Function::Do(Dst.x, A.x, B.x, C.x, D.x);
	}
};
//2
template <typename TPL_Function, typename TPL_dest, typename TPL_A, typename TPL_B, typename TPL_C, typename TPL_D>
struct GCU_OP_MULTIPLEXER<2, TPL_Function, TPL_dest, TPL_A, TPL_B, TPL_C, TPL_D>
{
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst)
	{
		TPL_Function::Do(Dst.x);
		TPL_Function::Do(Dst.y);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A)
	{
		TPL_Function::Do(Dst.x, A.x);
		TPL_Function::Do(Dst.y, A.y);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A, TPL_B & B)
	{
		TPL_Function::Do(Dst.x, A.x, B.x);
		TPL_Function::Do(Dst.y, A.y, B.y);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A, TPL_B & B, TPL_B & C)
	{
		TPL_Function::Do(Dst.x, A.x, B.x, C.x);
		TPL_Function::Do(Dst.y, A.y, B.y, C.y);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A, TPL_B & B, TPL_C & C, TPL_D & D)
	{
		TPL_Function::Do(Dst.x, A.x, B.x, C.x, D.x);
		TPL_Function::Do(Dst.y, A.y, B.y, C.y, D.y);
	}
};
//3
template <typename TPL_Function, typename TPL_dest, typename TPL_A, typename TPL_B, typename TPL_C, typename TPL_D>
struct GCU_OP_MULTIPLEXER<3, TPL_Function, TPL_dest, TPL_A, TPL_B, TPL_C, TPL_D>
{
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst)
	{
		TPL_Function::Do(Dst.x);
		TPL_Function::Do(Dst.y);
		TPL_Function::Do(Dst.z);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A)
	{
		TPL_Function::Do(Dst.x, A.x);
		TPL_Function::Do(Dst.y, A.y);
		TPL_Function::Do(Dst.z, A.z);
	}

	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A, TPL_B & B)
	{
		TPL_Function::Do(Dst.x, A.x, B.x);
		TPL_Function::Do(Dst.y, A.y, B.y);
		TPL_Function::Do(Dst.z, A.z, B.z);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A, TPL_B & B, TPL_C & C)
	{
		TPL_Function::Do(Dst.x, A.x, B.x, C.x);
		TPL_Function::Do(Dst.y, A.y, B.y, C.y);
		TPL_Function::Do(Dst.z, A.z, B.z, C.z);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A, TPL_B & B, TPL_C & C, TPL_D & D)
	{
		TPL_Function::Do(Dst.x, A.x, B.x, C.x, D.x);
		TPL_Function::Do(Dst.y, A.y, B.y, C.y, D.y);
		TPL_Function::Do(Dst.z, A.z, B.z, C.z, D.z);
	}
};
//4
template <typename TPL_Function, typename TPL_dest, typename TPL_A, typename TPL_B, typename TPL_C, typename TPL_D>
struct GCU_OP_MULTIPLEXER<4, TPL_Function, TPL_dest, TPL_A, TPL_B, TPL_C, TPL_D>
{
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst)
	{
		TPL_Function::Do(Dst.x);
		TPL_Function::Do(Dst.y);
		TPL_Function::Do(Dst.z);
		TPL_Function::Do(Dst.w);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A)
	{
		TPL_Function::Do(Dst.x, A.x);
		TPL_Function::Do(Dst.y, A.y);
		TPL_Function::Do(Dst.z, A.z);
		TPL_Function::Do(Dst.w, A.w);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst, TPL_A & A, TPL_B & B)
	{
		TPL_Function::Do(Dst.x, A.x, B.x);
		TPL_Function::Do(Dst.y, A.y, B.y);
		TPL_Function::Do(Dst.z, A.z, B.z);
		TPL_Function::Do(Dst.w, A.w, B.w);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst,  TPL_A & A, TPL_B & B,  TPL_B & C)
	{
		TPL_Function::Do(Dst.x, A.x, B.x, C.x);
		TPL_Function::Do(Dst.y, A.y, B.y, C.y);
		TPL_Function::Do(Dst.z, A.z, B.z, C.z);
		TPL_Function::Do(Dst.w, A.w, B.w, C.w);
	}
	__GCU_FCT_DEVICE_INLINE
		void Do(TPL_dest & Dst,  TPL_A & A,  TPL_B & B,  TPL_C & C,  TPL_D & D)
	{
		TPL_Function::Do(Dst.x, A.x, B.x, C.x, D.x);
		TPL_Function::Do(Dst.y, A.y, B.y, C.y, D.y);
		TPL_Function::Do(Dst.z, A.z, B.z, C.z, D.z);
		TPL_Function::Do(Dst.w, A.w, B.w, C.w, D.w);
	}
};


template  <int TPL_Channels, typename TPL_Function, typename TPL_dest>
__GCU_FCT_DEVICE_INLINE
void MultiPlex(TPL_dest & Dst)
{
	TPL_Function::Do(Dst.x);
#if (TPL_Channels>1)
	TPL_Function::Do(Dst.y);
#endif
#if (TPL_Channels>2)
	TPL_Function::Do(Dst.z);
#endif
#if (TPL_Channels>3)
	TPL_Function::Do(Dst.w);
#endif
}

template  <int TPL_Channels, typename TPL_Function, typename TPL_dest, typename TPL_A>
__GCU_FCT_DEVICE_INLINE
void MultiPlex(TPL_dest & Dst, TPL_A & A)
{
	TPL_Function::Do(Dst.x, A.x);
#if (TPL_Channels>1)
	TPL_Function::Do(Dst.y, A.y);
#endif
#if (TPL_Channels>2)
	TPL_Function::Do(Dst.z, A.z);
#endif
#if (TPL_Channels>3)
	TPL_Function::Do(Dst.w, A.w);
#endif
}

template  <int TPL_Channels, typename TPL_Function, typename TPL_dest, typename TPL_A, typename TPL_B>
__GCU_FCT_DEVICE_INLINE
void MultiPlex(TPL_dest & Dst, TPL_A & A, TPL_B & B)
{
	TPL_Function::Do(Dst.x, A.x, B.x);
#if (TPL_Channels>1)
	TPL_Function::Do(Dst.y, A.y, B.y);
#endif
#if (TPL_Channels>2)
	TPL_Function::Do(Dst.z, A.z, B.z);
#endif
#if (TPL_Channels>3)
	TPL_Function::Do(Dst.w, A.w, B.w);
#endif
}

template  <int TPL_Channels, typename TPL_Function, typename TPL_dest, typename TPL_A, typename TPL_B, typename TPL_C>
__GCU_FCT_DEVICE_INLINE
void MultiPlex(TPL_dest & Dst,  TPL_A & A, TPL_B & B,  TPL_B & C)
{
	TPL_Function::Do(Dst.x, A.x, B.x, C.x);
#if (TPL_Channels>1)
	TPL_Function::Do(Dst.y, A.y, B.y, C.y);
#endif
#if (TPL_Channels>2)
	TPL_Function::Do(Dst.z, A.z, B.z, C.z);
#endif
#if (TPL_Channels>3)
	TPL_Function::Do(Dst.w, A.w, B.w, C.w);
#endif
}

template  <int TPL_Channels, typename TPL_Function, typename TPL_dest, typename TPL_A, typename TPL_B, typename TPL_C, typename TPL_D>
__GCU_FCT_DEVICE_INLINE
void MultiPlex(TPL_dest & Dst,  TPL_A & A,  TPL_B & B,  TPL_C & C,  TPL_D & D)
{
	TPL_Function::Do(Dst.x, A.x, B.x, C.x, D.x);
#if (TPL_Channels>1)
	TPL_Function::Do(Dst.y, A.y, B.y, C.y, D.y);
#endif
#if (TPL_Channels>2)
	TPL_Function::Do(Dst.z, A.z, B.z, C.z, D.z);
#endif
#if (TPL_Channels>3)
	TPL_Function::Do(Dst.w, A.w, B.w, C.w, D.w);
#endif
}



/** \brief Multiplexing macro: execute the given macro(CALL) for the given format(FORMAT) using the given nbr of channels(CHANNELS). Switch depending on the FORMAT.
 * */
#define GCU_MULTIPLEX_1CHANNELS_ALLFORMAT(CALL, CHANNELS, FORMAT)						\
	switch(FORMAT) 																\
{ 																			\
	_GPUCV_CUDA_SUPPORT_UCHAR_FCT(	case GL_UNSIGNED_BYTE:		CALL(CHANNELS, uchar,	uchar);		break;)	\
	_GPUCV_CUDA_SUPPORT_CHAR_FCT(	case GL_BYTE:				CALL(CHANNELS, char,	char);		break;)	\
	_GPUCV_CUDA_SUPPORT_USHORT_FCT(	case GL_UNSIGNED_SHORT:		CALL(CHANNELS, ushort,	ushort);	break;)	\
	_GPUCV_CUDA_SUPPORT_SHORT_FCT(	case GL_SHORT:				CALL(CHANNELS, short,	short);		break;)	\
	_GPUCV_CUDA_SUPPORT_INT_FCT(	case GL_INT:				CALL(CHANNELS, int,		int);		break;)	\
	_GPUCV_CUDA_SUPPORT_UINT_FCT(	case GL_UNSIGNED_INT:		CALL(CHANNELS, uint,	uint);		break;)	\
	_GPUCV_CUDA_SUPPORT_FLOAT_FCT(	case GL_FLOAT:				CALL(CHANNELS, float,	float);		break;)	\
	_GPUCV_CUDA_SUPPORT_DOUBLE_FCT(	case GL_DOUBLE:				CALL(CHANNELS, double,	double);	break;)	\
	default:printf("\nGCU_MULTIPLEX_1CHANNELS_ALLFORMAT()=> unknown case.");\
}


/** \brief Multiplexing macro: execute the given macro(CALL) for the given format(FORMAT) using the given nbr of channels(CHANNELS). Switch depending on the CHANNELS
 * */
#define GCU_MULTIPLEX_ALLCHANNELS_ALLFORMAT(CALL, CHANNELS, FORMAT)		\
	switch(CHANNELS)													\
{																		\
	case 1:	GCU_MULTIPLEX_1CHANNELS_ALLFORMAT(CALL, 1, FORMAT);break;	\
	case 2:	GCU_MULTIPLEX_1CHANNELS_ALLFORMAT(CALL, 2, FORMAT);break;	\
	case 3:	GCU_MULTIPLEX_1CHANNELS_ALLFORMAT(CALL, 3, FORMAT);break;	\
	case 4:	GCU_MULTIPLEX_1CHANNELS_ALLFORMAT(CALL, 4, FORMAT);break;	\
	default:printf("\nGCU_MULTIPLEX_ALLCHANNELS_ALLFORMAT()=> unknown case.");\
}



//Used for convertScale operator
#if 0//_DEBUG
#define GCU_MULTIPLEX_CONVERT_DEBUG_CALL(CALL, CHANNELS, SRC_FORMAT, DST_FORMAT)\
        printf("\GCU_MULTIPLEX_CONVERT_DEBUG_CALL============================");\
        printf("\nFunction called: "#CALL);\
        printf("\nChannels: "#CHANNELS);\
        printf("\nSrc type: "#SRC_FORMAT);\
        printf("\nDst type: "#DST_FORMAT);\
        CALL(CHANNELS, SRC_FORMAT,	DST_FORMAT);\
        printf("\nGCU_MULTIPLEX_CONVERT_DEBUG_CALL============================");
#else
#define GCU_MULTIPLEX_CONVERT_DEBUG_CALL(CALL, CHANNELS, SRC_FORMAT, DST_FORMAT)\
        CALL(CHANNELS, SRC_FORMAT,	DST_FORMAT);
#endif

#define GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLDSTFORMAT(CALL, CHANNELS, SRC_FORMAT, DST_FORMAT)						\
        switch(DST_FORMAT) 																\
{ 																			\
        _GPUCV_CUDA_SUPPORT_UCHAR_FCT(	case GL_UNSIGNED_BYTE:		GCU_MULTIPLEX_CONVERT_DEBUG_CALL(CALL, CHANNELS, SRC_FORMAT,uchar);		break;)	\
        _GPUCV_CUDA_SUPPORT_CHAR_FCT(	case GL_BYTE:				GCU_MULTIPLEX_CONVERT_DEBUG_CALL(CALL, CHANNELS, SRC_FORMAT,char);		break;)	\
        _GPUCV_CUDA_SUPPORT_USHORT_FCT(	case GL_UNSIGNED_SHORT:		GCU_MULTIPLEX_CONVERT_DEBUG_CALL(CALL, CHANNELS, SRC_FORMAT,ushort);	break;)	\
        _GPUCV_CUDA_SUPPORT_SHORT_FCT(	case GL_SHORT:				GCU_MULTIPLEX_CONVERT_DEBUG_CALL(CALL, CHANNELS, SRC_FORMAT,short);		break;)	\
        _GPUCV_CUDA_SUPPORT_UINT_FCT(	case GL_UNSIGNED_INT:		GCU_MULTIPLEX_CONVERT_DEBUG_CALL(CALL, CHANNELS, SRC_FORMAT,uint);		break;)	\
        _GPUCV_CUDA_SUPPORT_INT_FCT(	case GL_INT:				GCU_MULTIPLEX_CONVERT_DEBUG_CALL(CALL, CHANNELS, SRC_FORMAT,int);		break;)	\
        _GPUCV_CUDA_SUPPORT_FLOAT_FCT(	case GL_FLOAT:				GCU_MULTIPLEX_CONVERT_DEBUG_CALL(CALL, CHANNELS, SRC_FORMAT,float);		break;)	\
        _GPUCV_CUDA_SUPPORT_DOUBLE_FCT(	case GL_DOUBLE:				GCU_MULTIPLEX_CONVERT_DEBUG_CALL(CALL, CHANNELS, SRC_FORMAT,double);	break;)	\
        default:printf("\nGCU_MULTIPLEX_CONVERT_1CHANNELS_ALLDSTFORMAT()=> unknown case.");\
}




#define GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLSRCFORMAT(CALL, CHANNELS, SRC_FORMAT, DST_FORMAT)						\
        switch(SRC_FORMAT) 																\
{ 																			\
        _GPUCV_CUDA_SUPPORT_UCHAR_FCT(	case GL_UNSIGNED_BYTE:		GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLDSTFORMAT(CALL,CHANNELS, uchar,DST_FORMAT);	break;)	\
        _GPUCV_CUDA_SUPPORT_CHAR_FCT(	case GL_BYTE:				GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLDSTFORMAT(CALL,CHANNELS, char,DST_FORMAT);	break;)	\
        _GPUCV_CUDA_SUPPORT_USHORT_FCT(	case GL_UNSIGNED_SHORT:		GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLDSTFORMAT(CALL,CHANNELS, ushort,DST_FORMAT);	break;)	\
        _GPUCV_CUDA_SUPPORT_SHORT_FCT(	case GL_SHORT:				GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLDSTFORMAT(CALL,CHANNELS, short,DST_FORMAT);	break;)	\
        _GPUCV_CUDA_SUPPORT_INT_FCT(	case GL_INT:				GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLDSTFORMAT(CALL,CHANNELS, int,DST_FORMAT);	break;)	\
        _GPUCV_CUDA_SUPPORT_UINT_FCT(		case GL_UNSIGNED_INT:		GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLDSTFORMAT(CALL,CHANNELS, uint,DST_FORMAT);	break;)	\
        _GPUCV_CUDA_SUPPORT_FLOAT_FCT(	case GL_FLOAT:				GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLDSTFORMAT(CALL,CHANNELS, float,DST_FORMAT);	break;)	\
        _GPUCV_CUDA_SUPPORT_DOUBLE_FCT( case GL_DOUBLE:				GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLDSTFORMAT(CALL,CHANNELS, double,DST_FORMAT);	break;)	\
        default:printf("\nGCU_MULTIPLEX_CONVERT_1CHANNELS_ALLSRCFORMAT()=> unknown case.");\
}

#define GCU_MULTIPLEX_CONVERT_ALLCHANNELS_ALLFORMAT(CALL, CHANNELS, SRC_FORMAT, DST_FORMAT)\
        switch(CHANNELS)														\
{																		\
        case 1:	GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLSRCFORMAT(CALL, 1, SRC_FORMAT, DST_FORMAT);break;	\
        case 2:	GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLSRCFORMAT(CALL, 2, SRC_FORMAT, DST_FORMAT);break;	\
        case 3:	GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLSRCFORMAT(CALL, 3, SRC_FORMAT, DST_FORMAT);break;	\
        case 4:	GCU_MULTIPLEX_CONVERT_1CHANNELS_ALLSRCFORMAT(CALL, 4, SRC_FORMAT, DST_FORMAT);break;	\
        default:printf("\nGCU_MULTIPLEX_CONVERT_ALLCHANNELS_ALLFORMAT()=> unknown case.");\
}







#endif

