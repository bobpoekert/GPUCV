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
#include <GPUCVSwitch/macro.h>
#include <GPUCVCore/GpuTextureManager.h>
#include <GPUCVSwitch/Cl_Dll.h>
#include <GPUCVSwitch/switch.h>

#define _GPUCV_FORCE_OPENCV_NP 1
#include <includecv.h>
#include <highgui.h>


using namespace std;
using namespace GCV;
#define CVAPI(MSG) MSG

#include <cxcore_switch/cxcore_switch.h>
#include <GPUCVSwitch/switch.h>
/*====================================*/
void cvg_cxcore_switch_RegisterTracerSingletons(SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> * _pAppliTracer, SG_TRC::CL_TRACING_EVENT_LIST *_pEventList)
{
SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance().RegisterNewSingleton(_pAppliTracer);
SG_TRC::CL_TRACING_EVENT_LIST::Instance().RegisterNewSingleton(_pEventList);
}
/*====================================*/

/*====================================*/
void* cvgswAlloc(size_t size)
{
	return cvAlloc((size_t) size);
}


/*====================================*/
void cvgswFree_(void* ptr)
{
	cvFree_((void*) ptr);
}


/*====================================*/
IplImage* cvgswCreateImageHeader(CvSize size, int depth, int channels)
{
	typedef IplImage*(*_CreateImageHeader) (CvSize, int, int ); 
	GPUCV_FUNCNAME("cvCreateImageHeader");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	IplImage* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((CvSize) size, (int) depth, (int) channels), _CreateImageHeader,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
IplImage* cvgswInitImageHeader(IplImage* image, CvSize size, int depth, int channels, int origin, int align)
{
	typedef IplImage*(*_InitImageHeader) (IplImage*, CvSize, int, int, int, int ); 
	GPUCV_FUNCNAME("cvInitImageHeader");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	IplImage* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((IplImage*) image, (CvSize) size, (int) depth, (int) channels, (int) origin, (int) align), _InitImageHeader,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
IplImage* cvgswCreateImage(CvSize size, int depth, int channels)
{
	typedef IplImage*(*_CreateImage) (CvSize, int, int ); 
	GPUCV_FUNCNAME("cvCreateImage");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	IplImage* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((CvSize) size, (int) depth, (int) channels), _CreateImage,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswReleaseImageHeader(IplImage** image)
{
	typedef void(*_ReleaseImageHeader) (IplImage** ); 
	GPUCV_FUNCNAME("cvReleaseImageHeader");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((IplImage**) image), _ReleaseImageHeader, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswReleaseImage(IplImage** image)
{
	typedef void(*_ReleaseImage) (IplImage** ); 
	GPUCV_FUNCNAME("cvReleaseImage");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((IplImage**) image), _ReleaseImage, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
IplImage* cvgswCloneImage( IplImage* image)
{
	typedef IplImage*(*_CloneImage) ( IplImage* ); 
	GPUCV_FUNCNAME("cvCloneImage");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	IplImage* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( IplImage*) image), _CloneImage,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswSetImageCOI(IplImage* image, int coi)
{
	typedef void(*_SetImageCOI) (IplImage*, int ); 
	GPUCV_FUNCNAME("cvSetImageCOI");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((IplImage*) image, (int) coi), _SetImageCOI, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswGetImageCOI( IplImage* image)
{
	typedef int(*_GetImageCOI) ( IplImage* ); 
	GPUCV_FUNCNAME("cvGetImageCOI");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( IplImage*) image), _GetImageCOI,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswSetImageROI(IplImage* image, CvRect rect)
{
	typedef void(*_SetImageROI) (IplImage*, CvRect ); 
	GPUCV_FUNCNAME("cvSetImageROI");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((IplImage*) image, (CvRect) rect), _SetImageROI, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswResetImageROI(IplImage* image)
{
	typedef void(*_ResetImageROI) (IplImage* ); 
	GPUCV_FUNCNAME("cvResetImageROI");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((IplImage*) image), _ResetImageROI, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
CvRect cvgswGetImageROI( IplImage* image)
{
	typedef CvRect(*_GetImageROI) ( IplImage* ); 
	GPUCV_FUNCNAME("cvGetImageROI");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	CvRect ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( IplImage*) image), _GetImageROI,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswCreateMatHeader(int rows, int cols, int type)
{
	typedef CvMat*(*_CreateMatHeader) (int, int, int ); 
	GPUCV_FUNCNAME("cvCreateMatHeader");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((int) rows, (int) cols, (int) type), _CreateMatHeader,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswInitMatHeader(CvMat* mat, int rows, int cols, int type, void* data, int step)
{
	typedef CvMat*(*_InitMatHeader) (CvMat*, int, int, int, void*, int ); 
	GPUCV_FUNCNAME("cvInitMatHeader");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mat)paramsobj->AddParam("option", "MASK");
	RUNOP(((CvMat*) mat, (int) rows, (int) cols, (int) type, (void*) data, (int) step), _InitMatHeader,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswCreateMat(int rows, int cols, int type)
{
	typedef CvMat*(*_CreateMat) (int, int, int ); 
	GPUCV_FUNCNAME("cvCreateMat");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((int) rows, (int) cols, (int) type), _CreateMat,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswReleaseMat(CvMat** mat)
{
	typedef void(*_ReleaseMat) (CvMat** ); 
	GPUCV_FUNCNAME("cvReleaseMat");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvMat**) mat), _ReleaseMat, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswDecRefData(CvArr* arr)
{
	typedef void(*_DecRefData) (CvArr* ); 
	GPUCV_FUNCNAME("cvDecRefData");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr), _DecRefData, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswIncRefData(CvArr* arr)
{
	typedef int(*_IncRefData) (CvArr* ); 
	GPUCV_FUNCNAME("cvIncRefData");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr), _IncRefData,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswCloneMat( CvMat* mat)
{
	typedef CvMat*(*_CloneMat) ( CvMat* ); 
	GPUCV_FUNCNAME("cvCloneMat");
	CvArr* SrcARR[] = { (CvArr*) mat};
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvMat*) mat), _CloneMat,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswGetSubRect( CvArr* arr, CvMat* submat, CvRect rect)
{
	typedef CvMat*(*_GetSubRect) ( CvArr*, CvMat*, CvRect ); 
	GPUCV_FUNCNAME("cvGetSubRect");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&submat)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) arr, (CvMat*) submat, (CvRect) rect), _GetSubRect,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswGetRows( CvArr* arr, CvMat* submat, int start_row, int end_row, int delta_row)
{
	typedef CvMat*(*_GetRows) ( CvArr*, CvMat*, int, int, int ); 
	GPUCV_FUNCNAME("cvGetRows");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&submat)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) arr, (CvMat*) submat, (int) start_row, (int) end_row, (int) delta_row), _GetRows,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswGetRow( CvArr* arr, CvMat* submat, int row)
{
	typedef CvMat*(*_GetRow) ( CvArr*, CvMat*, int ); 
	GPUCV_FUNCNAME("cvGetRow");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&submat)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) arr, (CvMat*) submat, (int) row), _GetRow,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswGetCols( CvArr* arr, CvMat* submat, int start_col, int end_col)
{
	typedef CvMat*(*_GetCols) ( CvArr*, CvMat*, int, int ); 
	GPUCV_FUNCNAME("cvGetCols");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&submat)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) arr, (CvMat*) submat, (int) start_col, (int) end_col), _GetCols,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswGetCol( CvArr* arr, CvMat* submat, int col)
{
	typedef CvMat*(*_GetCol) ( CvArr*, CvMat*, int ); 
	GPUCV_FUNCNAME("cvGetCol");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&submat)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) arr, (CvMat*) submat, (int) col), _GetCol,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswGetDiag( CvArr* arr, CvMat* submat, int diag)
{
	typedef CvMat*(*_GetDiag) ( CvArr*, CvMat*, int ); 
	GPUCV_FUNCNAME("cvGetDiag");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&submat)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) arr, (CvMat*) submat, (int) diag), _GetDiag,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswScalarToRawData(const  CvScalar* scalar, void* data, int type, int extend_to_12)
{
	cvScalarToRawData((const  CvScalar*) scalar, (void*) data, (int) type, (int) extend_to_12);
}


/*====================================*/
void cvgswRawDataToScalar(const  void* data, int type, CvScalar* scalar)
{
	cvRawDataToScalar((const  void*) data, (int) type, (CvScalar*) scalar);
}


/*====================================*/
CvMatND* cvgswCreateMatNDHeader(int dims, const  int* sizes, int type)
{
	return cvCreateMatNDHeader((int) dims, (const  int*) sizes, (int) type);
}


/*====================================*/
CvMatND* cvgswCreateMatND(int dims, const  int* sizes, int type)
{
	return cvCreateMatND((int) dims, (const  int*) sizes, (int) type);
}


/*====================================*/
CvMatND* cvgswInitMatNDHeader(CvMatND* mat, int dims, const  int* sizes, int type, void* data)
{
	return cvInitMatNDHeader((CvMatND*) mat, (int) dims, (const  int*) sizes, (int) type, (void*) data);
}


/*====================================*/
void cvgswReleaseMatND(CvMatND** mat)
{
	cvReleaseMatND((CvMatND**) mat);
}


/*====================================*/
CvMatND* cvgswCloneMatND(const  CvMatND* mat)
{
	return cvCloneMatND((const  CvMatND*) mat);
}


/*====================================*/
CvSparseMat* cvgswCreateSparseMat(int dims, const  int* sizes, int type)
{
	return cvCreateSparseMat((int) dims, (const  int*) sizes, (int) type);
}


/*====================================*/
void cvgswReleaseSparseMat(CvSparseMat** mat)
{
	cvReleaseSparseMat((CvSparseMat**) mat);
}


/*====================================*/
CvSparseMat* cvgswCloneSparseMat(const  CvSparseMat* mat)
{
	return cvCloneSparseMat((const  CvSparseMat*) mat);
}


/*====================================*/
CvSparseNode* cvgswInitSparseMatIterator(const  CvSparseMat* mat, CvSparseMatIterator* mat_iterator)
{
	return cvInitSparseMatIterator((const  CvSparseMat*) mat, (CvSparseMatIterator*) mat_iterator);
}


/*====================================*/
CvSparseNode* cvgswGetNextSparseNode(CvSparseMatIterator* mat_iterator)
{
	return cvGetNextSparseNode((CvSparseMatIterator*) mat_iterator);
}


/*====================================*/
int cvgswInitNArrayIterator(int count, CvArr** arrs,  CvArr* mask, CvMatND* stubs, CvNArrayIterator* array_iterator, int flags)
{
	typedef int(*_InitNArrayIterator) (int, CvArr**,  CvArr*, CvMatND*, CvNArrayIterator*, int ); 
	GPUCV_FUNCNAME("cvInitNArrayIterator");
	CvArr* SrcARR[] = { (CvArr*) mask};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP(((int) count, (CvArr**) arrs, ( CvArr*) mask, (CvMatND*) stubs, (CvNArrayIterator*) array_iterator, (int) flags), _InitNArrayIterator,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
int cvgswNextNArraySlice(CvNArrayIterator* array_iterator)
{
	return cvNextNArraySlice((CvNArrayIterator*) array_iterator);
}


/*====================================*/
int cvgswGetElemType( CvArr* arr)
{
	typedef int(*_GetElemType) ( CvArr* ); 
	GPUCV_FUNCNAME("cvGetElemType");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr), _GetElemType,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
int cvgswGetDims( CvArr* arr, int* sizes)
{
	typedef int(*_GetDims) ( CvArr*, int* ); 
	GPUCV_FUNCNAME("cvGetDims");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (int*) sizes), _GetDims,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
int cvgswGetDimSize( CvArr* arr, int index)
{
	typedef int(*_GetDimSize) ( CvArr*, int ); 
	GPUCV_FUNCNAME("cvGetDimSize");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (int) index), _GetDimSize,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
uchar* cvgswPtr1D( CvArr* arr, int idx0, int* type)
{
	typedef uchar*(*_Ptr1D) ( CvArr*, int, int* ); 
	GPUCV_FUNCNAME("cvPtr1D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	uchar* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (int) idx0, (int*) type), _Ptr1D,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
uchar* cvgswPtr2D( CvArr* arr, int idx0, int idx1, int* type)
{
	typedef uchar*(*_Ptr2D) ( CvArr*, int, int, int* ); 
	GPUCV_FUNCNAME("cvPtr2D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	uchar* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (int) idx0, (int) idx1, (int*) type), _Ptr2D,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
uchar* cvgswPtr3D( CvArr* arr, int idx0, int idx1, int idx2, int* type)
{
	typedef uchar*(*_Ptr3D) ( CvArr*, int, int, int, int* ); 
	GPUCV_FUNCNAME("cvPtr3D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	uchar* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (int) idx0, (int) idx1, (int) idx2, (int*) type), _Ptr3D,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
uchar* cvgswPtrND( CvArr* arr, const  int* idx, int* type, int create_node, unsigned* precalc_hashval)
{
	typedef uchar*(*_PtrND) ( CvArr*, const  int*, int*, int, unsigned* ); 
	GPUCV_FUNCNAME("cvPtrND");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	uchar* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (const  int*) idx, (int*) type, (int) create_node, (unsigned*) precalc_hashval), _PtrND,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvScalar cvgswGet1D( CvArr* arr, int idx0)
{
	typedef CvScalar(*_Get1D) ( CvArr*, int ); 
	GPUCV_FUNCNAME("cvGet1D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvScalar ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (int) idx0), _Get1D,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvScalar cvgswGet2D( CvArr* arr, int idx0, int idx1)
{
	typedef CvScalar(*_Get2D) ( CvArr*, int, int ); 
	GPUCV_FUNCNAME("cvGet2D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvScalar ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (int) idx0, (int) idx1), _Get2D,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvScalar cvgswGet3D( CvArr* arr, int idx0, int idx1, int idx2)
{
	typedef CvScalar(*_Get3D) ( CvArr*, int, int, int ); 
	GPUCV_FUNCNAME("cvGet3D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvScalar ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (int) idx0, (int) idx1, (int) idx2), _Get3D,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvScalar cvgswGetND( CvArr* arr, const  int* idx)
{
	typedef CvScalar(*_GetND) ( CvArr*, const  int* ); 
	GPUCV_FUNCNAME("cvGetND");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvScalar ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (const  int*) idx), _GetND,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
double cvgswGetReal1D( CvArr* arr, int idx0)
{
	typedef double(*_GetReal1D) ( CvArr*, int ); 
	GPUCV_FUNCNAME("cvGetReal1D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	double ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (int) idx0), _GetReal1D,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
double cvgswGetReal2D( CvArr* arr, int idx0, int idx1)
{
	typedef double(*_GetReal2D) ( CvArr*, int, int ); 
	GPUCV_FUNCNAME("cvGetReal2D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	double ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (int) idx0, (int) idx1), _GetReal2D,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
double cvgswGetReal3D( CvArr* arr, int idx0, int idx1, int idx2)
{
	typedef double(*_GetReal3D) ( CvArr*, int, int, int ); 
	GPUCV_FUNCNAME("cvGetReal3D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	double ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (int) idx0, (int) idx1, (int) idx2), _GetReal3D,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
double cvgswGetRealND( CvArr* arr, const  int* idx)
{
	typedef double(*_GetRealND) ( CvArr*, const  int* ); 
	GPUCV_FUNCNAME("cvGetRealND");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	double ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (const  int*) idx), _GetRealND,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswSet1D(CvArr* arr, int idx0, CvScalar value)
{
	typedef void(*_Set1D) (CvArr*, int, CvScalar ); 
	GPUCV_FUNCNAME("cvSet1D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr, (int) idx0, (CvScalar) value), _Set1D, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSet2D(CvArr* arr, int idx0, int idx1, CvScalar value)
{
	typedef void(*_Set2D) (CvArr*, int, int, CvScalar ); 
	GPUCV_FUNCNAME("cvSet2D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr, (int) idx0, (int) idx1, (CvScalar) value), _Set2D, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSet3D(CvArr* arr, int idx0, int idx1, int idx2, CvScalar value)
{
	typedef void(*_Set3D) (CvArr*, int, int, int, CvScalar ); 
	GPUCV_FUNCNAME("cvSet3D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr, (int) idx0, (int) idx1, (int) idx2, (CvScalar) value), _Set3D, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSetND(CvArr* arr, const  int* idx, CvScalar value)
{
	typedef void(*_SetND) (CvArr*, const  int*, CvScalar ); 
	GPUCV_FUNCNAME("cvSetND");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr, (const  int*) idx, (CvScalar) value), _SetND, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSetReal1D(CvArr* arr, int idx0, double value)
{
	typedef void(*_SetReal1D) (CvArr*, int, double ); 
	GPUCV_FUNCNAME("cvSetReal1D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr, (int) idx0, (double) value), _SetReal1D, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSetReal2D(CvArr* arr, int idx0, int idx1, double value)
{
	typedef void(*_SetReal2D) (CvArr*, int, int, double ); 
	GPUCV_FUNCNAME("cvSetReal2D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr, (int) idx0, (int) idx1, (double) value), _SetReal2D, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSetReal3D(CvArr* arr, int idx0, int idx1, int idx2, double value)
{
	typedef void(*_SetReal3D) (CvArr*, int, int, int, double ); 
	GPUCV_FUNCNAME("cvSetReal3D");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr, (int) idx0, (int) idx1, (int) idx2, (double) value), _SetReal3D, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSetRealND(CvArr* arr, const  int* idx, double value)
{
	typedef void(*_SetRealND) (CvArr*, const  int*, double ); 
	GPUCV_FUNCNAME("cvSetRealND");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr, (const  int*) idx, (double) value), _SetRealND, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswClearND(CvArr* arr, const  int* idx)
{
	typedef void(*_ClearND) (CvArr*, const  int* ); 
	GPUCV_FUNCNAME("cvClearND");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr, (const  int*) idx), _ClearND, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
CvMat* cvgswGetMat( CvArr* arr, CvMat* header, int* coi, int allowND)
{
	typedef CvMat*(*_GetMat) ( CvArr*, CvMat*, int*, int ); 
	GPUCV_FUNCNAME("cvGetMat");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&header)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) arr, (CvMat*) header, (int*) coi, (int) allowND), _GetMat,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
IplImage* cvgswGetImage( CvArr* arr, IplImage* image_header)
{
	typedef IplImage*(*_GetImage) ( CvArr*, IplImage* ); 
	GPUCV_FUNCNAME("cvGetImage");
	CvArr* SrcARR[] = { (CvArr*) arr,  (CvArr*) image_header};
	CvArr** DstARR = NULL;
	IplImage* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (IplImage*) image_header), _GetImage,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvArr* cvgswReshapeMatND( CvArr* arr, int sizeof_header, CvArr* header, int new_cn, int new_dims, int* new_sizes)
{
	typedef CvArr*(*_ReshapeMatND) ( CvArr*, int, CvArr*, int, int, int* ); 
	GPUCV_FUNCNAME("cvReshapeMatND");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvArr* ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&header)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) arr, (int) sizeof_header, (CvArr*) header, (int) new_cn, (int) new_dims, (int*) new_sizes), _ReshapeMatND,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgswReshape( CvArr* arr, CvMat* header, int new_cn, int new_rows)
{
	typedef CvMat*(*_Reshape) ( CvArr*, CvMat*, int, int ); 
	GPUCV_FUNCNAME("cvReshape");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&header)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) arr, (CvMat*) header, (int) new_cn, (int) new_rows), _Reshape,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswRepeat( CvArr* src, CvArr* dst)
{
	typedef void(*_Repeat) ( CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvRepeat");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst), _Repeat, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCreateData(CvArr* arr)
{
	typedef void(*_CreateData) (CvArr* ); 
	GPUCV_FUNCNAME("cvCreateData");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr), _CreateData, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswReleaseData(CvArr* arr)
{
	typedef void(*_ReleaseData) (CvArr* ); 
	GPUCV_FUNCNAME("cvReleaseData");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr), _ReleaseData, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSetData(CvArr* arr, void* data, int step)
{
	typedef void(*_SetData) (CvArr*, void*, int ); 
	GPUCV_FUNCNAME("cvSetData");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr, (void*) data, (int) step), _SetData, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswGetRawData( CvArr* arr, uchar** data, int* step, CvSize* roi_size)
{
	typedef void(*_GetRawData) ( CvArr*, uchar**, int*, CvSize* ); 
	GPUCV_FUNCNAME("cvGetRawData");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (uchar**) data, (int*) step, (CvSize*) roi_size), _GetRawData, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
CvSize cvgswGetSize( CvArr* arr)
{
	typedef CvSize(*_GetSize) ( CvArr* ); 
	GPUCV_FUNCNAME("cvGetSize");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvSize ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr), _GetSize,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswCopy( CvArr* src, CvArr* dst,  CvArr* mask)
{
	typedef void(*_Copy) ( CvArr*, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvCopy");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src, (CvArr*) dst, ( CvArr*) mask), _Copy, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSet(CvArr* arr, CvScalar value,  CvArr* mask)
{
	typedef void(*_Set) (CvArr*, CvScalar,  CvArr* ); 
	GPUCV_FUNCNAME("cvSet");
	CvArr* SrcARR[] = { (CvArr*) arr,  (CvArr*) mask};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP(((CvArr*) arr, (CvScalar) value, ( CvArr*) mask), _Set, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSetZero(CvArr* arr)
{
	typedef void(*_SetZero) (CvArr* ); 
	GPUCV_FUNCNAME("cvSetZero");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) arr), _SetZero, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSplit( CvArr* src, CvArr* dst0, CvArr* dst1, CvArr* dst2, CvArr* dst3)
{
	typedef void(*_Split) ( CvArr*, CvArr*, CvArr*, CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvSplit");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst0,  (CvArr*) dst1,  (CvArr*) dst2,  (CvArr*) dst3};
	SWITCH_START_OPR(dst0); 
	RUNOP((( CvArr*) src, (CvArr*) dst0, (CvArr*) dst1, (CvArr*) dst2, (CvArr*) dst3), _Split, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMerge( CvArr* src0,  CvArr* src1,  CvArr* src2,  CvArr* src3, CvArr* dst)
{
	typedef void(*_Merge) ( CvArr*,  CvArr*,  CvArr*,  CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvMerge");
	CvArr* SrcARR[] = { (CvArr*) src0,  (CvArr*) src1,  (CvArr*) src2,  (CvArr*) src3};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src0, ( CvArr*) src1, ( CvArr*) src2, ( CvArr*) src3, (CvArr*) dst), _Merge, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMixChannels( CvArr** src, int src_count, CvArr** dst, int dst_count, const  int* from_to, int pair_count)
{
	typedef void(*_MixChannels) ( CvArr**, int, CvArr**, int, const  int*, int ); 
	GPUCV_FUNCNAME("cvMixChannels");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr**) src, (int) src_count, (CvArr**) dst, (int) dst_count, (const  int*) from_to, (int) pair_count), _MixChannels, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswConvertScale( CvArr* src, CvArr* dst, double scale, double shift)
{
	typedef void(*_ConvertScale) ( CvArr*, CvArr*, double, double ); 
	GPUCV_FUNCNAME("cvConvertScale");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (double) scale, (double) shift), _ConvertScale, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswConvertScaleAbs( CvArr* src, CvArr* dst, double scale, double shift)
{
	typedef void(*_ConvertScaleAbs) ( CvArr*, CvArr*, double, double ); 
	GPUCV_FUNCNAME("cvConvertScaleAbs");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (double) scale, (double) shift), _ConvertScaleAbs, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
CvTermCriteria cvgswCheckTermCriteria(CvTermCriteria criteria, double default_eps, int default_max_iters)
{
	return cvCheckTermCriteria((CvTermCriteria) criteria, (double) default_eps, (int) default_max_iters);
}


/*====================================*/
void cvgswAdd( CvArr* src1,  CvArr* src2, CvArr* dst,  CvArr* mask)
{
	typedef void(*_Add) ( CvArr*,  CvArr*, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvAdd");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst, ( CvArr*) mask), _Add, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswAddS( CvArr* src, CvScalar value, CvArr* dst,  CvArr* mask)
{
	typedef void(*_AddS) ( CvArr*, CvScalar, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvAddS");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src, (CvScalar) value, (CvArr*) dst, ( CvArr*) mask), _AddS, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSub( CvArr* src1,  CvArr* src2, CvArr* dst,  CvArr* mask)
{
	typedef void(*_Sub) ( CvArr*,  CvArr*, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvSub");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst, ( CvArr*) mask), _Sub, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSubS( CvArr* src, CvScalar value, CvArr* dst,  CvArr* mask)
{
	typedef void(*_SubS) ( CvArr*, CvScalar, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvSubS");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src, (CvScalar) value, (CvArr*) dst, ( CvArr*) mask), _SubS, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSubRS( CvArr* src, CvScalar value, CvArr* dst,  CvArr* mask)
{
	typedef void(*_SubRS) ( CvArr*, CvScalar, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvSubRS");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src, (CvScalar) value, (CvArr*) dst, ( CvArr*) mask), _SubRS, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMul( CvArr* src1,  CvArr* src2, CvArr* dst, double scale)
{
	typedef void(*_Mul) ( CvArr*,  CvArr*, CvArr*, double ); 
	GPUCV_FUNCNAME("cvMul");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst, (double) scale), _Mul, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswDiv( CvArr* src1,  CvArr* src2, CvArr* dst, double scale)
{
	typedef void(*_Div) ( CvArr*,  CvArr*, CvArr*, double ); 
	GPUCV_FUNCNAME("cvDiv");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst, (double) scale), _Div, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswScaleAdd( CvArr* src1, CvScalar scale,  CvArr* src2, CvArr* dst)
{
	typedef void(*_ScaleAdd) ( CvArr*, CvScalar,  CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvScaleAdd");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src1, (CvScalar) scale, ( CvArr*) src2, (CvArr*) dst), _ScaleAdd, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswAddWeighted( CvArr* src1, double alpha,  CvArr* src2, double beta, double gamma, CvArr* dst)
{
	typedef void(*_AddWeighted) ( CvArr*, double,  CvArr*, double, double, CvArr* ); 
	GPUCV_FUNCNAME("cvAddWeighted");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src1, (double) alpha, ( CvArr*) src2, (double) beta, (double) gamma, (CvArr*) dst), _AddWeighted, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
double cvgswDotProduct( CvArr* src1,  CvArr* src2)
{
	typedef double(*_DotProduct) ( CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvDotProduct");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2};
	CvArr** DstARR = NULL;
	double ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) src1, ( CvArr*) src2), _DotProduct,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswAnd( CvArr* src1,  CvArr* src2, CvArr* dst,  CvArr* mask)
{
	typedef void(*_And) ( CvArr*,  CvArr*, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvAnd");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst, ( CvArr*) mask), _And, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswAndS( CvArr* src, CvScalar value, CvArr* dst,  CvArr* mask)
{
	typedef void(*_AndS) ( CvArr*, CvScalar, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvAndS");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src, (CvScalar) value, (CvArr*) dst, ( CvArr*) mask), _AndS, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswOr( CvArr* src1,  CvArr* src2, CvArr* dst,  CvArr* mask)
{
	typedef void(*_Or) ( CvArr*,  CvArr*, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvOr");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst, ( CvArr*) mask), _Or, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswOrS( CvArr* src, CvScalar value, CvArr* dst,  CvArr* mask)
{
	typedef void(*_OrS) ( CvArr*, CvScalar, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvOrS");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src, (CvScalar) value, (CvArr*) dst, ( CvArr*) mask), _OrS, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswXor( CvArr* src1,  CvArr* src2, CvArr* dst,  CvArr* mask)
{
	typedef void(*_Xor) ( CvArr*,  CvArr*, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvXor");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst, ( CvArr*) mask), _Xor, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswXorS( CvArr* src, CvScalar value, CvArr* dst,  CvArr* mask)
{
	typedef void(*_XorS) ( CvArr*, CvScalar, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvXorS");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src, (CvScalar) value, (CvArr*) dst, ( CvArr*) mask), _XorS, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswNot( CvArr* src, CvArr* dst)
{
	typedef void(*_Not) ( CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvNot");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst), _Not, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswInRange( CvArr* src,  CvArr* lower,  CvArr* upper, CvArr* dst)
{
	typedef void(*_InRange) ( CvArr*,  CvArr*,  CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvInRange");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) lower,  (CvArr*) upper};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, ( CvArr*) lower, ( CvArr*) upper, (CvArr*) dst), _InRange, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswInRangeS( CvArr* src, CvScalar lower, CvScalar upper, CvArr* dst)
{
	typedef void(*_InRangeS) ( CvArr*, CvScalar, CvScalar, CvArr* ); 
	GPUCV_FUNCNAME("cvInRangeS");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvScalar) lower, (CvScalar) upper, (CvArr*) dst), _InRangeS, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCmp( CvArr* src1,  CvArr* src2, CvArr* dst, int cmp_op)
{
	typedef void(*_Cmp) ( CvArr*,  CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvCmp");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst, (int) cmp_op), _Cmp, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCmpS( CvArr* src, double value, CvArr* dst, int cmp_op)
{
	typedef void(*_CmpS) ( CvArr*, double, CvArr*, int ); 
	GPUCV_FUNCNAME("cvCmpS");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (double) value, (CvArr*) dst, (int) cmp_op), _CmpS, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMin( CvArr* src1,  CvArr* src2, CvArr* dst)
{
	typedef void(*_Min) ( CvArr*,  CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvMin");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst), _Min, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMax( CvArr* src1,  CvArr* src2, CvArr* dst)
{
	typedef void(*_Max) ( CvArr*,  CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvMax");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst), _Max, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMinS( CvArr* src, double value, CvArr* dst)
{
	typedef void(*_MinS) ( CvArr*, double, CvArr* ); 
	GPUCV_FUNCNAME("cvMinS");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (double) value, (CvArr*) dst), _MinS, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMaxS( CvArr* src, double value, CvArr* dst)
{
	typedef void(*_MaxS) ( CvArr*, double, CvArr* ); 
	GPUCV_FUNCNAME("cvMaxS");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (double) value, (CvArr*) dst), _MaxS, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswAbsDiff( CvArr* src1,  CvArr* src2, CvArr* dst)
{
	typedef void(*_AbsDiff) ( CvArr*,  CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvAbsDiff");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst), _AbsDiff, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswAbsDiffS( CvArr* src, CvArr* dst, CvScalar value)
{
	typedef void(*_AbsDiffS) ( CvArr*, CvArr*, CvScalar ); 
	GPUCV_FUNCNAME("cvAbsDiffS");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (CvScalar) value), _AbsDiffS, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCartToPolar( CvArr* x,  CvArr* y, CvArr* magnitude, CvArr* angle, int angle_in_degrees)
{
	typedef void(*_CartToPolar) ( CvArr*,  CvArr*, CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvCartToPolar");
	CvArr* SrcARR[] = { (CvArr*) x,  (CvArr*) y};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&magnitude)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&angle)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) x, ( CvArr*) y, (CvArr*) magnitude, (CvArr*) angle, (int) angle_in_degrees), _CartToPolar, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswPolarToCart( CvArr* magnitude,  CvArr* angle, CvArr* x, CvArr* y, int angle_in_degrees)
{
	typedef void(*_PolarToCart) ( CvArr*,  CvArr*, CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvPolarToCart");
	CvArr* SrcARR[] = { (CvArr*) magnitude,  (CvArr*) angle};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&x)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&y)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) magnitude, ( CvArr*) angle, (CvArr*) x, (CvArr*) y, (int) angle_in_degrees), _PolarToCart, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswPow( CvArr* src, CvArr* dst, double power)
{
	typedef void(*_Pow) ( CvArr*, CvArr*, double ); 
	GPUCV_FUNCNAME("cvPow");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (double) power), _Pow, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswExp( CvArr* src, CvArr* dst)
{
	typedef void(*_Exp) ( CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvExp");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst), _Exp, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswLog( CvArr* src, CvArr* dst)
{
	typedef void(*_Log) ( CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvLog");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst), _Log, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
float cvgswFastArctan(float y, float x)
{
	return cvFastArctan((float) y, (float) x);
}


/*====================================*/
float cvgswCbrt(float value)
{
	return cvCbrt((float) value);
}


/*====================================*/
int cvgswCheckArr( CvArr* arr, int flags, double min_val, double max_val)
{
	typedef int(*_CheckArr) ( CvArr*, int, double, double ); 
	GPUCV_FUNCNAME("cvCheckArr");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (int) flags, (double) min_val, (double) max_val), _CheckArr,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswRandArr(CvRNG* rng, CvArr* arr, int dist_type, CvScalar param1, CvScalar param2)
{
	typedef void(*_RandArr) (CvRNG*, CvArr*, int, CvScalar, CvScalar ); 
	GPUCV_FUNCNAME("cvRandArr");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvRNG*) rng, (CvArr*) arr, (int) dist_type, (CvScalar) param1, (CvScalar) param2), _RandArr, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswRandShuffle(CvArr* mat, CvRNG* rng, double iter_factor)
{
	cvRandShuffle((CvArr*) mat, (CvRNG*) rng, (double) iter_factor);
}


/*====================================*/
void cvgswSort( CvArr* src, CvArr* dst, CvArr* idxmat, int flags)
{
	typedef void(*_Sort) ( CvArr*, CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvSort");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&idxmat)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src, (CvArr*) dst, (CvArr*) idxmat, (int) flags), _Sort, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswSolveCubic( CvMat* coeffs, CvMat* roots)
{
	typedef int(*_SolveCubic) ( CvMat*, CvMat* ); 
	GPUCV_FUNCNAME("cvSolveCubic");
	CvArr* SrcARR[] = { (CvArr*) coeffs};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&roots)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) coeffs, (CvMat*) roots), _SolveCubic,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswSolvePoly( CvMat* coeffs, CvMat * roots2, 			int maxiter, int fig)
{
	typedef void(*_SolvePoly) ( CvMat*, CvMat *, 			int, int ); 
	GPUCV_FUNCNAME("cvSolvePoly");
	CvArr* SrcARR[] = { (CvArr*) coeffs};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP((( CvMat*) coeffs, (CvMat *) roots2, (			int) maxiter, (int) fig), _SolvePoly, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCrossProduct( CvArr* src1,  CvArr* src2, CvArr* dst)
{
	typedef void(*_CrossProduct) ( CvArr*,  CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvCrossProduct");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst), _CrossProduct, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswGEMM( CvArr* src1,  CvArr* src2, double alpha,  CvArr* src3, double beta, CvArr* dst, int tABC)
{
	typedef void(*_GEMM) ( CvArr*,  CvArr*, double,  CvArr*, double, CvArr*, int ); 
	GPUCV_FUNCNAME("cvGEMM");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2,  (CvArr*) src3};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (double) alpha, ( CvArr*) src3, (double) beta, (CvArr*) dst, (int) tABC), _GEMM, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswTransform( CvArr* src, CvArr* dst,  CvMat* transmat,  CvMat* shiftvec)
{
	typedef void(*_Transform) ( CvArr*, CvArr*,  CvMat*,  CvMat* ); 
	GPUCV_FUNCNAME("cvTransform");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) transmat,  (CvArr*) shiftvec};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, ( CvMat*) transmat, ( CvMat*) shiftvec), _Transform, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswPerspectiveTransform( CvArr* src, CvArr* dst,  CvMat* mat)
{
	typedef void(*_PerspectiveTransform) ( CvArr*, CvArr*,  CvMat* ); 
	GPUCV_FUNCNAME("cvPerspectiveTransform");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) mat};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, ( CvMat*) mat), _PerspectiveTransform, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMulTransposed( CvArr* src, CvArr* dst, int order,  CvArr* delta, double scale)
{
	typedef void(*_MulTransposed) ( CvArr*, CvArr*, int,  CvArr*, double ); 
	GPUCV_FUNCNAME("cvMulTransposed");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) delta};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) order, ( CvArr*) delta, (double) scale), _MulTransposed, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswTranspose( CvArr* src, CvArr* dst)
{
	typedef void(*_Transpose) ( CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvTranspose");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst), _Transpose, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCompleteSymm(CvMat* matrix, int LtoR)
{
	cvCompleteSymm((CvMat*) matrix, (int) LtoR);
}


/*====================================*/
void cvgswFlip( CvArr* src, CvArr* dst, int flip_mode)
{
	typedef void(*_Flip) ( CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvFlip");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) flip_mode), _Flip, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSVD(CvArr* A, CvArr* W, CvArr* U, CvArr* V, int flags)
{
	cvSVD((CvArr*) A, (CvArr*) W, (CvArr*) U, (CvArr*) V, (int) flags);
}


/*====================================*/
void cvgswSVBkSb( CvArr* W,  CvArr* U,  CvArr* V,  CvArr* B, CvArr* X, int flags)
{
	typedef void(*_SVBkSb) ( CvArr*,  CvArr*,  CvArr*,  CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvSVBkSb");
	CvArr* SrcARR[] = { (CvArr*) W,  (CvArr*) U,  (CvArr*) V,  (CvArr*) B};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&X)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) W, ( CvArr*) U, ( CvArr*) V, ( CvArr*) B, (CvArr*) X, (int) flags), _SVBkSb, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
double cvgswInvert( CvArr* src, CvArr* dst, int method)
{
	typedef double(*_Invert) ( CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvInvert");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	double ReturnObj;	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) method), _Invert,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
int cvgswSolve( CvArr* src1,  CvArr* src2, CvArr* dst, int method)
{
	typedef int(*_Solve) ( CvArr*,  CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvSolve");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2};
	CvArr* DstARR[] = { (CvArr*) dst};
	int ReturnObj;	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst, (int) method), _Solve,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
double cvgswDet( CvArr* mat)
{
	typedef double(*_Det) ( CvArr* ); 
	GPUCV_FUNCNAME("cvDet");
	CvArr* SrcARR[] = { (CvArr*) mat};
	CvArr** DstARR = NULL;
	double ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) mat), _Det,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvScalar cvgswTrace( CvArr* mat)
{
	typedef CvScalar(*_Trace) ( CvArr* ); 
	GPUCV_FUNCNAME("cvTrace");
	CvArr* SrcARR[] = { (CvArr*) mat};
	CvArr** DstARR = NULL;
	CvScalar ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) mat), _Trace,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswEigenVV(CvArr* mat, CvArr* evects, CvArr* evals, double eps, int lowindex, int highindex)
{
	cvEigenVV((CvArr*) mat, (CvArr*) evects, (CvArr*) evals, (double) eps, (int) lowindex, (int) highindex);
}


/*====================================*/
void cvgswSetIdentity(CvArr* mat, CvScalar value)
{
	cvSetIdentity((CvArr*) mat, (CvScalar) value);
}


/*====================================*/
CvArr* cvgswRange(CvArr* mat, double start, double end)
{
	typedef CvArr*(*_Range) (CvArr*, double, double ); 
	GPUCV_FUNCNAME("cvRange");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	CvArr* ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mat)paramsobj->AddParam("option", "MASK");
	RUNOP(((CvArr*) mat, (double) start, (double) end), _Range,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswCalcCovarMatrix( CvArr** vects, int count, CvArr* cov_mat, CvArr* avg, int flags)
{
	typedef void(*_CalcCovarMatrix) ( CvArr**, int, CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvCalcCovarMatrix");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&cov_mat)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&avg)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr**) vects, (int) count, (CvArr*) cov_mat, (CvArr*) avg, (int) flags), _CalcCovarMatrix, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCalcPCA( CvArr* data, CvArr* mean, CvArr* eigenvals, CvArr* eigenvects, int flags)
{
	typedef void(*_CalcPCA) ( CvArr*, CvArr*, CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvCalcPCA");
	CvArr* SrcARR[] = { (CvArr*) data};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mean)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&eigenvals)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&eigenvects)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) data, (CvArr*) mean, (CvArr*) eigenvals, (CvArr*) eigenvects, (int) flags), _CalcPCA, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswProjectPCA( CvArr* data,  CvArr* mean,  CvArr* eigenvects, CvArr* result)
{
	typedef void(*_ProjectPCA) ( CvArr*,  CvArr*,  CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvProjectPCA");
	CvArr* SrcARR[] = { (CvArr*) data,  (CvArr*) mean,  (CvArr*) eigenvects};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&result)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) data, ( CvArr*) mean, ( CvArr*) eigenvects, (CvArr*) result), _ProjectPCA, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswBackProjectPCA( CvArr* proj,  CvArr* mean,  CvArr* eigenvects, CvArr* result)
{
	typedef void(*_BackProjectPCA) ( CvArr*,  CvArr*,  CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvBackProjectPCA");
	CvArr* SrcARR[] = { (CvArr*) proj,  (CvArr*) mean,  (CvArr*) eigenvects};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&result)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) proj, ( CvArr*) mean, ( CvArr*) eigenvects, (CvArr*) result), _BackProjectPCA, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
double cvgswMahalanobis( CvArr* vec1,  CvArr* vec2,  CvArr* mat)
{
	typedef double(*_Mahalanobis) ( CvArr*,  CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvMahalanobis");
	CvArr* SrcARR[] = { (CvArr*) vec1,  (CvArr*) vec2,  (CvArr*) mat};
	CvArr** DstARR = NULL;
	double ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) vec1, ( CvArr*) vec2, ( CvArr*) mat), _Mahalanobis,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvScalar cvgswSum( CvArr* arr)
{
	typedef CvScalar(*_Sum) ( CvArr* ); 
	GPUCV_FUNCNAME("cvSum");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	CvScalar ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr), _Sum,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
int cvgswCountNonZero( CvArr* arr)
{
	typedef int(*_CountNonZero) ( CvArr* ); 
	GPUCV_FUNCNAME("cvCountNonZero");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr), _CountNonZero,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvScalar cvgswAvg( CvArr* arr,  CvArr* mask)
{
	typedef CvScalar(*_Avg) ( CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvAvg");
	CvArr* SrcARR[] = { (CvArr*) arr,  (CvArr*) mask};
	CvArr** DstARR = NULL;
	CvScalar ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) arr, ( CvArr*) mask), _Avg,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswAvgSdv( CvArr* arr, CvScalar* mean, CvScalar* std_dev,  CvArr* mask)
{
	typedef void(*_AvgSdv) ( CvArr*, CvScalar*, CvScalar*,  CvArr* ); 
	GPUCV_FUNCNAME("cvAvgSdv");
	CvArr* SrcARR[] = { (CvArr*) arr,  (CvArr*) mask};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) arr, (CvScalar*) mean, (CvScalar*) std_dev, ( CvArr*) mask), _AvgSdv, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMinMaxLoc( CvArr* arr, double* min_val, double* max_val, CvPoint* min_loc, CvPoint* max_loc,  CvArr* mask)
{
	typedef void(*_MinMaxLoc) ( CvArr*, double*, double*, CvPoint*, CvPoint*,  CvArr* ); 
	GPUCV_FUNCNAME("cvMinMaxLoc");
	CvArr* SrcARR[] = { (CvArr*) arr,  (CvArr*) mask};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) arr, (double*) min_val, (double*) max_val, (CvPoint*) min_loc, (CvPoint*) max_loc, ( CvArr*) mask), _MinMaxLoc, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
double cvgswNorm( CvArr* arr1,  CvArr* arr2, int norm_type,  CvArr* mask)
{
	typedef double(*_Norm) ( CvArr*,  CvArr*, int,  CvArr* ); 
	GPUCV_FUNCNAME("cvNorm");
	CvArr* SrcARR[] = { (CvArr*) arr1,  (CvArr*) arr2,  (CvArr*) mask};
	CvArr** DstARR = NULL;
	double ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) arr1, ( CvArr*) arr2, (int) norm_type, ( CvArr*) mask), _Norm,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswNormalize( CvArr* src, CvArr* dst, double a, double b, int norm_type,  CvArr* mask)
{
	typedef void(*_Normalize) ( CvArr*, CvArr*, double, double, int,  CvArr* ); 
	GPUCV_FUNCNAME("cvNormalize");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src, (CvArr*) dst, (double) a, (double) b, (int) norm_type, ( CvArr*) mask), _Normalize, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswReduce( CvArr* src, CvArr* dst, int dim, int op)
{
	typedef void(*_Reduce) ( CvArr*, CvArr*, int, int ); 
	GPUCV_FUNCNAME("cvReduce");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) dim, (int) op), _Reduce, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswDFT( CvArr* src, CvArr* dst, int flags, int nonzero_rows)
{
	typedef void(*_DFT) ( CvArr*, CvArr*, int, int ); 
	GPUCV_FUNCNAME("cvDFT");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) flags, (int) nonzero_rows), _DFT, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMulSpectrums( CvArr* src1,  CvArr* src2, CvArr* dst, int flags)
{
	typedef void(*_MulSpectrums) ( CvArr*,  CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvMulSpectrums");
	CvArr* SrcARR[] = { (CvArr*) src1,  (CvArr*) src2};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src1, ( CvArr*) src2, (CvArr*) dst, (int) flags), _MulSpectrums, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswGetOptimalDFTSize(int size0)
{
	return cvGetOptimalDFTSize((int) size0);
}


/*====================================*/
void cvgswDCT( CvArr* src, CvArr* dst, int flags)
{
	typedef void(*_DCT) ( CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvDCT");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) flags), _DCT, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswSliceLength(CvSlice slice, const  CvSeq* seq)
{
	return cvSliceLength((CvSlice) slice, (const  CvSeq*) seq);
}


/*====================================*/
CvMemStorage* cvgswCreateMemStorage(int block_size)
{
	return cvCreateMemStorage((int) block_size);
}


/*====================================*/
CvMemStorage* cvgswCreateChildMemStorage(CvMemStorage* parent)
{
	return cvCreateChildMemStorage((CvMemStorage*) parent);
}


/*====================================*/
void cvgswReleaseMemStorage(CvMemStorage** storage)
{
	cvReleaseMemStorage((CvMemStorage**) storage);
}


/*====================================*/
void cvgswClearMemStorage(CvMemStorage* storage)
{
	cvClearMemStorage((CvMemStorage*) storage);
}


/*====================================*/
void cvgswSaveMemStoragePos(const  CvMemStorage* storage, CvMemStoragePos* pos)
{
	cvSaveMemStoragePos((const  CvMemStorage*) storage, (CvMemStoragePos*) pos);
}


/*====================================*/
void cvgswRestoreMemStoragePos(CvMemStorage* storage, CvMemStoragePos* pos)
{
	cvRestoreMemStoragePos((CvMemStorage*) storage, (CvMemStoragePos*) pos);
}


/*====================================*/
void* cvgswMemStorageAlloc(CvMemStorage* storage, size_t size)
{
	return cvMemStorageAlloc((CvMemStorage*) storage, (size_t) size);
}


/*====================================*/
CvString cvgswMemStorageAllocString(CvMemStorage* storage, const  char* ptr, int len)
{
	return cvMemStorageAllocString((CvMemStorage*) storage, (const  char*) ptr, (int) len);
}


/*====================================*/
CvSeq* cvgswCreateSeq(int seq_flags, int header_size, int elem_size, CvMemStorage* storage)
{
	return cvCreateSeq((int) seq_flags, (int) header_size, (int) elem_size, (CvMemStorage*) storage);
}


/*====================================*/
void cvgswSetSeqBlockSize(CvSeq* seq, int delta_elems)
{
	cvSetSeqBlockSize((CvSeq*) seq, (int) delta_elems);
}


/*====================================*/
CVAPI(schar*) cvgswSeqPush(CvSeq* seq, const  void* element)
{
	return cvSeqPush((CvSeq*) seq, (const  void*) element);
}


/*====================================*/
CVAPI(schar*) cvgswSeqPushFront(CvSeq* seq, const  void* element)
{
	return cvSeqPushFront((CvSeq*) seq, (const  void*) element);
}


/*====================================*/
void cvgswSeqPop(CvSeq* seq, void* element)
{
	cvSeqPop((CvSeq*) seq, (void*) element);
}


/*====================================*/
void cvgswSeqPopFront(CvSeq* seq, void* element)
{
	cvSeqPopFront((CvSeq*) seq, (void*) element);
}


/*====================================*/
void cvgswSeqPushMulti(CvSeq* seq, const  void* elements, int count, int in_front)
{
	cvSeqPushMulti((CvSeq*) seq, (const  void*) elements, (int) count, (int) in_front);
}


/*====================================*/
void cvgswSeqPopMulti(CvSeq* seq, void* elements, int count, int in_front)
{
	cvSeqPopMulti((CvSeq*) seq, (void*) elements, (int) count, (int) in_front);
}


/*====================================*/
CVAPI(schar*) cvgswSeqInsert(CvSeq* seq, int before_index, const  void* element)
{
	return cvSeqInsert((CvSeq*) seq, (int) before_index, (const  void*) element);
}


/*====================================*/
void cvgswSeqRemove(CvSeq* seq, int index)
{
	cvSeqRemove((CvSeq*) seq, (int) index);
}


/*====================================*/
void cvgswClearSeq(CvSeq* seq)
{
	cvClearSeq((CvSeq*) seq);
}


/*====================================*/
CVAPI(schar*) cvgswGetSeqElem(const  CvSeq* seq, int index)
{
	return cvGetSeqElem((const  CvSeq*) seq, (int) index);
}


/*====================================*/
int cvgswSeqElemIdx(const  CvSeq* seq, const  void* element, CvSeqBlock** block)
{
	return cvSeqElemIdx((const  CvSeq*) seq, (const  void*) element, (CvSeqBlock**) block);
}


/*====================================*/
void cvgswStartAppendToSeq(CvSeq* seq, CvSeqWriter* writer)
{
	cvStartAppendToSeq((CvSeq*) seq, (CvSeqWriter*) writer);
}


/*====================================*/
void cvgswStartWriteSeq(int seq_flags, int header_size, int elem_size, CvMemStorage* storage, CvSeqWriter* writer)
{
	cvStartWriteSeq((int) seq_flags, (int) header_size, (int) elem_size, (CvMemStorage*) storage, (CvSeqWriter*) writer);
}


/*====================================*/
CvSeq* cvgswEndWriteSeq(CvSeqWriter* writer)
{
	return cvEndWriteSeq((CvSeqWriter*) writer);
}


/*====================================*/
void cvgswFlushSeqWriter(CvSeqWriter* writer)
{
	cvFlushSeqWriter((CvSeqWriter*) writer);
}


/*====================================*/
void cvgswStartReadSeq(const  CvSeq* seq, CvSeqReader* reader, int reverse)
{
	cvStartReadSeq((const  CvSeq*) seq, (CvSeqReader*) reader, (int) reverse);
}


/*====================================*/
int cvgswGetSeqReaderPos(CvSeqReader* reader)
{
	return cvGetSeqReaderPos((CvSeqReader*) reader);
}


/*====================================*/
void cvgswSetSeqReaderPos(CvSeqReader* reader, int index, int is_relative)
{
	cvSetSeqReaderPos((CvSeqReader*) reader, (int) index, (int) is_relative);
}


/*====================================*/
void* cvgswCvtSeqToArray(const  CvSeq* seq, void* elements, CvSlice slice)
{
	return cvCvtSeqToArray((const  CvSeq*) seq, (void*) elements, (CvSlice) slice);
}


/*====================================*/
CvSeq* cvgswMakeSeqHeaderForArray(int seq_type, int header_size, int elem_size, void* elements, int total, CvSeq* seq, CvSeqBlock* block)
{
	return cvMakeSeqHeaderForArray((int) seq_type, (int) header_size, (int) elem_size, (void*) elements, (int) total, (CvSeq*) seq, (CvSeqBlock*) block);
}


/*====================================*/
CvSeq* cvgswSeqSlice(const  CvSeq* seq, CvSlice slice, CvMemStorage* storage, int copy_data)
{
	return cvSeqSlice((const  CvSeq*) seq, (CvSlice) slice, (CvMemStorage*) storage, (int) copy_data);
}


/*====================================*/
CvSeq* cvgswCloneSeq(const  CvSeq* seq, CvMemStorage* storage)
{
	return cvCloneSeq((const  CvSeq*) seq, (CvMemStorage*) storage);
}


/*====================================*/
void cvgswSeqRemoveSlice(CvSeq* seq, CvSlice slice)
{
	cvSeqRemoveSlice((CvSeq*) seq, (CvSlice) slice);
}


/*====================================*/
void cvgswSeqInsertSlice(CvSeq* seq, int before_index,  CvArr* from_arr)
{
	typedef void(*_SeqInsertSlice) (CvSeq*, int,  CvArr* ); 
	GPUCV_FUNCNAME("cvSeqInsertSlice");
	CvArr* SrcARR[] = { (CvArr*) from_arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvSeq*) seq, (int) before_index, ( CvArr*) from_arr), _SeqInsertSlice, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSeqSort(CvSeq* seq, CvCmpFunc func, void* userdata)
{
	cvSeqSort((CvSeq*) seq, (CvCmpFunc) func, (void*) userdata);
}


/*====================================*/
CVAPI(schar*) cvgswSeqSearch(CvSeq* seq, const  void* elem, CvCmpFunc func, int is_sorted, int* elem_idx, void* userdata)
{
	return cvSeqSearch((CvSeq*) seq, (const  void*) elem, (CvCmpFunc) func, (int) is_sorted, (int*) elem_idx, (void*) userdata);
}


/*====================================*/
void cvgswSeqInvert(CvSeq* seq)
{
	cvSeqInvert((CvSeq*) seq);
}


/*====================================*/
int cvgswSeqPartition(const  CvSeq* seq, CvMemStorage* storage, CvSeq** labels, CvCmpFunc is_equal, void* userdata)
{
	return cvSeqPartition((const  CvSeq*) seq, (CvMemStorage*) storage, (CvSeq**) labels, (CvCmpFunc) is_equal, (void*) userdata);
}


/*====================================*/
void cvgswChangeSeqBlock(void* reader, int direction)
{
	cvChangeSeqBlock((void*) reader, (int) direction);
}


/*====================================*/
void cvgswCreateSeqBlock(CvSeqWriter* writer)
{
	cvCreateSeqBlock((CvSeqWriter*) writer);
}


/*====================================*/
CvSet* cvgswCreateSet(int set_flags, int header_size, int elem_size, CvMemStorage* storage)
{
	return cvCreateSet((int) set_flags, (int) header_size, (int) elem_size, (CvMemStorage*) storage);
}


/*====================================*/
int cvgswSetAdd(CvSet* set_header, CvSetElem* elem, CvSetElem** inserted_elem)
{
	return cvSetAdd((CvSet*) set_header, (CvSetElem*) elem, (CvSetElem**) inserted_elem);
}


/*====================================*/
CvSetElem* cvgswSetNew(CvSet* set_header)
{
	return cvSetNew((CvSet*) set_header);
}


/*====================================*/
void cvgswSetRemoveByPtr(CvSet* set_header, void* elem)
{
	cvSetRemoveByPtr((CvSet*) set_header, (void*) elem);
}


/*====================================*/
void cvgswSetRemove(CvSet* set_header, int index)
{
	cvSetRemove((CvSet*) set_header, (int) index);
}


/*====================================*/
CvSetElem* cvgswGetSetElem(const  CvSet* set_header, int index)
{
	return cvGetSetElem((const  CvSet*) set_header, (int) index);
}


/*====================================*/
void cvgswClearSet(CvSet* set_header)
{
	cvClearSet((CvSet*) set_header);
}


/*====================================*/
CvGraph* cvgswCreateGraph(int graph_flags, int header_size, int vtx_size, int edge_size, CvMemStorage* storage)
{
	return cvCreateGraph((int) graph_flags, (int) header_size, (int) vtx_size, (int) edge_size, (CvMemStorage*) storage);
}


/*====================================*/
int cvgswGraphAddVtx(CvGraph* graph, const  CvGraphVtx* vtx, CvGraphVtx** inserted_vtx)
{
	return cvGraphAddVtx((CvGraph*) graph, (const  CvGraphVtx*) vtx, (CvGraphVtx**) inserted_vtx);
}


/*====================================*/
int cvgswGraphRemoveVtx(CvGraph* graph, int index)
{
	return cvGraphRemoveVtx((CvGraph*) graph, (int) index);
}


/*====================================*/
int cvgswGraphRemoveVtxByPtr(CvGraph* graph, CvGraphVtx* vtx)
{
	return cvGraphRemoveVtxByPtr((CvGraph*) graph, (CvGraphVtx*) vtx);
}


/*====================================*/
int cvgswGraphAddEdge(CvGraph* graph, int start_idx, int end_idx, const  CvGraphEdge* edge, CvGraphEdge** inserted_edge)
{
	return cvGraphAddEdge((CvGraph*) graph, (int) start_idx, (int) end_idx, (const  CvGraphEdge*) edge, (CvGraphEdge**) inserted_edge);
}


/*====================================*/
int cvgswGraphAddEdgeByPtr(CvGraph* graph, CvGraphVtx* start_vtx, CvGraphVtx* end_vtx, const  CvGraphEdge* edge, CvGraphEdge** inserted_edge)
{
	return cvGraphAddEdgeByPtr((CvGraph*) graph, (CvGraphVtx*) start_vtx, (CvGraphVtx*) end_vtx, (const  CvGraphEdge*) edge, (CvGraphEdge**) inserted_edge);
}


/*====================================*/
void cvgswGraphRemoveEdge(CvGraph* graph, int start_idx, int end_idx)
{
	cvGraphRemoveEdge((CvGraph*) graph, (int) start_idx, (int) end_idx);
}


/*====================================*/
void cvgswGraphRemoveEdgeByPtr(CvGraph* graph, CvGraphVtx* start_vtx, CvGraphVtx* end_vtx)
{
	cvGraphRemoveEdgeByPtr((CvGraph*) graph, (CvGraphVtx*) start_vtx, (CvGraphVtx*) end_vtx);
}


/*====================================*/
CvGraphEdge* cvgswFindGraphEdge(const  CvGraph* graph, int start_idx, int end_idx)
{
	return cvFindGraphEdge((const  CvGraph*) graph, (int) start_idx, (int) end_idx);
}


/*====================================*/
CvGraphEdge* cvgswFindGraphEdgeByPtr(const  CvGraph* graph, const  CvGraphVtx* start_vtx, const  CvGraphVtx* end_vtx)
{
	return cvFindGraphEdgeByPtr((const  CvGraph*) graph, (const  CvGraphVtx*) start_vtx, (const  CvGraphVtx*) end_vtx);
}


/*====================================*/
void cvgswClearGraph(CvGraph* graph)
{
	cvClearGraph((CvGraph*) graph);
}


/*====================================*/
int cvgswGraphVtxDegree(const  CvGraph* graph, int vtx_idx)
{
	return cvGraphVtxDegree((const  CvGraph*) graph, (int) vtx_idx);
}


/*====================================*/
int cvgswGraphVtxDegreeByPtr(const  CvGraph* graph, const  CvGraphVtx* vtx)
{
	return cvGraphVtxDegreeByPtr((const  CvGraph*) graph, (const  CvGraphVtx*) vtx);
}


/*====================================*/
CvGraphScanner* cvgswCreateGraphScanner(CvGraph* graph, CvGraphVtx* vtx, int mask)
{
	return cvCreateGraphScanner((CvGraph*) graph, (CvGraphVtx*) vtx, (int) mask);
}


/*====================================*/
void cvgswReleaseGraphScanner(CvGraphScanner** scanner)
{
	cvReleaseGraphScanner((CvGraphScanner**) scanner);
}


/*====================================*/
int cvgswNextGraphItem(CvGraphScanner* scanner)
{
	return cvNextGraphItem((CvGraphScanner*) scanner);
}


/*====================================*/
CvGraph* cvgswCloneGraph(const  CvGraph* graph, CvMemStorage* storage)
{
	return cvCloneGraph((const  CvGraph*) graph, (CvMemStorage*) storage);
}


/*====================================*/
void cvgswLine(CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness, int line_type, int shift)
{
	cvLine((CvArr*) img, (CvPoint) pt1, (CvPoint) pt2, (CvScalar) color, (int) thickness, (int) line_type, (int) shift);
}


/*====================================*/
void cvgswRectangle(CvArr* img, CvPoint pt1, CvPoint pt2, CvScalar color, int thickness, int line_type, int shift)
{
	cvRectangle((CvArr*) img, (CvPoint) pt1, (CvPoint) pt2, (CvScalar) color, (int) thickness, (int) line_type, (int) shift);
}


/*====================================*/
void cvgswRectangleR(CvArr* img, CvRect r, CvScalar color, int thickness, int line_type, int shift)
{
	cvRectangleR((CvArr*) img, (CvRect) r, (CvScalar) color, (int) thickness, (int) line_type, (int) shift);
}


/*====================================*/
void cvgswCircle(CvArr* img, CvPoint center, int radius, CvScalar color, int thickness, int line_type, int shift)
{
	cvCircle((CvArr*) img, (CvPoint) center, (int) radius, (CvScalar) color, (int) thickness, (int) line_type, (int) shift);
}


/*====================================*/
void cvgswEllipse(CvArr* img, CvPoint center, CvSize axes, double angle, double start_angle, double end_angle, CvScalar color, int thickness, int line_type, int shift)
{
	cvEllipse((CvArr*) img, (CvPoint) center, (CvSize) axes, (double) angle, (double) start_angle, (double) end_angle, (CvScalar) color, (int) thickness, (int) line_type, (int) shift);
}


/*====================================*/
void cvgswEllipseBox(CvArr* img, CvBox2D box, CvScalar color, int thickness, int line_type, int shift)
{
	cvEllipseBox((CvArr*) img, (CvBox2D) box, (CvScalar) color, (int) thickness, (int) line_type, (int) shift);
}


/*====================================*/
void cvgswFillConvexPoly(CvArr* img, const  CvPoint* pts, int npts, CvScalar color, int line_type, int shift)
{
	cvFillConvexPoly((CvArr*) img, (const  CvPoint*) pts, (int) npts, (CvScalar) color, (int) line_type, (int) shift);
}


/*====================================*/
void cvgswFillPoly(CvArr* img, CvPoint** pts, const  int* npts, int contours, CvScalar color, int line_type, int shift)
{
	cvFillPoly((CvArr*) img, (CvPoint**) pts, (const  int*) npts, (int) contours, (CvScalar) color, (int) line_type, (int) shift);
}


/*====================================*/
void cvgswPolyLine(CvArr* img, CvPoint** pts, const  int* npts, int contours, int is_closed, CvScalar color, int thickness, int line_type, int shift)
{
	cvPolyLine((CvArr*) img, (CvPoint**) pts, (const  int*) npts, (int) contours, (int) is_closed, (CvScalar) color, (int) thickness, (int) line_type, (int) shift);
}


/*====================================*/
int cvgswClipLine(CvSize img_size, CvPoint* pt1, CvPoint* pt2)
{
	return cvClipLine((CvSize) img_size, (CvPoint*) pt1, (CvPoint*) pt2);
}


/*====================================*/
int cvgswInitLineIterator( CvArr* image, CvPoint pt1, CvPoint pt2, CvLineIterator* line_iterator, int connectivity, int left_to_right)
{
	typedef int(*_InitLineIterator) ( CvArr*, CvPoint, CvPoint, CvLineIterator*, int, int ); 
	GPUCV_FUNCNAME("cvInitLineIterator");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) image, (CvPoint) pt1, (CvPoint) pt2, (CvLineIterator*) line_iterator, (int) connectivity, (int) left_to_right), _InitLineIterator,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswInitFont(CvFont* font, int font_face, double hscale, double vscale, double shear, int thickness, int line_type)
{
	cvInitFont((CvFont*) font, (int) font_face, (double) hscale, (double) vscale, (double) shear, (int) thickness, (int) line_type);
}


/*====================================*/
CvFont cvgswFont(double scale, int thickness)
{
	return cvFont((double) scale, (int) thickness);
}


/*====================================*/
void cvgswPutText(CvArr* img, const  char* text, CvPoint org, const  CvFont* font, CvScalar color)
{
	cvPutText((CvArr*) img, (const  char*) text, (CvPoint) org, (const  CvFont*) font, (CvScalar) color);
}


/*====================================*/
void cvgswGetTextSize(const  char* text_string, const  CvFont* font, CvSize* text_size, int* baseline)
{
	cvGetTextSize((const  char*) text_string, (const  CvFont*) font, (CvSize*) text_size, (int*) baseline);
}


/*====================================*/
CvScalar cvgswColorToScalar(double packed_color, int arrtype)
{
	return cvColorToScalar((double) packed_color, (int) arrtype);
}


/*====================================*/
int cvgswEllipse2Poly(CvPoint center, CvSize axes, int angle, int arc_start, int arc_end, CvPoint * pts, int delta)
{
	return cvEllipse2Poly((CvPoint) center, (CvSize) axes, (int) angle, (int) arc_start, (int) arc_end, (CvPoint *) pts, (int) delta);
}


/*====================================*/
void cvgswDrawContours(CvArr * img, CvSeq* contour, CvScalar external_color, CvScalar hole_color, int max_level, int thickness, int line_type, CvPoint offset)
{
	cvDrawContours((CvArr *) img, (CvSeq*) contour, (CvScalar) external_color, (CvScalar) hole_color, (int) max_level, (int) thickness, (int) line_type, (CvPoint) offset);
}


/*====================================*/
void cvgswLUT( CvArr* src, CvArr* dst,  CvArr* lut)
{
	typedef void(*_LUT) ( CvArr*, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvLUT");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) lut};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, ( CvArr*) lut), _LUT, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswInitTreeNodeIterator(CvTreeNodeIterator* tree_iterator, const  void* first, int max_level)
{
	cvInitTreeNodeIterator((CvTreeNodeIterator*) tree_iterator, (const  void*) first, (int) max_level);
}


/*====================================*/
void* cvgswNextTreeNode(CvTreeNodeIterator* tree_iterator)
{
	return cvNextTreeNode((CvTreeNodeIterator*) tree_iterator);
}


/*====================================*/
void* cvgswPrevTreeNode(CvTreeNodeIterator* tree_iterator)
{
	return cvPrevTreeNode((CvTreeNodeIterator*) tree_iterator);
}


/*====================================*/
void cvgswInsertNodeIntoTree(void* node, void* parent, void* frame)
{
	cvInsertNodeIntoTree((void*) node, (void*) parent, (void*) frame);
}


/*====================================*/
void cvgswRemoveNodeFromTree(void* node, void* frame)
{
	cvRemoveNodeFromTree((void*) node, (void*) frame);
}


/*====================================*/
CvSeq* cvgswTreeToNodeSeq(const  void* first, int header_size, CvMemStorage* storage)
{
	return cvTreeToNodeSeq((const  void*) first, (int) header_size, (CvMemStorage*) storage);
}


/*====================================*/
int cvgswKMeans2( CvArr* samples, int cluster_count, CvArr* labels, CvTermCriteria termcrit, int attempts, CvRNG* rng, int flags, CvArr* _centers, double* compactness)
{
	typedef int(*_KMeans2) ( CvArr*, int, CvArr*, CvTermCriteria, int, CvRNG*, int, CvArr*, double* ); 
	GPUCV_FUNCNAME("cvKMeans2");
	CvArr* SrcARR[] = { (CvArr*) samples};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&labels)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&_centers)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) samples, (int) cluster_count, (CvArr*) labels, (CvTermCriteria) termcrit, (int) attempts, (CvRNG*) rng, (int) flags, (CvArr*) _centers, (double*) compactness), _KMeans2,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
int cvgswRegisterModule(const  CvModuleInfo* module_info)
{
	return cvRegisterModule((const  CvModuleInfo*) module_info);
}


/*====================================*/
int cvgswUseOptimized(int on_off)
{
	return cvUseOptimized((int) on_off);
}


/*====================================*/
void cvgswGetModuleInfo(const  char* module_name, const  char** version, const  char** loaded_addon_plugins)
{
	cvGetModuleInfo((const  char*) module_name, (const  char**) version, (const  char**) loaded_addon_plugins);
}


/*====================================*/
int cvgswGetErrStatus()
{
	return cvGetErrStatus();
}


/*====================================*/
void cvgswSetErrStatus(int status)
{
	cvSetErrStatus((int) status);
}


/*====================================*/
int cvgswGetErrMode()
{
	return cvGetErrMode();
}


/*====================================*/
int cvgswSetErrMode(int mode)
{
	return cvSetErrMode((int) mode);
}


/*====================================*/
void cvgswError(int status, const  char* func_name, const  char* err_msg, const  char* file_name, int line)
{
	cvError((int) status, (const  char*) func_name, (const  char*) err_msg, (const  char*) file_name, (int) line);
}


/*====================================*/
const char* cvgswErrorStr(int status)
{
	return cvErrorStr((int) status);
}


/*====================================*/
int cvgswGetErrInfo(const  char** errcode_desc, const  char** description, const  char** filename, int* line)
{
	return cvGetErrInfo((const  char**) errcode_desc, (const  char**) description, (const  char**) filename, (int*) line);
}


/*====================================*/
int cvgswErrorFromIppStatus(int ipp_status)
{
	return cvErrorFromIppStatus((int) ipp_status);
}


/*====================================*/
CvErrorCallback cvgswRedirectError(CvErrorCallback error_handler, void* userdata, void** prev_userdata)
{
	return cvRedirectError((CvErrorCallback) error_handler, (void*) userdata, (void**) prev_userdata);
}


/*====================================*/
int cvgswNulDevReport(int status, const  char* func_name, const  char* err_msg, const  char* file_name, int line, void* userdata)
{
	return cvNulDevReport((int) status, (const  char*) func_name, (const  char*) err_msg, (const  char*) file_name, (int) line, (void*) userdata);
}


/*====================================*/
int cvgswStdErrReport(int status, const  char* func_name, const  char* err_msg, const  char* file_name, int line, void* userdata)
{
	return cvStdErrReport((int) status, (const  char*) func_name, (const  char*) err_msg, (const  char*) file_name, (int) line, (void*) userdata);
}


/*====================================*/
int cvgswGuiBoxReport(int status, const  char* func_name, const  char* err_msg, const  char* file_name, int line, void* userdata)
{
	return cvGuiBoxReport((int) status, (const  char*) func_name, (const  char*) err_msg, (const  char*) file_name, (int) line, (void*) userdata);
}


/*====================================*/
void cvgswSetMemoryManager(CvAllocFunc alloc_func, CvFreeFunc free_func, void* userdata)
{
	cvSetMemoryManager((CvAllocFunc) alloc_func, (CvFreeFunc) free_func, (void*) userdata);
}


/*====================================*/
void cvgswSetIPLAllocators(Cv_iplCreateImageHeader create_header, Cv_iplAllocateImageData allocate_data, Cv_iplDeallocate deallocate, Cv_iplCreateROI create_roi, Cv_iplCloneImage clone_image)
{
	cvSetIPLAllocators((Cv_iplCreateImageHeader) create_header, (Cv_iplAllocateImageData) allocate_data, (Cv_iplDeallocate) deallocate, (Cv_iplCreateROI) create_roi, (Cv_iplCloneImage) clone_image);
}


/*====================================*/
CvFileStorage* cvgswOpenFileStorage(const  char* filename, CvMemStorage* memstorage, int flags)
{
	return cvOpenFileStorage((const  char*) filename, (CvMemStorage*) memstorage, (int) flags);
}


/*====================================*/
void cvgswReleaseFileStorage(CvFileStorage** fs)
{
	cvReleaseFileStorage((CvFileStorage**) fs);
}


/*====================================*/
const char* cvgswAttrValue(const  CvAttrList* attr, const  char* attr_name)
{
	return cvAttrValue((const  CvAttrList*) attr, (const  char*) attr_name);
}


/*====================================*/
void cvgswStartWriteStruct(CvFileStorage* fs, const  char* name, int struct_flags, const  char* type_name, CvAttrList attributes)
{
	cvStartWriteStruct((CvFileStorage*) fs, (const  char*) name, (int) struct_flags, (const  char*) type_name, (CvAttrList) attributes);
}


/*====================================*/
void cvgswEndWriteStruct(CvFileStorage* fs)
{
	cvEndWriteStruct((CvFileStorage*) fs);
}


/*====================================*/
void cvgswWriteInt(CvFileStorage* fs, const  char* name, int value)
{
	cvWriteInt((CvFileStorage*) fs, (const  char*) name, (int) value);
}


/*====================================*/
void cvgswWriteReal(CvFileStorage* fs, const  char* name, double value)
{
	cvWriteReal((CvFileStorage*) fs, (const  char*) name, (double) value);
}


/*====================================*/
void cvgswWriteString(CvFileStorage* fs, const  char* name, const  char* str, int quote)
{
	cvWriteString((CvFileStorage*) fs, (const  char*) name, (const  char*) str, (int) quote);
}


/*====================================*/
void cvgswWriteComment(CvFileStorage* fs, const  char* comment, int eol_comment)
{
	cvWriteComment((CvFileStorage*) fs, (const  char*) comment, (int) eol_comment);
}


/*====================================*/
void cvgswWrite(CvFileStorage* fs, const  char* name, const  void* ptr, CvAttrList attributes)
{
	cvWrite((CvFileStorage*) fs, (const  char*) name, (const  void*) ptr, (CvAttrList) attributes);
}


/*====================================*/
void cvgswStartNextStream(CvFileStorage* fs)
{
	cvStartNextStream((CvFileStorage*) fs);
}


/*====================================*/
void cvgswWriteRawData(CvFileStorage* fs, const  void* src, int len, const  char* dt)
{
	cvWriteRawData((CvFileStorage*) fs, (const  void*) src, (int) len, (const  char*) dt);
}


/*====================================*/
CvStringHashNode* cvgswGetHashedKey(CvFileStorage* fs, const  char* name, int len, int create_missing)
{
	return cvGetHashedKey((CvFileStorage*) fs, (const  char*) name, (int) len, (int) create_missing);
}


/*====================================*/
CvFileNode* cvgswGetRootFileNode(const  CvFileStorage* fs, int stream_index)
{
	return cvGetRootFileNode((const  CvFileStorage*) fs, (int) stream_index);
}


/*====================================*/
CvFileNode* cvgswGetFileNode(CvFileStorage* fs, CvFileNode* map, const  CvStringHashNode* key, int create_missing)
{
	return cvGetFileNode((CvFileStorage*) fs, (CvFileNode*) map, (const  CvStringHashNode*) key, (int) create_missing);
}


/*====================================*/
CvFileNode* cvgswGetFileNodeByName(const  CvFileStorage* fs, const  CvFileNode* map, const  char* name)
{
	return cvGetFileNodeByName((const  CvFileStorage*) fs, (const  CvFileNode*) map, (const  char*) name);
}


/*====================================*/
int cvgswReadInt(const  CvFileNode* node, int default_value)
{
	return cvReadInt((const  CvFileNode*) node, (int) default_value);
}


/*====================================*/
int cvgswReadIntByName(const  CvFileStorage* fs, const  CvFileNode* map, const  char* name, int default_value)
{
	return cvReadIntByName((const  CvFileStorage*) fs, (const  CvFileNode*) map, (const  char*) name, (int) default_value);
}


/*====================================*/
double cvgswReadReal(const  CvFileNode* node, double default_value)
{
	return cvReadReal((const  CvFileNode*) node, (double) default_value);
}


/*====================================*/
double cvgswReadRealByName(const  CvFileStorage* fs, const  CvFileNode* map, const  char* name, double default_value)
{
	return cvReadRealByName((const  CvFileStorage*) fs, (const  CvFileNode*) map, (const  char*) name, (double) default_value);
}


/*====================================*/
const char* cvgswReadString(const  CvFileNode* node, const  char* default_value)
{
	return cvReadString((const  CvFileNode*) node, (const  char*) default_value);
}


/*====================================*/
const char* cvgswReadStringByName(const  CvFileStorage* fs, const  CvFileNode* map, const  char* name, const  char* default_value)
{
	return cvReadStringByName((const  CvFileStorage*) fs, (const  CvFileNode*) map, (const  char*) name, (const  char*) default_value);
}


/*====================================*/
void* cvgswRead(CvFileStorage* fs, CvFileNode* node, CvAttrList* attributes)
{
	return cvRead((CvFileStorage*) fs, (CvFileNode*) node, (CvAttrList*) attributes);
}


/*====================================*/
void* cvgswReadByName(CvFileStorage* fs, const  CvFileNode* map, const  char* name, CvAttrList* attributes)
{
	return cvReadByName((CvFileStorage*) fs, (const  CvFileNode*) map, (const  char*) name, (CvAttrList*) attributes);
}


/*====================================*/
void cvgswStartReadRawData(const  CvFileStorage* fs, const  CvFileNode* src, CvSeqReader* reader)
{
	cvStartReadRawData((const  CvFileStorage*) fs, (const  CvFileNode*) src, (CvSeqReader*) reader);
}


/*====================================*/
void cvgswReadRawDataSlice(const  CvFileStorage* fs, CvSeqReader* reader, int count, void* dst, const  char* dt)
{
	cvReadRawDataSlice((const  CvFileStorage*) fs, (CvSeqReader*) reader, (int) count, (void*) dst, (const  char*) dt);
}


/*====================================*/
void cvgswReadRawData(const  CvFileStorage* fs, const  CvFileNode* src, void* dst, const  char* dt)
{
	cvReadRawData((const  CvFileStorage*) fs, (const  CvFileNode*) src, (void*) dst, (const  char*) dt);
}


/*====================================*/
void cvgswWriteFileNode(CvFileStorage* fs, const  char* new_node_name, const  CvFileNode* node, int embed)
{
	cvWriteFileNode((CvFileStorage*) fs, (const  char*) new_node_name, (const  CvFileNode*) node, (int) embed);
}


/*====================================*/
const char* cvgswGetFileNodeName(const  CvFileNode* node)
{
	return cvGetFileNodeName((const  CvFileNode*) node);
}


/*====================================*/
void cvgswRegisterType(const  CvTypeInfo* info)
{
	cvRegisterType((const  CvTypeInfo*) info);
}


/*====================================*/
void cvgswUnregisterType(const  char* type_name)
{
	cvUnregisterType((const  char*) type_name);
}


/*====================================*/
CvTypeInfo* cvgswFirstType()
{
	return cvFirstType();
}


/*====================================*/
CvTypeInfo* cvgswFindType(const  char* type_name)
{
	return cvFindType((const  char*) type_name);
}


/*====================================*/
CvTypeInfo* cvgswTypeOf(const  void* struct_ptr)
{
	return cvTypeOf((const  void*) struct_ptr);
}


/*====================================*/
void cvgswRelease(void** struct_ptr)
{
	cvRelease((void**) struct_ptr);
}


/*====================================*/
void* cvgswClone(const  void* struct_ptr)
{
	return cvClone((const  void*) struct_ptr);
}


/*====================================*/
void cvgswSave(const  char* filename, const  void* struct_ptr, const  char* name, const  char* comment, CvAttrList attributes)
{
	cvSave((const  char*) filename, (const  void*) struct_ptr, (const  char*) name, (const  char*) comment, (CvAttrList) attributes);
}


/*====================================*/
void* cvgswLoad(const  char* filename, CvMemStorage* memstorage, const  char* name, const  char** real_name)
{
	return cvLoad((const  char*) filename, (CvMemStorage*) memstorage, (const  char*) name, (const  char**) real_name);
}


/*====================================*/
int64 cvgswGetTickCount()
{
	return cvGetTickCount();
}


/*====================================*/
double cvgswGetTickFrequency()
{
	return cvGetTickFrequency();
}


/*====================================*/
int cvgswCheckHardwareSupport(int feature)
{
	return cvCheckHardwareSupport((int) feature);
}


/*====================================*/
int cvgswGetNumThreads()
{
	return cvGetNumThreads();
}


/*====================================*/
void cvgswSetNumThreads(int threads)
{
	cvSetNumThreads((int) threads);
}


/*====================================*/
int cvgswGetThreadNum()
{
	return cvGetThreadNum();
}

/*........End Code.............*/

