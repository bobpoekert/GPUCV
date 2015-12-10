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

#include <cv_switch/cv_switch.h>
#include <GPUCVSwitch/switch.h>
/*====================================*/
void cvg_cv_switch_RegisterTracerSingletons(SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> * _pAppliTracer, SG_TRC::CL_TRACING_EVENT_LIST *_pEventList)
{
SG_TRC::TTCL_APPLI_TRACER<SG_TRC::SG_TRC_Default_Trc_Type>::Instance().RegisterNewSingleton(_pAppliTracer);
SG_TRC::CL_TRACING_EVENT_LIST::Instance().RegisterNewSingleton(_pEventList);
}
/*====================================*/

/*====================================*/
void cvgswCopyMakeBorder( CvArr* src, CvArr* dst, CvPoint offset, int bordertype, CvScalar value)
{
	typedef void(*_CopyMakeBorder) ( CvArr*, CvArr*, CvPoint, int, CvScalar ); 
	GPUCV_FUNCNAME("cvCopyMakeBorder");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (CvPoint) offset, (int) bordertype, (CvScalar) value), _CopyMakeBorder, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSmooth( CvArr* src, CvArr* dst, int smoothtype, int size1, int size2, double sigma1, double sigma2)
{
	typedef void(*_Smooth) ( CvArr*, CvArr*, int, int, int, double, double ); 
	GPUCV_FUNCNAME("cvSmooth");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) smoothtype, (int) size1, (int) size2, (double) sigma1, (double) sigma2), _Smooth, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswFilter2D( CvArr* src, CvArr* dst,  CvMat* kernel, CvPoint anchor)
{
	typedef void(*_Filter2D) ( CvArr*, CvArr*,  CvMat*, CvPoint ); 
	GPUCV_FUNCNAME("cvFilter2D");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) kernel};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, ( CvMat*) kernel, (CvPoint) anchor), _Filter2D, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswIntegral( CvArr* image, CvArr* sum, CvArr* sqsum, CvArr* tilted_sum)
{
	typedef void(*_Integral) ( CvArr*, CvArr*, CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvIntegral");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr* DstARR[] = { (CvArr*) sum,  (CvArr*) sqsum,  (CvArr*) tilted_sum};
	SWITCH_START_OPR(sum); 
	RUNOP((( CvArr*) image, (CvArr*) sum, (CvArr*) sqsum, (CvArr*) tilted_sum), _Integral, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswPyrDown( CvArr* src, CvArr* dst, int filter)
{
	typedef void(*_PyrDown) ( CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvPyrDown");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) filter), _PyrDown, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswPyrUp( CvArr* src, CvArr* dst, int filter)
{
	typedef void(*_PyrUp) ( CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvPyrUp");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) filter), _PyrUp, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
CVAPI(CvMat**) cvgswCreatePyramid( CvArr* img, int extra_layers, double rate, const  CvSize* layer_sizes, CvArr* bufarr, int calc, int filter)
{
	typedef CVAPI(CvMat**)(*_CreatePyramid) ( CvArr*, int, double, const  CvSize*, CvArr*, int, int ); 
	GPUCV_FUNCNAME("cvCreatePyramid");
	CvArr* SrcARR[] = { (CvArr*) img,  (CvArr*) bufarr};
	CvArr** DstARR = NULL;
	CVAPI(CvMat**) ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) img, (int) extra_layers, (double) rate, (const  CvSize*) layer_sizes, (CvArr*) bufarr, (int) calc, (int) filter), _CreatePyramid,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswReleasePyramid(CvMat*** pyramid, int extra_layers)
{
	typedef void(*_ReleasePyramid) (CvMat***, int ); 
	GPUCV_FUNCNAME("cvReleasePyramid");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvMat***) pyramid, (int) extra_layers), _ReleasePyramid, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswPyrSegmentation(IplImage* src, IplImage* dst, CvMemStorage* storage, CvSeq** comp, int level, double threshold1, double threshold2)
{
	typedef void(*_PyrSegmentation) (IplImage*, IplImage*, CvMemStorage*, CvSeq**, int, double, double ); 
	GPUCV_FUNCNAME("cvPyrSegmentation");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP(((IplImage*) src, (IplImage*) dst, (CvMemStorage*) storage, (CvSeq**) comp, (int) level, (double) threshold1, (double) threshold2), _PyrSegmentation, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswPyrMeanShiftFiltering( CvArr* src, CvArr* dst, double sp, double sr, int max_level, CvTermCriteria termcrit)
{
	typedef void(*_PyrMeanShiftFiltering) ( CvArr*, CvArr*, double, double, int, CvTermCriteria ); 
	GPUCV_FUNCNAME("cvPyrMeanShiftFiltering");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (double) sp, (double) sr, (int) max_level, (CvTermCriteria) termcrit), _PyrMeanShiftFiltering, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswWatershed( CvArr* image, CvArr* markers)
{
	typedef void(*_Watershed) ( CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvWatershed");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr* DstARR[] = { (CvArr*) markers};
	SWITCH_START_OPR(markers); 
	RUNOP((( CvArr*) image, (CvArr*) markers), _Watershed, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswInpaint( CvArr* src,  CvArr* inpaint_mask, CvArr* dst, double inpaintRange, int flags)
{
	typedef void(*_Inpaint) ( CvArr*,  CvArr*, CvArr*, double, int ); 
	GPUCV_FUNCNAME("cvInpaint");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) inpaint_mask};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, ( CvArr*) inpaint_mask, (CvArr*) dst, (double) inpaintRange, (int) flags), _Inpaint, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSobel( CvArr* src, CvArr* dst, int xorder, int yorder, int aperture_size)
{
	typedef void(*_Sobel) ( CvArr*, CvArr*, int, int, int ); 
	GPUCV_FUNCNAME("cvSobel");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) xorder, (int) yorder, (int) aperture_size), _Sobel, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswLaplace( CvArr* src, CvArr* dst, int aperture_size)
{
	typedef void(*_Laplace) ( CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvLaplace");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) aperture_size), _Laplace, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCvtColor( CvArr* src, CvArr* dst, int code)
{
	typedef void(*_CvtColor) ( CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvCvtColor");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) code), _CvtColor, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswResize( CvArr* src, CvArr* dst, int interpolation)
{
	typedef void(*_Resize) ( CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvResize");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) interpolation), _Resize, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswWarpAffine( CvArr* src, CvArr* dst,  CvMat* map_matrix, int flags, CvScalar fillval)
{
	typedef void(*_WarpAffine) ( CvArr*, CvArr*,  CvMat*, int, CvScalar ); 
	GPUCV_FUNCNAME("cvWarpAffine");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) map_matrix};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, ( CvMat*) map_matrix, (int) flags, (CvScalar) fillval), _WarpAffine, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
CvMat* cvgswGetAffineTransform(const  CvPoint2D32f * src, const  CvPoint2D32f * dst, CvMat * map_matrix)
{
	typedef CvMat*(*_GetAffineTransform) (const  CvPoint2D32f *, const  CvPoint2D32f *, CvMat * ); 
	GPUCV_FUNCNAME("cvGetAffineTransform");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((const  CvPoint2D32f *) src, (const  CvPoint2D32f *) dst, (CvMat *) map_matrix), _GetAffineTransform,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvMat* cvgsw2DRotationMatrix(CvPoint2D32f center, double angle, double scale, CvMat* map_matrix)
{
	typedef CvMat*(*_2DRotationMatrix) (CvPoint2D32f, double, double, CvMat* ); 
	GPUCV_FUNCNAME("cv2DRotationMatrix");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&map_matrix)paramsobj->AddParam("option", "MASK");
	RUNOP(((CvPoint2D32f) center, (double) angle, (double) scale, (CvMat*) map_matrix), _2DRotationMatrix,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswWarpPerspective( CvArr* src, CvArr* dst,  CvMat* map_matrix, int flags, CvScalar fillval)
{
	typedef void(*_WarpPerspective) ( CvArr*, CvArr*,  CvMat*, int, CvScalar ); 
	GPUCV_FUNCNAME("cvWarpPerspective");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) map_matrix};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, ( CvMat*) map_matrix, (int) flags, (CvScalar) fillval), _WarpPerspective, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
CvMat* cvgswGetPerspectiveTransform(const  CvPoint2D32f* src, const  CvPoint2D32f* dst, CvMat* map_matrix)
{
	typedef CvMat*(*_GetPerspectiveTransform) (const  CvPoint2D32f*, const  CvPoint2D32f*, CvMat* ); 
	GPUCV_FUNCNAME("cvGetPerspectiveTransform");
	CvArr** SrcARR = NULL;
	CvArr** DstARR = NULL;
	CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&map_matrix)paramsobj->AddParam("option", "MASK");
	RUNOP(((const  CvPoint2D32f*) src, (const  CvPoint2D32f*) dst, (CvMat*) map_matrix), _GetPerspectiveTransform,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswRemap( CvArr* src, CvArr* dst,  CvArr* mapx,  CvArr* mapy, int flags, CvScalar fillval)
{
	typedef void(*_Remap) ( CvArr*, CvArr*,  CvArr*,  CvArr*, int, CvScalar ); 
	GPUCV_FUNCNAME("cvRemap");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) mapx,  (CvArr*) mapy};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, ( CvArr*) mapx, ( CvArr*) mapy, (int) flags, (CvScalar) fillval), _Remap, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswConvertMaps( CvArr* mapx,  CvArr* mapy, CvArr* mapxy, CvArr* mapalpha)
{
	typedef void(*_ConvertMaps) ( CvArr*,  CvArr*, CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvConvertMaps");
	CvArr* SrcARR[] = { (CvArr*) mapx,  (CvArr*) mapy};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mapxy)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&mapalpha)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) mapx, ( CvArr*) mapy, (CvArr*) mapxy, (CvArr*) mapalpha), _ConvertMaps, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswLogPolar( CvArr* src, CvArr* dst, CvPoint2D32f center, double M, int flags)
{
	typedef void(*_LogPolar) ( CvArr*, CvArr*, CvPoint2D32f, double, int ); 
	GPUCV_FUNCNAME("cvLogPolar");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (CvPoint2D32f) center, (double) M, (int) flags), _LogPolar, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswLinearPolar( CvArr* src, CvArr* dst, CvPoint2D32f center, double maxRadius, int flags)
{
	typedef void(*_LinearPolar) ( CvArr*, CvArr*, CvPoint2D32f, double, int ); 
	GPUCV_FUNCNAME("cvLinearPolar");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (CvPoint2D32f) center, (double) maxRadius, (int) flags), _LinearPolar, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
IplConvKernel* cvgswCreateStructuringElementEx(int cols, int rows, int anchor_x, int anchor_y, int shape, int* values)
{
	return cvCreateStructuringElementEx((int) cols, (int) rows, (int) anchor_x, (int) anchor_y, (int) shape, (int*) values);
}


/*====================================*/
void cvgswReleaseStructuringElement(IplConvKernel** element)
{
	cvReleaseStructuringElement((IplConvKernel**) element);
}


/*====================================*/
void cvgswErode( CvArr* src, CvArr* dst, IplConvKernel* element, int iterations)
{
	typedef void(*_Erode) ( CvArr*, CvArr*, IplConvKernel*, int ); 
	GPUCV_FUNCNAME("cvErode");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (IplConvKernel*) element, (int) iterations), _Erode, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswDilate( CvArr* src, CvArr* dst, IplConvKernel* element, int iterations)
{
	typedef void(*_Dilate) ( CvArr*, CvArr*, IplConvKernel*, int ); 
	GPUCV_FUNCNAME("cvDilate");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (IplConvKernel*) element, (int) iterations), _Dilate, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMorphologyEx( CvArr* src, CvArr* dst, CvArr* temp, IplConvKernel* element, int operation, int iterations)
{
	typedef void(*_MorphologyEx) ( CvArr*, CvArr*, CvArr*, IplConvKernel*, int, int ); 
	GPUCV_FUNCNAME("cvMorphologyEx");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&temp)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src, (CvArr*) dst, (CvArr*) temp, (IplConvKernel*) element, (int) operation, (int) iterations), _MorphologyEx, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMoments( CvArr* arr, CvMoments* moments, int binary)
{
	typedef void(*_Moments) ( CvArr*, CvMoments*, int ); 
	GPUCV_FUNCNAME("cvMoments");
	CvArr* SrcARR[] = { (CvArr*) arr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) arr, (CvMoments*) moments, (int) binary), _Moments, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
double cvgswGetSpatialMoment(CvMoments* moments, int x_order, int y_order)
{
	return cvGetSpatialMoment((CvMoments*) moments, (int) x_order, (int) y_order);
}


/*====================================*/
double cvgswGetCentralMoment(CvMoments* moments, int x_order, int y_order)
{
	return cvGetCentralMoment((CvMoments*) moments, (int) x_order, (int) y_order);
}


/*====================================*/
double cvgswGetNormalizedCentralMoment(CvMoments* moments, int x_order, int y_order)
{
	return cvGetNormalizedCentralMoment((CvMoments*) moments, (int) x_order, (int) y_order);
}


/*====================================*/
void cvgswGetHuMoments(CvMoments* moments, CvHuMoments* hu_moments)
{
	cvGetHuMoments((CvMoments*) moments, (CvHuMoments*) hu_moments);
}


/*====================================*/
int cvgswSampleLine( CvArr* image, CvPoint pt1, CvPoint pt2, void* buffer, int connectivity)
{
	typedef int(*_SampleLine) ( CvArr*, CvPoint, CvPoint, void*, int ); 
	GPUCV_FUNCNAME("cvSampleLine");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) image, (CvPoint) pt1, (CvPoint) pt2, (void*) buffer, (int) connectivity), _SampleLine,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswGetRectSubPix( CvArr* src, CvArr* dst, CvPoint2D32f center)
{
	typedef void(*_GetRectSubPix) ( CvArr*, CvArr*, CvPoint2D32f ); 
	GPUCV_FUNCNAME("cvGetRectSubPix");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (CvPoint2D32f) center), _GetRectSubPix, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswGetQuadrangleSubPix( CvArr* src, CvArr* dst,  CvMat* map_matrix)
{
	typedef void(*_GetQuadrangleSubPix) ( CvArr*, CvArr*,  CvMat* ); 
	GPUCV_FUNCNAME("cvGetQuadrangleSubPix");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) map_matrix};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, ( CvMat*) map_matrix), _GetQuadrangleSubPix, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMatchTemplate( CvArr* image,  CvArr* templ, CvArr* result, int method)
{
	typedef void(*_MatchTemplate) ( CvArr*,  CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvMatchTemplate");
	CvArr* SrcARR[] = { (CvArr*) image,  (CvArr*) templ};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&result)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) image, ( CvArr*) templ, (CvArr*) result, (int) method), _MatchTemplate, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
float cvgswCalcEMD2( CvArr* signature1,  CvArr* signature2, int distance_type, CvDistanceFunction distance_func,  CvArr* cost_matrix, CvArr* flow, float* lower_bound, void* userdata)
{
	typedef float(*_CalcEMD2) ( CvArr*,  CvArr*, int, CvDistanceFunction,  CvArr*, CvArr*, float*, void* ); 
	GPUCV_FUNCNAME("cvCalcEMD2");
	CvArr* SrcARR[] = { (CvArr*) signature1,  (CvArr*) signature2,  (CvArr*) cost_matrix};
	CvArr** DstARR = NULL;
	float ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&flow)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) signature1, ( CvArr*) signature2, (int) distance_type, (CvDistanceFunction) distance_func, ( CvArr*) cost_matrix, (CvArr*) flow, (float*) lower_bound, (void*) userdata), _CalcEMD2,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
int cvgswFindContours(CvArr* image, CvMemStorage* storage, CvSeq** first_contour, int header_size, int mode, int method, CvPoint offset)
{
	typedef int(*_FindContours) (CvArr*, CvMemStorage*, CvSeq**, int, int, int, CvPoint ); 
	GPUCV_FUNCNAME("cvFindContours");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) image, (CvMemStorage*) storage, (CvSeq**) first_contour, (int) header_size, (int) mode, (int) method, (CvPoint) offset), _FindContours,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvContourScanner cvgswStartFindContours(CvArr* image, CvMemStorage* storage, int header_size, int mode, int method, CvPoint offset)
{
	typedef CvContourScanner(*_StartFindContours) (CvArr*, CvMemStorage*, int, int, int, CvPoint ); 
	GPUCV_FUNCNAME("cvStartFindContours");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	CvContourScanner ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) image, (CvMemStorage*) storage, (int) header_size, (int) mode, (int) method, (CvPoint) offset), _StartFindContours,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvSeq* cvgswFindNextContour(CvContourScanner scanner)
{
	return cvFindNextContour((CvContourScanner) scanner);
}


/*====================================*/
void cvgswSubstituteContour(CvContourScanner scanner, CvSeq* new_contour)
{
	cvSubstituteContour((CvContourScanner) scanner, (CvSeq*) new_contour);
}


/*====================================*/
CvSeq* cvgswEndFindContours(CvContourScanner* scanner)
{
	return cvEndFindContours((CvContourScanner*) scanner);
}


/*====================================*/
CvSeq* cvgswApproxChains(CvSeq* src_seq, CvMemStorage* storage, int method, double parameter, int minimal_perimeter, int recursive)
{
	return cvApproxChains((CvSeq*) src_seq, (CvMemStorage*) storage, (int) method, (double) parameter, (int) minimal_perimeter, (int) recursive);
}


/*====================================*/
void cvgswStartReadChainPoints(CvChain* chain, CvChainPtReader* reader)
{
	cvStartReadChainPoints((CvChain*) chain, (CvChainPtReader*) reader);
}


/*====================================*/
CvPoint cvgswReadChainPoint(CvChainPtReader* reader)
{
	return cvReadChainPoint((CvChainPtReader*) reader);
}


/*====================================*/
void cvgswCalcOpticalFlowLK( CvArr* prev,  CvArr* curr, CvSize win_size, CvArr* velx, CvArr* vely)
{
	typedef void(*_CalcOpticalFlowLK) ( CvArr*,  CvArr*, CvSize, CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvCalcOpticalFlowLK");
	CvArr* SrcARR[] = { (CvArr*) prev,  (CvArr*) curr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&velx)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&vely)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) prev, ( CvArr*) curr, (CvSize) win_size, (CvArr*) velx, (CvArr*) vely), _CalcOpticalFlowLK, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCalcOpticalFlowBM( CvArr* prev,  CvArr* curr, CvSize block_size, CvSize shift_size, CvSize max_range, int use_previous, CvArr* velx, CvArr* vely)
{
	typedef void(*_CalcOpticalFlowBM) ( CvArr*,  CvArr*, CvSize, CvSize, CvSize, int, CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvCalcOpticalFlowBM");
	CvArr* SrcARR[] = { (CvArr*) prev,  (CvArr*) curr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&velx)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&vely)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) prev, ( CvArr*) curr, (CvSize) block_size, (CvSize) shift_size, (CvSize) max_range, (int) use_previous, (CvArr*) velx, (CvArr*) vely), _CalcOpticalFlowBM, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCalcOpticalFlowHS( CvArr* prev,  CvArr* curr, int use_previous, CvArr* velx, CvArr* vely, double lambda, CvTermCriteria criteria)
{
	typedef void(*_CalcOpticalFlowHS) ( CvArr*,  CvArr*, int, CvArr*, CvArr*, double, CvTermCriteria ); 
	GPUCV_FUNCNAME("cvCalcOpticalFlowHS");
	CvArr* SrcARR[] = { (CvArr*) prev,  (CvArr*) curr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&velx)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&vely)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) prev, ( CvArr*) curr, (int) use_previous, (CvArr*) velx, (CvArr*) vely, (double) lambda, (CvTermCriteria) criteria), _CalcOpticalFlowHS, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCalcOpticalFlowPyrLK( CvArr* prev,  CvArr* curr, CvArr* prev_pyr, CvArr* curr_pyr, const  CvPoint2D32f* prev_features, CvPoint2D32f* curr_features, int count, CvSize win_size, int level, char* status, float* track_error, CvTermCriteria criteria, int flags)
{
	typedef void(*_CalcOpticalFlowPyrLK) ( CvArr*,  CvArr*, CvArr*, CvArr*, const  CvPoint2D32f*, CvPoint2D32f*, int, CvSize, int, char*, float*, CvTermCriteria, int ); 
	GPUCV_FUNCNAME("cvCalcOpticalFlowPyrLK");
	CvArr* SrcARR[] = { (CvArr*) prev,  (CvArr*) curr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&prev_pyr)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&curr_pyr)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) prev, ( CvArr*) curr, (CvArr*) prev_pyr, (CvArr*) curr_pyr, (const  CvPoint2D32f*) prev_features, (CvPoint2D32f*) curr_features, (int) count, (CvSize) win_size, (int) level, (char*) status, (float*) track_error, (CvTermCriteria) criteria, (int) flags), _CalcOpticalFlowPyrLK, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCalcAffineFlowPyrLK( CvArr* prev,  CvArr* curr, CvArr* prev_pyr, CvArr* curr_pyr, const  CvPoint2D32f* prev_features, CvPoint2D32f* curr_features, float* matrices, int count, CvSize win_size, int level, char* status, float* track_error, CvTermCriteria criteria, int flags)
{
	typedef void(*_CalcAffineFlowPyrLK) ( CvArr*,  CvArr*, CvArr*, CvArr*, const  CvPoint2D32f*, CvPoint2D32f*, float*, int, CvSize, int, char*, float*, CvTermCriteria, int ); 
	GPUCV_FUNCNAME("cvCalcAffineFlowPyrLK");
	CvArr* SrcARR[] = { (CvArr*) prev,  (CvArr*) curr};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&prev_pyr)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&curr_pyr)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) prev, ( CvArr*) curr, (CvArr*) prev_pyr, (CvArr*) curr_pyr, (const  CvPoint2D32f*) prev_features, (CvPoint2D32f*) curr_features, (float*) matrices, (int) count, (CvSize) win_size, (int) level, (char*) status, (float*) track_error, (CvTermCriteria) criteria, (int) flags), _CalcAffineFlowPyrLK, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswEstimateRigidTransform( CvArr* A,  CvArr* B, CvMat* M, int full_affine)
{
	typedef int(*_EstimateRigidTransform) ( CvArr*,  CvArr*, CvMat*, int ); 
	GPUCV_FUNCNAME("cvEstimateRigidTransform");
	CvArr* SrcARR[] = { (CvArr*) A,  (CvArr*) B};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&M)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) A, ( CvArr*) B, (CvMat*) M, (int) full_affine), _EstimateRigidTransform,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswCalcOpticalFlowFarneback( CvArr* prev,  CvArr* next, CvArr* flow, double pyr_scale, int levels, int winsize, int iterations, int poly_n, double poly_sigma, int flags)
{
	typedef void(*_CalcOpticalFlowFarneback) ( CvArr*,  CvArr*, CvArr*, double, int, int, int, int, double, int ); 
	GPUCV_FUNCNAME("cvCalcOpticalFlowFarneback");
	CvArr* SrcARR[] = { (CvArr*) prev,  (CvArr*) next};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&flow)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) prev, ( CvArr*) next, (CvArr*) flow, (double) pyr_scale, (int) levels, (int) winsize, (int) iterations, (int) poly_n, (double) poly_sigma, (int) flags), _CalcOpticalFlowFarneback, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswUpdateMotionHistory( CvArr* silhouette, CvArr* mhi, double timestamp, double duration)
{
	typedef void(*_UpdateMotionHistory) ( CvArr*, CvArr*, double, double ); 
	GPUCV_FUNCNAME("cvUpdateMotionHistory");
	CvArr* SrcARR[] = { (CvArr*) silhouette};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mhi)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) silhouette, (CvArr*) mhi, (double) timestamp, (double) duration), _UpdateMotionHistory, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCalcMotionGradient( CvArr* mhi, CvArr* mask, CvArr* orientation, double delta1, double delta2, int aperture_size)
{
	typedef void(*_CalcMotionGradient) ( CvArr*, CvArr*, CvArr*, double, double, int ); 
	GPUCV_FUNCNAME("cvCalcMotionGradient");
	CvArr* SrcARR[] = { (CvArr*) mhi,  (CvArr*) mask};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&orientation)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) mhi, (CvArr*) mask, (CvArr*) orientation, (double) delta1, (double) delta2, (int) aperture_size), _CalcMotionGradient, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
double cvgswCalcGlobalOrientation( CvArr* orientation,  CvArr* mask,  CvArr* mhi, double timestamp, double duration)
{
	typedef double(*_CalcGlobalOrientation) ( CvArr*,  CvArr*,  CvArr*, double, double ); 
	GPUCV_FUNCNAME("cvCalcGlobalOrientation");
	CvArr* SrcARR[] = { (CvArr*) orientation,  (CvArr*) mask,  (CvArr*) mhi};
	CvArr** DstARR = NULL;
	double ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) orientation, ( CvArr*) mask, ( CvArr*) mhi, (double) timestamp, (double) duration), _CalcGlobalOrientation,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvSeq* cvgswSegmentMotion( CvArr* mhi, CvArr* seg_mask, CvMemStorage* storage, double timestamp, double seg_thresh)
{
	typedef CvSeq*(*_SegmentMotion) ( CvArr*, CvArr*, CvMemStorage*, double, double ); 
	GPUCV_FUNCNAME("cvSegmentMotion");
	CvArr* SrcARR[] = { (CvArr*) mhi,  (CvArr*) seg_mask};
	CvArr** DstARR = NULL;
	CvSeq* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) mhi, (CvArr*) seg_mask, (CvMemStorage*) storage, (double) timestamp, (double) seg_thresh), _SegmentMotion,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswAcc( CvArr* image, CvArr* sum,  CvArr* mask)
{
	typedef void(*_Acc) ( CvArr*, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvAcc");
	CvArr* SrcARR[] = { (CvArr*) image,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) sum};
	SWITCH_START_OPR(sum); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) image, (CvArr*) sum, ( CvArr*) mask), _Acc, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSquareAcc( CvArr* image, CvArr* sqsum,  CvArr* mask)
{
	typedef void(*_SquareAcc) ( CvArr*, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvSquareAcc");
	CvArr* SrcARR[] = { (CvArr*) image,  (CvArr*) mask};
	CvArr* DstARR[] = { (CvArr*) sqsum};
	SWITCH_START_OPR(sqsum); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) image, (CvArr*) sqsum, ( CvArr*) mask), _SquareAcc, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswMultiplyAcc( CvArr* image1,  CvArr* image2, CvArr* acc,  CvArr* mask)
{
	typedef void(*_MultiplyAcc) ( CvArr*,  CvArr*, CvArr*,  CvArr* ); 
	GPUCV_FUNCNAME("cvMultiplyAcc");
	CvArr* SrcARR[] = { (CvArr*) image1,  (CvArr*) image2,  (CvArr*) mask};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&acc)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) image1, ( CvArr*) image2, (CvArr*) acc, ( CvArr*) mask), _MultiplyAcc, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswRunningAvg( CvArr* image, CvArr* acc, double alpha,  CvArr* mask)
{
	typedef void(*_RunningAvg) ( CvArr*, CvArr*, double,  CvArr* ); 
	GPUCV_FUNCNAME("cvRunningAvg");
	CvArr* SrcARR[] = { (CvArr*) image,  (CvArr*) mask};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&acc)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) image, (CvArr*) acc, (double) alpha, ( CvArr*) mask), _RunningAvg, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswCamShift( CvArr* prob_image, CvRect window, CvTermCriteria criteria, CvConnectedComp* comp, CvBox2D* box)
{
	typedef int(*_CamShift) ( CvArr*, CvRect, CvTermCriteria, CvConnectedComp*, CvBox2D* ); 
	GPUCV_FUNCNAME("cvCamShift");
	CvArr* SrcARR[] = { (CvArr*) prob_image};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) prob_image, (CvRect) window, (CvTermCriteria) criteria, (CvConnectedComp*) comp, (CvBox2D*) box), _CamShift,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
int cvgswMeanShift( CvArr* prob_image, CvRect window, CvTermCriteria criteria, CvConnectedComp* comp)
{
	typedef int(*_MeanShift) ( CvArr*, CvRect, CvTermCriteria, CvConnectedComp* ); 
	GPUCV_FUNCNAME("cvMeanShift");
	CvArr* SrcARR[] = { (CvArr*) prob_image};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) prob_image, (CvRect) window, (CvTermCriteria) criteria, (CvConnectedComp*) comp), _MeanShift,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvKalman* cvgswCreateKalman(int dynam_params, int measure_params, int control_params)
{
	return cvCreateKalman((int) dynam_params, (int) measure_params, (int) control_params);
}


/*====================================*/
void cvgswReleaseKalman(CvKalman** kalman)
{
	cvReleaseKalman((CvKalman**) kalman);
}


/*====================================*/
const CvMat* cvgswKalmanPredict(CvKalman* kalman,  CvMat* control)
{
	typedef const CvMat*(*_KalmanPredict) (CvKalman*,  CvMat* ); 
	GPUCV_FUNCNAME("cvKalmanPredict");
	CvArr* SrcARR[] = { (CvArr*) control};
	CvArr** DstARR = NULL;
	const CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((CvKalman*) kalman, ( CvMat*) control), _KalmanPredict,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
const CvMat* cvgswKalmanCorrect(CvKalman* kalman,  CvMat* measurement)
{
	typedef const CvMat*(*_KalmanCorrect) (CvKalman*,  CvMat* ); 
	GPUCV_FUNCNAME("cvKalmanCorrect");
	CvArr* SrcARR[] = { (CvArr*) measurement};
	CvArr** DstARR = NULL;
	const CvMat* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((CvKalman*) kalman, ( CvMat*) measurement), _KalmanCorrect,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswInitSubdivDelaunay2D(CvSubdiv2D* subdiv, CvRect rect)
{
	cvInitSubdivDelaunay2D((CvSubdiv2D*) subdiv, (CvRect) rect);
}


/*====================================*/
CvSubdiv2D* cvgswCreateSubdiv2D(int subdiv_type, int header_size, int vtx_size, int quadedge_size, CvMemStorage* storage)
{
	return cvCreateSubdiv2D((int) subdiv_type, (int) header_size, (int) vtx_size, (int) quadedge_size, (CvMemStorage*) storage);
}


/*====================================*/
CvSubdiv2D* cvgswCreateSubdivDelaunay2D(CvRect rect, CvMemStorage* storage)
{
	return cvCreateSubdivDelaunay2D((CvRect) rect, (CvMemStorage*) storage);
}


/*====================================*/
CvSubdiv2DPoint* cvgswSubdivDelaunay2DInsert(CvSubdiv2D* subdiv, CvPoint2D32f pt)
{
	return cvSubdivDelaunay2DInsert((CvSubdiv2D*) subdiv, (CvPoint2D32f) pt);
}


/*====================================*/
CvSubdiv2DPointLocation cvgswSubdiv2DLocate(CvSubdiv2D* subdiv, CvPoint2D32f pt, CvSubdiv2DEdge* edge, CvSubdiv2DPoint** vertex)
{
	return cvSubdiv2DLocate((CvSubdiv2D*) subdiv, (CvPoint2D32f) pt, (CvSubdiv2DEdge*) edge, (CvSubdiv2DPoint**) vertex);
}


/*====================================*/
void cvgswCalcSubdivVoronoi2D(CvSubdiv2D* subdiv)
{
	cvCalcSubdivVoronoi2D((CvSubdiv2D*) subdiv);
}


/*====================================*/
void cvgswClearSubdivVoronoi2D(CvSubdiv2D* subdiv)
{
	cvClearSubdivVoronoi2D((CvSubdiv2D*) subdiv);
}


/*====================================*/
CvSubdiv2DPoint* cvgswFindNearestPoint2D(CvSubdiv2D* subdiv, CvPoint2D32f pt)
{
	return cvFindNearestPoint2D((CvSubdiv2D*) subdiv, (CvPoint2D32f) pt);
}


/*====================================*/
CvSubdiv2DEdge cvgswSubdiv2DNextEdge(CvSubdiv2DEdge edge)
{
	return cvSubdiv2DNextEdge((CvSubdiv2DEdge) edge);
}


/*====================================*/
CvSubdiv2DEdge cvgswSubdiv2DRotateEdge(CvSubdiv2DEdge edge, int rotate)
{
	return cvSubdiv2DRotateEdge((CvSubdiv2DEdge) edge, (int) rotate);
}


/*====================================*/
CvSubdiv2DEdge cvgswSubdiv2DSymEdge(CvSubdiv2DEdge edge)
{
	return cvSubdiv2DSymEdge((CvSubdiv2DEdge) edge);
}


/*====================================*/
CvSubdiv2DEdge cvgswSubdiv2DGetEdge(CvSubdiv2DEdge edge, CvNextEdgeType type)
{
	return cvSubdiv2DGetEdge((CvSubdiv2DEdge) edge, (CvNextEdgeType) type);
}


/*====================================*/
CvSubdiv2DPoint* cvgswSubdiv2DEdgeOrg(CvSubdiv2DEdge edge)
{
	return cvSubdiv2DEdgeOrg((CvSubdiv2DEdge) edge);
}


/*====================================*/
CvSubdiv2DPoint* cvgswSubdiv2DEdgeDst(CvSubdiv2DEdge edge)
{
	return cvSubdiv2DEdgeDst((CvSubdiv2DEdge) edge);
}


/*====================================*/
double cvgswTriangleArea(CvPoint2D32f a, CvPoint2D32f b, CvPoint2D32f c)
{
	return cvTriangleArea((CvPoint2D32f) a, (CvPoint2D32f) b, (CvPoint2D32f) c);
}


/*====================================*/
CvSeq* cvgswApproxPoly(const  void* src_seq, int header_size, CvMemStorage* storage, int method, double parameter, int parameter2)
{
	return cvApproxPoly((const  void*) src_seq, (int) header_size, (CvMemStorage*) storage, (int) method, (double) parameter, (int) parameter2);
}


/*====================================*/
double cvgswArcLength(const  void* curve, CvSlice slice, int is_closed)
{
	return cvArcLength((const  void*) curve, (CvSlice) slice, (int) is_closed);
}


/*====================================*/
CvRect cvgswBoundingRect(CvArr* points, int update)
{
	return cvBoundingRect((CvArr*) points, (int) update);
}


/*====================================*/
double cvgswContourArea( CvArr* contour, CvSlice slice, int oriented)
{
	typedef double(*_ContourArea) ( CvArr*, CvSlice, int ); 
	GPUCV_FUNCNAME("cvContourArea");
	CvArr* SrcARR[] = { (CvArr*) contour};
	CvArr** DstARR = NULL;
	double ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) contour, (CvSlice) slice, (int) oriented), _ContourArea,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvBox2D cvgswMinAreaRect2( CvArr* points, CvMemStorage* storage)
{
	typedef CvBox2D(*_MinAreaRect2) ( CvArr*, CvMemStorage* ); 
	GPUCV_FUNCNAME("cvMinAreaRect2");
	CvArr* SrcARR[] = { (CvArr*) points};
	CvArr** DstARR = NULL;
	CvBox2D ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) points, (CvMemStorage*) storage), _MinAreaRect2,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
int cvgswMinEnclosingCircle( CvArr* points, CvPoint2D32f* center, float* radius)
{
	typedef int(*_MinEnclosingCircle) ( CvArr*, CvPoint2D32f*, float* ); 
	GPUCV_FUNCNAME("cvMinEnclosingCircle");
	CvArr* SrcARR[] = { (CvArr*) points};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) points, (CvPoint2D32f*) center, (float*) radius), _MinEnclosingCircle,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
double cvgswMatchShapes(const  void* object1, const  void* object2, int method, double parameter)
{
	return cvMatchShapes((const  void*) object1, (const  void*) object2, (int) method, (double) parameter);
}


/*====================================*/
CvContourTree* cvgswCreateContourTree(const  CvSeq* contour, CvMemStorage* storage, double threshold)
{
	return cvCreateContourTree((const  CvSeq*) contour, (CvMemStorage*) storage, (double) threshold);
}


/*====================================*/
CvSeq* cvgswContourFromContourTree(const  CvContourTree* tree, CvMemStorage* storage, CvTermCriteria criteria)
{
	return cvContourFromContourTree((const  CvContourTree*) tree, (CvMemStorage*) storage, (CvTermCriteria) criteria);
}


/*====================================*/
double cvgswMatchContourTrees(const  CvContourTree* tree1, const  CvContourTree* tree2, int method, double threshold)
{
	return cvMatchContourTrees((const  CvContourTree*) tree1, (const  CvContourTree*) tree2, (int) method, (double) threshold);
}


/*====================================*/
CvSeq* cvgswConvexHull2( CvArr* input, void* hull_storage, int orientation, int return_points)
{
	typedef CvSeq*(*_ConvexHull2) ( CvArr*, void*, int, int ); 
	GPUCV_FUNCNAME("cvConvexHull2");
	CvArr* SrcARR[] = { (CvArr*) input};
	CvArr** DstARR = NULL;
	CvSeq* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) input, (void*) hull_storage, (int) orientation, (int) return_points), _ConvexHull2,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
int cvgswCheckContourConvexity( CvArr* contour)
{
	typedef int(*_CheckContourConvexity) ( CvArr* ); 
	GPUCV_FUNCNAME("cvCheckContourConvexity");
	CvArr* SrcARR[] = { (CvArr*) contour};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) contour), _CheckContourConvexity,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvSeq* cvgswConvexityDefects( CvArr* contour,  CvArr* convexhull, CvMemStorage* storage)
{
	typedef CvSeq*(*_ConvexityDefects) ( CvArr*,  CvArr*, CvMemStorage* ); 
	GPUCV_FUNCNAME("cvConvexityDefects");
	CvArr* SrcARR[] = { (CvArr*) contour,  (CvArr*) convexhull};
	CvArr** DstARR = NULL;
	CvSeq* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) contour, ( CvArr*) convexhull, (CvMemStorage*) storage), _ConvexityDefects,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvBox2D cvgswFitEllipse2( CvArr* points)
{
	typedef CvBox2D(*_FitEllipse2) ( CvArr* ); 
	GPUCV_FUNCNAME("cvFitEllipse2");
	CvArr* SrcARR[] = { (CvArr*) points};
	CvArr** DstARR = NULL;
	CvBox2D ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) points), _FitEllipse2,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvRect cvgswMaxRect(const  CvRect* rect1, const  CvRect* rect2)
{
	return cvMaxRect((const  CvRect*) rect1, (const  CvRect*) rect2);
}


/*====================================*/
void cvgswBoxPoints(CvBox2D box, CvPoint2D32f *  pt)
{
	cvBoxPoints((CvBox2D) box, (CvPoint2D32f * ) pt);
}


/*====================================*/
CvSeq* cvgswPointSeqFromMat(int seq_kind,  CvArr* mat, CvContour* contour_header, CvSeqBlock* block)
{
	typedef CvSeq*(*_PointSeqFromMat) (int,  CvArr*, CvContour*, CvSeqBlock* ); 
	GPUCV_FUNCNAME("cvPointSeqFromMat");
	CvArr* SrcARR[] = { (CvArr*) mat};
	CvArr** DstARR = NULL;
	CvSeq* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((int) seq_kind, ( CvArr*) mat, (CvContour*) contour_header, (CvSeqBlock*) block), _PointSeqFromMat,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
double cvgswPointPolygonTest( CvArr* contour, CvPoint2D32f pt, int measure_dist)
{
	typedef double(*_PointPolygonTest) ( CvArr*, CvPoint2D32f, int ); 
	GPUCV_FUNCNAME("cvPointPolygonTest");
	CvArr* SrcARR[] = { (CvArr*) contour};
	CvArr** DstARR = NULL;
	double ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) contour, (CvPoint2D32f) pt, (int) measure_dist), _PointPolygonTest,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvHistogram* cvgswCreateHist(int dims, int* sizes, int type, float** ranges, int uniform)
{
	return cvCreateHist((int) dims, (int*) sizes, (int) type, (float**) ranges, (int) uniform);
}


/*====================================*/
void cvgswSetHistBinRanges(CvHistogram* hist, float** ranges, int uniform)
{
	cvSetHistBinRanges((CvHistogram*) hist, (float**) ranges, (int) uniform);
}


/*====================================*/
CvHistogram* cvgswMakeHistHeaderForArray(int dims, int* sizes, CvHistogram* hist, float* data, float** ranges, int uniform)
{
	return cvMakeHistHeaderForArray((int) dims, (int*) sizes, (CvHistogram*) hist, (float*) data, (float**) ranges, (int) uniform);
}


/*====================================*/
void cvgswReleaseHist(CvHistogram** hist)
{
	cvReleaseHist((CvHistogram**) hist);
}


/*====================================*/
void cvgswClearHist(CvHistogram* hist)
{
	cvClearHist((CvHistogram*) hist);
}


/*====================================*/
void cvgswGetMinMaxHistValue(const  CvHistogram* hist, float* min_value, float* max_value, int* min_idx, int* max_idx)
{
	cvGetMinMaxHistValue((const  CvHistogram*) hist, (float*) min_value, (float*) max_value, (int*) min_idx, (int*) max_idx);
}


/*====================================*/
void cvgswNormalizeHist(CvHistogram* hist, double factor)
{
	cvNormalizeHist((CvHistogram*) hist, (double) factor);
}


/*====================================*/
void cvgswThreshHist(CvHistogram* hist, double threshold)
{
	cvThreshHist((CvHistogram*) hist, (double) threshold);
}


/*====================================*/
double cvgswCompareHist(const  CvHistogram* hist1, const  CvHistogram* hist2, int method)
{
	return cvCompareHist((const  CvHistogram*) hist1, (const  CvHistogram*) hist2, (int) method);
}


/*====================================*/
void cvgswCopyHist(const  CvHistogram* src, CvHistogram** dst)
{
	cvCopyHist((const  CvHistogram*) src, (CvHistogram**) dst);
}


/*====================================*/
void cvgswCalcBayesianProb(CvHistogram** src, int number, CvHistogram** dst)
{
	cvCalcBayesianProb((CvHistogram**) src, (int) number, (CvHistogram**) dst);
}


/*====================================*/
void cvgswCalcArrHist(CvArr** arr, CvHistogram* hist, int accumulate,  CvArr* mask)
{
	typedef void(*_CalcArrHist) (CvArr**, CvHistogram*, int,  CvArr* ); 
	GPUCV_FUNCNAME("cvCalcArrHist");
	CvArr* SrcARR[] = { (CvArr*) mask};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP(((CvArr**) arr, (CvHistogram*) hist, (int) accumulate, ( CvArr*) mask), _CalcArrHist, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCalcHist(IplImage** image, CvHistogram* hist, int accumulate,  CvArr* mask)
{
	typedef void(*_CalcHist) (IplImage**, CvHistogram*, int,  CvArr* ); 
	GPUCV_FUNCNAME("cvCalcHist");
	CvArr* SrcARR[] = { (CvArr*) mask};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP(((IplImage**) image, (CvHistogram*) hist, (int) accumulate, ( CvArr*) mask), _CalcHist, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCalcArrBackProject(CvArr** image, CvArr* dst, const  CvHistogram* hist)
{
	typedef void(*_CalcArrBackProject) (CvArr**, CvArr*, const  CvHistogram* ); 
	GPUCV_FUNCNAME("cvCalcArrBackProject");
	CvArr** SrcARR = NULL;
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP(((CvArr**) image, (CvArr*) dst, (const  CvHistogram*) hist), _CalcArrBackProject, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCalcArrBackProjectPatch(CvArr** image, CvArr* dst, CvSize range, CvHistogram* hist, int method, double factor)
{
	typedef void(*_CalcArrBackProjectPatch) (CvArr**, CvArr*, CvSize, CvHistogram*, int, double ); 
	GPUCV_FUNCNAME("cvCalcArrBackProjectPatch");
	CvArr** SrcARR = NULL;
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP(((CvArr**) image, (CvArr*) dst, (CvSize) range, (CvHistogram*) hist, (int) method, (double) factor), _CalcArrBackProjectPatch, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCalcProbDensity(const  CvHistogram* hist1, const  CvHistogram* hist2, CvHistogram* dst_hist, double scale)
{
	cvCalcProbDensity((const  CvHistogram*) hist1, (const  CvHistogram*) hist2, (CvHistogram*) dst_hist, (double) scale);
}


/*====================================*/
void cvgswEqualizeHist( CvArr* src, CvArr* dst)
{
	typedef void(*_EqualizeHist) ( CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvEqualizeHist");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst), _EqualizeHist, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswSnakeImage( IplImage* image, CvPoint* points, int length, float* alpha, float* beta, float* gamma, int coeff_usage, CvSize win, CvTermCriteria criteria, int calc_gradient)
{
	typedef void(*_SnakeImage) ( IplImage*, CvPoint*, int, float*, float*, float*, int, CvSize, CvTermCriteria, int ); 
	GPUCV_FUNCNAME("cvSnakeImage");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP((( IplImage*) image, (CvPoint*) points, (int) length, (float*) alpha, (float*) beta, (float*) gamma, (int) coeff_usage, (CvSize) win, (CvTermCriteria) criteria, (int) calc_gradient), _SnakeImage, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswDistTransform( CvArr* src, CvArr* dst, int distance_type, int mask_size, const  float* mask, CvArr* labels)
{
	typedef void(*_DistTransform) ( CvArr*, CvArr*, int, int, const  float*, CvArr* ); 
	GPUCV_FUNCNAME("cvDistTransform");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&labels)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) src, (CvArr*) dst, (int) distance_type, (int) mask_size, (const  float*) mask, (CvArr*) labels), _DistTransform, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
double cvgswThreshold( CvArr* src, CvArr* dst, double threshold, double max_value, int threshold_type)
{
	typedef double(*_Threshold) ( CvArr*, CvArr*, double, double, int ); 
	GPUCV_FUNCNAME("cvThreshold");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	double ReturnObj;	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (double) threshold, (double) max_value, (int) threshold_type), _Threshold,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswAdaptiveThreshold( CvArr* src, CvArr* dst, double max_value, int adaptive_method, int threshold_type, int block_size, double param1)
{
	typedef void(*_AdaptiveThreshold) ( CvArr*, CvArr*, double, int, int, int, double ); 
	GPUCV_FUNCNAME("cvAdaptiveThreshold");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, (double) max_value, (int) adaptive_method, (int) threshold_type, (int) block_size, (double) param1), _AdaptiveThreshold, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswFloodFill(CvArr* image, CvPoint seed_point, CvScalar new_val, CvScalar lo_diff, CvScalar up_diff, CvConnectedComp* comp, int flags, CvArr* mask)
{
	typedef void(*_FloodFill) (CvArr*, CvPoint, CvScalar, CvScalar, CvScalar, CvConnectedComp*, int, CvArr* ); 
	GPUCV_FUNCNAME("cvFloodFill");
	CvArr* SrcARR[] = { (CvArr*) image,  (CvArr*) mask};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP(((CvArr*) image, (CvPoint) seed_point, (CvScalar) new_val, (CvScalar) lo_diff, (CvScalar) up_diff, (CvConnectedComp*) comp, (int) flags, (CvArr*) mask), _FloodFill, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCanny( CvArr* image, CvArr* edges, double threshold1, double threshold2, int aperture_size)
{
	typedef void(*_Canny) ( CvArr*, CvArr*, double, double, int ); 
	GPUCV_FUNCNAME("cvCanny");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&edges)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) image, (CvArr*) edges, (double) threshold1, (double) threshold2, (int) aperture_size), _Canny, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswPreCornerDetect( CvArr* image, CvArr* corners, int aperture_size)
{
	typedef void(*_PreCornerDetect) ( CvArr*, CvArr*, int ); 
	GPUCV_FUNCNAME("cvPreCornerDetect");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&corners)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) image, (CvArr*) corners, (int) aperture_size), _PreCornerDetect, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCornerEigenValsAndVecs( CvArr* image, CvArr* eigenvv, int block_size, int aperture_size)
{
	typedef void(*_CornerEigenValsAndVecs) ( CvArr*, CvArr*, int, int ); 
	GPUCV_FUNCNAME("cvCornerEigenValsAndVecs");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&eigenvv)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) image, (CvArr*) eigenvv, (int) block_size, (int) aperture_size), _CornerEigenValsAndVecs, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCornerMinEigenVal( CvArr* image, CvArr* eigenval, int block_size, int aperture_size)
{
	typedef void(*_CornerMinEigenVal) ( CvArr*, CvArr*, int, int ); 
	GPUCV_FUNCNAME("cvCornerMinEigenVal");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&eigenval)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) image, (CvArr*) eigenval, (int) block_size, (int) aperture_size), _CornerMinEigenVal, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswCornerHarris( CvArr* image, CvArr* harris_responce, int block_size, int aperture_size, double k)
{
	typedef void(*_CornerHarris) ( CvArr*, CvArr*, int, int, double ); 
	GPUCV_FUNCNAME("cvCornerHarris");
	CvArr* SrcARR[] = { (CvArr*) image,  (CvArr*) harris_responce};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) image, (CvArr*) harris_responce, (int) block_size, (int) aperture_size, (double) k), _CornerHarris, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswFindCornerSubPix( CvArr* image, CvPoint2D32f* corners, int count, CvSize win, CvSize zero_zone, CvTermCriteria criteria)
{
	typedef void(*_FindCornerSubPix) ( CvArr*, CvPoint2D32f*, int, CvSize, CvSize, CvTermCriteria ); 
	GPUCV_FUNCNAME("cvFindCornerSubPix");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) image, (CvPoint2D32f*) corners, (int) count, (CvSize) win, (CvSize) zero_zone, (CvTermCriteria) criteria), _FindCornerSubPix, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswGoodFeaturesToTrack( CvArr* image, CvArr* eig_image, CvArr* temp_image, CvPoint2D32f* corners, int* corner_count, double quality_level, double min_distance,  CvArr* mask, int block_size, int use_harris, double k)
{
	typedef void(*_GoodFeaturesToTrack) ( CvArr*, CvArr*, CvArr*, CvPoint2D32f*, int*, double, double,  CvArr*, int, int, double ); 
	GPUCV_FUNCNAME("cvGoodFeaturesToTrack");
	CvArr* SrcARR[] = { (CvArr*) image,  (CvArr*) eig_image,  (CvArr*) temp_image,  (CvArr*) mask};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) image, (CvArr*) eig_image, (CvArr*) temp_image, (CvPoint2D32f*) corners, (int*) corner_count, (double) quality_level, (double) min_distance, ( CvArr*) mask, (int) block_size, (int) use_harris, (double) k), _GoodFeaturesToTrack, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
CvSeq* cvgswHoughLines2(CvArr* image, void* line_storage, int method, double rho, double theta, int threshold, double param1, double param2)
{
	typedef CvSeq*(*_HoughLines2) (CvArr*, void*, int, double, double, int, double, double ); 
	GPUCV_FUNCNAME("cvHoughLines2");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	CvSeq* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) image, (void*) line_storage, (int) method, (double) rho, (double) theta, (int) threshold, (double) param1, (double) param2), _HoughLines2,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvSeq* cvgswHoughCircles(CvArr* image, void* circle_storage, int method, double dp, double min_dist, double param1, double param2, int min_radius, int max_radius)
{
	typedef CvSeq*(*_HoughCircles) (CvArr*, void*, int, double, double, double, double, int, int ); 
	GPUCV_FUNCNAME("cvHoughCircles");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	CvSeq* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) image, (void*) circle_storage, (int) method, (double) dp, (double) min_dist, (double) param1, (double) param2, (int) min_radius, (int) max_radius), _HoughCircles,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswFitLine( CvArr* points, int dist_type, double param, double reps, double aeps, float* line)
{
	typedef void(*_FitLine) ( CvArr*, int, double, double, double, float* ); 
	GPUCV_FUNCNAME("cvFitLine");
	CvArr* SrcARR[] = { (CvArr*) points};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) points, (int) dist_type, (double) param, (double) reps, (double) aeps, (float*) line), _FitLine, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswReleaseFeatureTree(struct CvFeatureTree* tr)
{
	cvReleaseFeatureTree((struct CvFeatureTree*) tr);
}


/*====================================*/
void cvgswFindFeatures(struct CvFeatureTree* tr,  CvMat* query_points, CvMat* indices, CvMat* dist, int k, int emax)
{
	typedef void(*_FindFeatures) (struct CvFeatureTree*,  CvMat*, CvMat*, CvMat*, int, int ); 
	GPUCV_FUNCNAME("cvFindFeatures");
	CvArr* SrcARR[] = { (CvArr*) query_points};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&indices)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dist)paramsobj->AddParam("option", "MASK");
	RUNOP(((struct CvFeatureTree*) tr, ( CvMat*) query_points, (CvMat*) indices, (CvMat*) dist, (int) k, (int) emax), _FindFeatures, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswFindFeaturesBoxed(struct CvFeatureTree* tr, CvMat* bounds_min, CvMat* bounds_max, CvMat* out_indices)
{
	return cvFindFeaturesBoxed((struct CvFeatureTree*) tr, (CvMat*) bounds_min, (CvMat*) bounds_max, (CvMat*) out_indices);
}


/*====================================*/
void cvgswReleaseLSH(struct CvLSH** lsh)
{
	cvReleaseLSH((struct CvLSH**) lsh);
}


/*====================================*/
void cvgswLSHAdd(struct CvLSH* lsh,  CvMat* data, CvMat* indices)
{
	typedef void(*_LSHAdd) (struct CvLSH*,  CvMat*, CvMat* ); 
	GPUCV_FUNCNAME("cvLSHAdd");
	CvArr* SrcARR[] = { (CvArr*) data};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&indices)paramsobj->AddParam("option", "MASK");
	RUNOP(((struct CvLSH*) lsh, ( CvMat*) data, (CvMat*) indices), _LSHAdd, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswLSHRemove(struct CvLSH* lsh,  CvMat* indices)
{
	typedef void(*_LSHRemove) (struct CvLSH*,  CvMat* ); 
	GPUCV_FUNCNAME("cvLSHRemove");
	CvArr* SrcARR[] = { (CvArr*) indices};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((struct CvLSH*) lsh, ( CvMat*) indices), _LSHRemove, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswLSHQuery(struct CvLSH* lsh,  CvMat* query_points, CvMat* indices, CvMat* dist, int k, int emax)
{
	typedef void(*_LSHQuery) (struct CvLSH*,  CvMat*, CvMat*, CvMat*, int, int ); 
	GPUCV_FUNCNAME("cvLSHQuery");
	CvArr* SrcARR[] = { (CvArr*) query_points};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&indices)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dist)paramsobj->AddParam("option", "MASK");
	RUNOP(((struct CvLSH*) lsh, ( CvMat*) query_points, (CvMat*) indices, (CvMat*) dist, (int) k, (int) emax), _LSHQuery, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
CvSURFPoint cvgswSURFPoint(CvPoint2D32f pt, int laplacian, int size, float dir, float hessian)
{
	return cvSURFPoint((CvPoint2D32f) pt, (int) laplacian, (int) size, (float) dir, (float) hessian);
}


/*====================================*/
CVAPI(CvSURFParams) cvgswSURFParams(double hessianThreshold, int extended)
{
	return cvSURFParams((double) hessianThreshold, (int) extended);
}


/*====================================*/
void cvgswExtractSURF( CvArr* img,  CvArr* mask, CvSeq** keypoints, CvSeq** descriptors, CvMemStorage* storage, CvSURFParams params, int useProvidedKeyPts)
{
	typedef void(*_ExtractSURF) ( CvArr*,  CvArr*, CvSeq**, CvSeq**, CvMemStorage*, CvSURFParams, int ); 
	GPUCV_FUNCNAME("cvExtractSURF");
	CvArr* SrcARR[] = { (CvArr*) img,  (CvArr*) mask};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) img, ( CvArr*) mask, (CvSeq**) keypoints, (CvSeq**) descriptors, (CvMemStorage*) storage, (CvSURFParams) params, (int) useProvidedKeyPts), _ExtractSURF, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
CVAPI(CvMSERParams) cvgswMSERParams(int delta, int min_area, int max_area, float max_variation, float min_diversity, int max_evolution, double area_threshold, double min_margin, int edge_blur_size)
{
	return cvMSERParams((int) delta, (int) min_area, (int) max_area, (float) max_variation, (float) min_diversity, (int) max_evolution, (double) area_threshold, (double) min_margin, (int) edge_blur_size);
}


/*====================================*/
void cvgswExtractMSER(CvArr* _img, CvArr* _mask, CvSeq** contours, CvMemStorage* storage, CvMSERParams params)
{
	typedef void(*_ExtractMSER) (CvArr*, CvArr*, CvSeq**, CvMemStorage*, CvMSERParams ); 
	GPUCV_FUNCNAME("cvExtractMSER");
	CvArr* SrcARR[] = { (CvArr*) _mask};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&_img)paramsobj->AddParam("option", "MASK");
	RUNOP(((CvArr*) _img, (CvArr*) _mask, (CvSeq**) contours, (CvMemStorage*) storage, (CvMSERParams) params), _ExtractMSER, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
CvStarKeypoint cvgswStarKeypoint(CvPoint pt, int size, float response)
{
	return cvStarKeypoint((CvPoint) pt, (int) size, (float) response);
}


/*====================================*/
CvStarDetectorParams cvgswStarDetectorParams(int maxSize, int responseThreshold, int lineThresholdProjected, int lineThresholdBinarized, int suppressNonmaxSize)
{
	return cvStarDetectorParams((int) maxSize, (int) responseThreshold, (int) lineThresholdProjected, (int) lineThresholdBinarized, (int) suppressNonmaxSize);
}


/*====================================*/
CvSeq* cvgswGetStarKeypoints( CvArr* img, CvMemStorage* storage, CvStarDetectorParams params)
{
	typedef CvSeq*(*_GetStarKeypoints) ( CvArr*, CvMemStorage*, CvStarDetectorParams ); 
	GPUCV_FUNCNAME("cvGetStarKeypoints");
	CvArr* SrcARR[] = { (CvArr*) img};
	CvArr** DstARR = NULL;
	CvSeq* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) img, (CvMemStorage*) storage, (CvStarDetectorParams) params), _GetStarKeypoints,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvHaarClassifierCascade* cvgswLoadHaarClassifierCascade(const  char* directory, CvSize orig_window_size)
{
	return cvLoadHaarClassifierCascade((const  char*) directory, (CvSize) orig_window_size);
}


/*====================================*/
void cvgswReleaseHaarClassifierCascade(CvHaarClassifierCascade** cascade)
{
	cvReleaseHaarClassifierCascade((CvHaarClassifierCascade**) cascade);
}


/*====================================*/
CvSeq* cvgswHaarDetectObjects( CvArr* image, CvHaarClassifierCascade* cascade, CvMemStorage* storage, double scale_factor, int min_neighbors, int flags, CvSize min_size)
{
	typedef CvSeq*(*_HaarDetectObjects) ( CvArr*, CvHaarClassifierCascade*, CvMemStorage*, double, int, int, CvSize ); 
	GPUCV_FUNCNAME("cvHaarDetectObjects");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	CvSeq* ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP((( CvArr*) image, (CvHaarClassifierCascade*) cascade, (CvMemStorage*) storage, (double) scale_factor, (int) min_neighbors, (int) flags, (CvSize) min_size), _HaarDetectObjects,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswSetImagesForHaarClassifierCascade(CvHaarClassifierCascade* cascade,  CvArr* sum,  CvArr* sqsum,  CvArr* tilted_sum, double scale)
{
	typedef void(*_SetImagesForHaarClassifierCascade) (CvHaarClassifierCascade*,  CvArr*,  CvArr*,  CvArr*, double ); 
	GPUCV_FUNCNAME("cvSetImagesForHaarClassifierCascade");
	CvArr* SrcARR[] = { (CvArr*) sum,  (CvArr*) sqsum,  (CvArr*) tilted_sum};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvHaarClassifierCascade*) cascade, ( CvArr*) sum, ( CvArr*) sqsum, ( CvArr*) tilted_sum, (double) scale), _SetImagesForHaarClassifierCascade, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswRunHaarClassifierCascade(const  CvHaarClassifierCascade* cascade, CvPoint pt, int start_stage)
{
	return cvRunHaarClassifierCascade((const  CvHaarClassifierCascade*) cascade, (CvPoint) pt, (int) start_stage);
}


/*====================================*/
void cvgswUndistort2( CvArr* src, CvArr* dst,  CvMat* camera_matrix,  CvMat* distortion_coeffs,  CvMat* new_camera_matrix)
{
	typedef void(*_Undistort2) ( CvArr*, CvArr*,  CvMat*,  CvMat*,  CvMat* ); 
	GPUCV_FUNCNAME("cvUndistort2");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) camera_matrix,  (CvArr*) distortion_coeffs,  (CvArr*) new_camera_matrix};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvArr*) src, (CvArr*) dst, ( CvMat*) camera_matrix, ( CvMat*) distortion_coeffs, ( CvMat*) new_camera_matrix), _Undistort2, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswInitUndistortMap( CvMat* camera_matrix,  CvMat* distortion_coeffs, CvArr* mapx, CvArr* mapy)
{
	typedef void(*_InitUndistortMap) ( CvMat*,  CvMat*, CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvInitUndistortMap");
	CvArr* SrcARR[] = { (CvArr*) camera_matrix,  (CvArr*) distortion_coeffs};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mapx)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&mapy)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) camera_matrix, ( CvMat*) distortion_coeffs, (CvArr*) mapx, (CvArr*) mapy), _InitUndistortMap, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswInitUndistortRectifyMap( CvMat* camera_matrix,  CvMat* dist_coeffs, const  CvMat * R,  CvMat* new_camera_matrix, CvArr* mapx, CvArr* mapy)
{
	typedef void(*_InitUndistortRectifyMap) ( CvMat*,  CvMat*, const  CvMat *,  CvMat*, CvArr*, CvArr* ); 
	GPUCV_FUNCNAME("cvInitUndistortRectifyMap");
	CvArr* SrcARR[] = { (CvArr*) camera_matrix,  (CvArr*) dist_coeffs,  (CvArr*) new_camera_matrix};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&mapx)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&mapy)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) camera_matrix, ( CvMat*) dist_coeffs, (const  CvMat *) R, ( CvMat*) new_camera_matrix, (CvArr*) mapx, (CvArr*) mapy), _InitUndistortRectifyMap, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswUndistortPoints( CvMat* src, CvMat* dst,  CvMat* camera_matrix,  CvMat* dist_coeffs,  CvMat* R,  CvMat* P)
{
	typedef void(*_UndistortPoints) ( CvMat*, CvMat*,  CvMat*,  CvMat*,  CvMat*,  CvMat* ); 
	GPUCV_FUNCNAME("cvUndistortPoints");
	CvArr* SrcARR[] = { (CvArr*) src,  (CvArr*) camera_matrix,  (CvArr*) dist_coeffs,  (CvArr*) R,  (CvArr*) P};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvMat*) src, (CvMat*) dst, ( CvMat*) camera_matrix, ( CvMat*) dist_coeffs, ( CvMat*) R, ( CvMat*) P), _UndistortPoints, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswGetOptimalNewCameraMatrix( CvMat* camera_matrix,  CvMat* dist_coeffs, CvSize image_size, double alpha, CvMat* new_camera_matrix, CvSize new_imag_size, CvRect* valid_pixel_ROI)
{
	typedef void(*_GetOptimalNewCameraMatrix) ( CvMat*,  CvMat*, CvSize, double, CvMat*, CvSize, CvRect* ); 
	GPUCV_FUNCNAME("cvGetOptimalNewCameraMatrix");
	CvArr* SrcARR[] = { (CvArr*) camera_matrix,  (CvArr*) dist_coeffs};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&new_camera_matrix)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) camera_matrix, ( CvMat*) dist_coeffs, (CvSize) image_size, (double) alpha, (CvMat*) new_camera_matrix, (CvSize) new_imag_size, (CvRect*) valid_pixel_ROI), _GetOptimalNewCameraMatrix, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswRodrigues2( CvMat* src, CvMat* dst, CvMat* jacobian)
{
	typedef int(*_Rodrigues2) ( CvMat*, CvMat*, CvMat* ); 
	GPUCV_FUNCNAME("cvRodrigues2");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	int ReturnObj;	SWITCH_START_OPR(dst); 
//Mask has been found, add it to params.
	 if(paramsobj &&jacobian)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) src, (CvMat*) dst, (CvMat*) jacobian), _Rodrigues2,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
int cvgswFindHomography( CvMat* src_points,  CvMat* dst_points, CvMat* homography, int method, double ransacReprojThreshold, CvMat* mask)
{
	typedef int(*_FindHomography) ( CvMat*,  CvMat*, CvMat*, int, double, CvMat* ); 
	GPUCV_FUNCNAME("cvFindHomography");
	CvArr* SrcARR[] = { (CvArr*) src_points,  (CvArr*) dst_points,  (CvArr*) mask};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&homography)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&mask)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) src_points, ( CvMat*) dst_points, (CvMat*) homography, (int) method, (double) ransacReprojThreshold, (CvMat*) mask), _FindHomography,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswRQDecomp3x3(const  CvMat * matrixM, CvMat * matrixR, CvMat * matrixQ, CvMat * matrixQx, CvMat * matrixQy, CvMat * matrixQz, CvPoint3D64f * eulerAngles)
{
	cvRQDecomp3x3((const  CvMat *) matrixM, (CvMat *) matrixR, (CvMat *) matrixQ, (CvMat *) matrixQx, (CvMat *) matrixQy, (CvMat *) matrixQz, (CvPoint3D64f *) eulerAngles);
}


/*====================================*/
void cvgswDecomposeProjectionMatrix(const  CvMat * projMatr, CvMat * calibMatr, CvMat * rotMatr, CvMat * posVect, CvMat * rotMatrX, CvMat * rotMatrY, CvMat * rotMatrZ, CvPoint3D64f * eulerAngles)
{
	cvDecomposeProjectionMatrix((const  CvMat *) projMatr, (CvMat *) calibMatr, (CvMat *) rotMatr, (CvMat *) posVect, (CvMat *) rotMatrX, (CvMat *) rotMatrY, (CvMat *) rotMatrZ, (CvPoint3D64f *) eulerAngles);
}


/*====================================*/
void cvgswCalcMatMulDeriv( CvMat* A,  CvMat* B, CvMat* dABdA, CvMat* dABdB)
{
	typedef void(*_CalcMatMulDeriv) ( CvMat*,  CvMat*, CvMat*, CvMat* ); 
	GPUCV_FUNCNAME("cvCalcMatMulDeriv");
	CvArr* SrcARR[] = { (CvArr*) A,  (CvArr*) B};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&dABdA)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dABdB)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) A, ( CvMat*) B, (CvMat*) dABdA, (CvMat*) dABdB), _CalcMatMulDeriv, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswComposeRT( CvMat* _rvec1,  CvMat* _tvec1,  CvMat* _rvec2,  CvMat* _tvec2, CvMat* _rvec3, CvMat* _tvec3, CvMat* dr3dr1, CvMat* dr3dt1, CvMat* dr3dr2, CvMat* dr3dt2, CvMat* dt3dr1, CvMat* dt3dt1, CvMat* dt3dr2, CvMat* dt3dt2)
{
	typedef void(*_ComposeRT) ( CvMat*,  CvMat*,  CvMat*,  CvMat*, CvMat*, CvMat*, CvMat*, CvMat*, CvMat*, CvMat*, CvMat*, CvMat*, CvMat*, CvMat* ); 
	GPUCV_FUNCNAME("cvComposeRT");
	CvArr* SrcARR[] = { (CvArr*) _rvec1,  (CvArr*) _tvec1,  (CvArr*) _rvec2,  (CvArr*) _tvec2};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&_rvec3)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&_tvec3)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dr3dr1)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dr3dt1)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dr3dr2)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dr3dt2)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dt3dr1)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dt3dt1)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dt3dr2)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dt3dt2)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) _rvec1, ( CvMat*) _tvec1, ( CvMat*) _rvec2, ( CvMat*) _tvec2, (CvMat*) _rvec3, (CvMat*) _tvec3, (CvMat*) dr3dr1, (CvMat*) dr3dt1, (CvMat*) dr3dr2, (CvMat*) dr3dt2, (CvMat*) dt3dr1, (CvMat*) dt3dt1, (CvMat*) dt3dr2, (CvMat*) dt3dt2), _ComposeRT, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswProjectPoints2( CvMat* object_points,  CvMat* rotation_vector,  CvMat* translation_vector,  CvMat* camera_matrix,  CvMat* distortion_coeffs, CvMat* image_points, CvMat* dpdrot, CvMat* dpdt, CvMat* dpdf, CvMat* dpdc, CvMat* dpddist, double aspect_ratio)
{
	typedef void(*_ProjectPoints2) ( CvMat*,  CvMat*,  CvMat*,  CvMat*,  CvMat*, CvMat*, CvMat*, CvMat*, CvMat*, CvMat*, CvMat*, double ); 
	GPUCV_FUNCNAME("cvProjectPoints2");
	CvArr* SrcARR[] = { (CvArr*) object_points,  (CvArr*) rotation_vector,  (CvArr*) translation_vector,  (CvArr*) camera_matrix,  (CvArr*) distortion_coeffs,  (CvArr*) image_points};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&dpdrot)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dpdt)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dpdf)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dpdc)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dpddist)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) object_points, ( CvMat*) rotation_vector, ( CvMat*) translation_vector, ( CvMat*) camera_matrix, ( CvMat*) distortion_coeffs, (CvMat*) image_points, (CvMat*) dpdrot, (CvMat*) dpdt, (CvMat*) dpdf, (CvMat*) dpdc, (CvMat*) dpddist, (double) aspect_ratio), _ProjectPoints2, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswFindExtrinsicCameraParams2( CvMat* object_points,  CvMat* image_points,  CvMat* camera_matrix,  CvMat* distortion_coeffs, CvMat* rotation_vector, CvMat* translation_vector, int use_extrinsic_guess)
{
	typedef void(*_FindExtrinsicCameraParams2) ( CvMat*,  CvMat*,  CvMat*,  CvMat*, CvMat*, CvMat*, int ); 
	GPUCV_FUNCNAME("cvFindExtrinsicCameraParams2");
	CvArr* SrcARR[] = { (CvArr*) object_points,  (CvArr*) image_points,  (CvArr*) camera_matrix,  (CvArr*) distortion_coeffs};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&rotation_vector)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&translation_vector)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) object_points, ( CvMat*) image_points, ( CvMat*) camera_matrix, ( CvMat*) distortion_coeffs, (CvMat*) rotation_vector, (CvMat*) translation_vector, (int) use_extrinsic_guess), _FindExtrinsicCameraParams2, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswInitIntrinsicParams2D( CvMat* object_points,  CvMat* image_points,  CvMat* npoints, CvSize image_size, CvMat* camera_matrix, double aspect_ratio)
{
	typedef void(*_InitIntrinsicParams2D) ( CvMat*,  CvMat*,  CvMat*, CvSize, CvMat*, double ); 
	GPUCV_FUNCNAME("cvInitIntrinsicParams2D");
	CvArr* SrcARR[] = { (CvArr*) object_points,  (CvArr*) image_points,  (CvArr*) npoints};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&camera_matrix)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) object_points, ( CvMat*) image_points, ( CvMat*) npoints, (CvSize) image_size, (CvMat*) camera_matrix, (double) aspect_ratio), _InitIntrinsicParams2D, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswCheckChessboard(IplImage* src, CvSize size)
{
	typedef int(*_CheckChessboard) (IplImage*, CvSize ); 
	GPUCV_FUNCNAME("cvCheckChessboard");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
	RUNOP(((IplImage*) src, (CvSize) size), _CheckChessboard,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
int cvgswFindChessboardCorners(const  void* image, CvSize pattern_size, CvPoint2D32f* corners, int* corner_count, int flags)
{
	return cvFindChessboardCorners((const  void*) image, (CvSize) pattern_size, (CvPoint2D32f*) corners, (int*) corner_count, (int) flags);
}


/*====================================*/
void cvgswDrawChessboardCorners(CvArr* image, CvSize pattern_size, CvPoint2D32f* corners, int count, int pattern_was_found)
{
	typedef void(*_DrawChessboardCorners) (CvArr*, CvSize, CvPoint2D32f*, int, int ); 
	GPUCV_FUNCNAME("cvDrawChessboardCorners");
	CvArr* SrcARR[] = { (CvArr*) image};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
	RUNOP(((CvArr*) image, (CvSize) pattern_size, (CvPoint2D32f*) corners, (int) count, (int) pattern_was_found), _DrawChessboardCorners, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
double cvgswCalibrateCamera2( CvMat* object_points,  CvMat* image_points,  CvMat* point_counts, CvSize image_size, CvMat* camera_matrix, CvMat* distortion_coeffs, CvMat* rotation_vectors, CvMat* translation_vectors, int flags)
{
	typedef double(*_CalibrateCamera2) ( CvMat*,  CvMat*,  CvMat*, CvSize, CvMat*, CvMat*, CvMat*, CvMat*, int ); 
	GPUCV_FUNCNAME("cvCalibrateCamera2");
	CvArr* SrcARR[] = { (CvArr*) object_points,  (CvArr*) image_points,  (CvArr*) point_counts};
	CvArr** DstARR = NULL;
	double ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&camera_matrix)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&distortion_coeffs)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&rotation_vectors)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&translation_vectors)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) object_points, ( CvMat*) image_points, ( CvMat*) point_counts, (CvSize) image_size, (CvMat*) camera_matrix, (CvMat*) distortion_coeffs, (CvMat*) rotation_vectors, (CvMat*) translation_vectors, (int) flags), _CalibrateCamera2,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswCalibrationMatrixValues(const  CvMat * camera_matrix, CvSize image_size, double aperture_width, double aperture_height, double * fovx, double * fovy, double * focal_length, CvPoint2D64f * principal_point, double * pixel_aspect_ratio)
{
	cvCalibrationMatrixValues((const  CvMat *) camera_matrix, (CvSize) image_size, (double) aperture_width, (double) aperture_height, (double *) fovx, (double *) fovy, (double *) focal_length, (CvPoint2D64f *) principal_point, (double *) pixel_aspect_ratio);
}


/*====================================*/
double cvgswStereoCalibrate( CvMat* object_points,  CvMat* image_points1,  CvMat* image_points2,  CvMat* npoints, CvMat* camera_matrix1, CvMat* dist_coeffs1, CvMat* camera_matrix2, CvMat* dist_coeffs2, CvSize image_size, CvMat* R, CvMat* T, CvMat* E, CvMat* F, CvTermCriteria term_crit, int flags)
{
	typedef double(*_StereoCalibrate) ( CvMat*,  CvMat*,  CvMat*,  CvMat*, CvMat*, CvMat*, CvMat*, CvMat*, CvSize, CvMat*, CvMat*, CvMat*, CvMat*, CvTermCriteria, int ); 
	GPUCV_FUNCNAME("cvStereoCalibrate");
	CvArr* SrcARR[] = { (CvArr*) object_points,  (CvArr*) image_points1,  (CvArr*) image_points2,  (CvArr*) npoints};
	CvArr** DstARR = NULL;
	double ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&camera_matrix1)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dist_coeffs1)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&camera_matrix2)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&dist_coeffs2)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&R)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&T)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&E)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&F)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) object_points, ( CvMat*) image_points1, ( CvMat*) image_points2, ( CvMat*) npoints, (CvMat*) camera_matrix1, (CvMat*) dist_coeffs1, (CvMat*) camera_matrix2, (CvMat*) dist_coeffs2, (CvSize) image_size, (CvMat*) R, (CvMat*) T, (CvMat*) E, (CvMat*) F, (CvTermCriteria) term_crit, (int) flags), _StereoCalibrate,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswStereoRectify( CvMat* camera_matrix1,  CvMat* camera_matrix2,  CvMat* dist_coeffs1,  CvMat* dist_coeffs2, CvSize image_size,  CvMat* R,  CvMat* T, CvMat* R1, CvMat* R2, CvMat* P1, CvMat* P2, CvMat* Q, int flags, double alpha, CvSize new_image_size, CvRect* valid_pix_ROI1, CvRect* valid_pix_ROI2)
{
	typedef void(*_StereoRectify) ( CvMat*,  CvMat*,  CvMat*,  CvMat*, CvSize,  CvMat*,  CvMat*, CvMat*, CvMat*, CvMat*, CvMat*, CvMat*, int, double, CvSize, CvRect*, CvRect* ); 
	GPUCV_FUNCNAME("cvStereoRectify");
	CvArr* SrcARR[] = { (CvArr*) camera_matrix1,  (CvArr*) camera_matrix2,  (CvArr*) dist_coeffs1,  (CvArr*) dist_coeffs2,  (CvArr*) R,  (CvArr*) T};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&R1)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&R2)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&P1)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&P2)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&Q)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) camera_matrix1, ( CvMat*) camera_matrix2, ( CvMat*) dist_coeffs1, ( CvMat*) dist_coeffs2, (CvSize) image_size, ( CvMat*) R, ( CvMat*) T, (CvMat*) R1, (CvMat*) R2, (CvMat*) P1, (CvMat*) P2, (CvMat*) Q, (int) flags, (double) alpha, (CvSize) new_image_size, (CvRect*) valid_pix_ROI1, (CvRect*) valid_pix_ROI2), _StereoRectify, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswStereoRectifyUncalibrated( CvMat* points1,  CvMat* points2,  CvMat* F, CvSize img_size, CvMat* H1, CvMat* H2, double threshold)
{
	typedef int(*_StereoRectifyUncalibrated) ( CvMat*,  CvMat*,  CvMat*, CvSize, CvMat*, CvMat*, double ); 
	GPUCV_FUNCNAME("cvStereoRectifyUncalibrated");
	CvArr* SrcARR[] = { (CvArr*) points1,  (CvArr*) points2,  (CvArr*) F};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&H1)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&H2)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) points1, ( CvMat*) points2, ( CvMat*) F, (CvSize) img_size, (CvMat*) H1, (CvMat*) H2, (double) threshold), _StereoRectifyUncalibrated,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
CvPOSITObject* cvgswCreatePOSITObject(CvPoint3D32f* points, int point_count)
{
	return cvCreatePOSITObject((CvPoint3D32f*) points, (int) point_count);
}


/*====================================*/
void cvgswPOSIT(CvPOSITObject* posit_object, CvPoint2D32f* image_points, double focal_length, CvTermCriteria criteria, CvMatr32f rotation_matrix, CvVect32f translation_vector)
{
	cvPOSIT((CvPOSITObject*) posit_object, (CvPoint2D32f*) image_points, (double) focal_length, (CvTermCriteria) criteria, (CvMatr32f) rotation_matrix, (CvVect32f) translation_vector);
}


/*====================================*/
void cvgswReleasePOSITObject(CvPOSITObject** posit_object)
{
	cvReleasePOSITObject((CvPOSITObject**) posit_object);
}


/*====================================*/
int cvgswRANSACUpdateNumIters(double p, double err_prob, int model_points, int max_iters)
{
	return cvRANSACUpdateNumIters((double) p, (double) err_prob, (int) model_points, (int) max_iters);
}


/*====================================*/
void cvgswConvertPointsHomogeneous( CvMat* src, CvMat* dst)
{
	typedef void(*_ConvertPointsHomogeneous) ( CvMat*, CvMat* ); 
	GPUCV_FUNCNAME("cvConvertPointsHomogeneous");
	CvArr* SrcARR[] = { (CvArr*) src};
	CvArr* DstARR[] = { (CvArr*) dst};
	SWITCH_START_OPR(dst); 
	RUNOP((( CvMat*) src, (CvMat*) dst), _ConvertPointsHomogeneous, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
int cvgswFindFundamentalMat( CvMat* points1,  CvMat* points2, CvMat* fundamental_matrix, int method, double param1, double param2, CvMat* status)
{
	typedef int(*_FindFundamentalMat) ( CvMat*,  CvMat*, CvMat*, int, double, double, CvMat* ); 
	GPUCV_FUNCNAME("cvFindFundamentalMat");
	CvArr* SrcARR[] = { (CvArr*) points1,  (CvArr*) points2};
	CvArr** DstARR = NULL;
	int ReturnObj;	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&fundamental_matrix)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&status)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) points1, ( CvMat*) points2, (CvMat*) fundamental_matrix, (int) method, (double) param1, (double) param2, (CvMat*) status), _FindFundamentalMat,  ReturnObj =); 
	SWITCH_STOP_OPR();
	return ReturnObj;

}


/*====================================*/
void cvgswComputeCorrespondEpilines( CvMat* points, int which_image,  CvMat* fundamental_matrix, CvMat* correspondent_lines)
{
	typedef void(*_ComputeCorrespondEpilines) ( CvMat*, int,  CvMat*, CvMat* ); 
	GPUCV_FUNCNAME("cvComputeCorrespondEpilines");
	CvArr* SrcARR[] = { (CvArr*) points,  (CvArr*) fundamental_matrix};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&correspondent_lines)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvMat*) points, (int) which_image, ( CvMat*) fundamental_matrix, (CvMat*) correspondent_lines), _ComputeCorrespondEpilines, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswTriangulatePoints(CvMat* projMatr1, CvMat* projMatr2, CvMat* projPoints1, CvMat* projPoints2, CvMat* points4D)
{
	cvTriangulatePoints((CvMat*) projMatr1, (CvMat*) projMatr2, (CvMat*) projPoints1, (CvMat*) projPoints2, (CvMat*) points4D);
}


/*====================================*/
void cvgswCorrectMatches(CvMat* F, CvMat* points1, CvMat* points2, CvMat* new_points1, CvMat* new_points2)
{
	cvCorrectMatches((CvMat*) F, (CvMat*) points1, (CvMat*) points2, (CvMat*) new_points1, (CvMat*) new_points2);
}


/*====================================*/
CvStereoBMState* cvgswCreateStereoBMState(int preset, int numberOfDisparities)
{
	return cvCreateStereoBMState((int) preset, (int) numberOfDisparities);
}


/*====================================*/
void cvgswReleaseStereoBMState(CvStereoBMState** state)
{
	cvReleaseStereoBMState((CvStereoBMState**) state);
}


/*====================================*/
void cvgswFindStereoCorrespondenceBM( CvArr* left,  CvArr* right, CvArr* disparity, CvStereoBMState* state)
{
	typedef void(*_FindStereoCorrespondenceBM) ( CvArr*,  CvArr*, CvArr*, CvStereoBMState* ); 
	GPUCV_FUNCNAME("cvFindStereoCorrespondenceBM");
	CvArr* SrcARR[] = { (CvArr*) left,  (CvArr*) right};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&disparity)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) left, ( CvArr*) right, (CvArr*) disparity, (CvStereoBMState*) state), _FindStereoCorrespondenceBM, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
CvRect cvgswGetValidDisparityROI(CvRect roi1, CvRect roi2, int minDisparity, int numberOfDisparities, int SADWindowSize)
{
	return cvGetValidDisparityROI((CvRect) roi1, (CvRect) roi2, (int) minDisparity, (int) numberOfDisparities, (int) SADWindowSize);
}


/*====================================*/
void cvgswValidateDisparity(CvArr* disparity,  CvArr* cost, int minDisparity, int numberOfDisparities, int disp12MaxDiff)
{
	typedef void(*_ValidateDisparity) (CvArr*,  CvArr*, int, int, int ); 
	GPUCV_FUNCNAME("cvValidateDisparity");
	CvArr* SrcARR[] = { (CvArr*) cost};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&disparity)paramsobj->AddParam("option", "MASK");
	RUNOP(((CvArr*) disparity, ( CvArr*) cost, (int) minDisparity, (int) numberOfDisparities, (int) disp12MaxDiff), _ValidateDisparity, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
CvStereoGCState* cvgswCreateStereoGCState(int numberOfDisparities, int maxIters)
{
	return cvCreateStereoGCState((int) numberOfDisparities, (int) maxIters);
}


/*====================================*/
void cvgswReleaseStereoGCState(CvStereoGCState** state)
{
	cvReleaseStereoGCState((CvStereoGCState**) state);
}


/*====================================*/
void cvgswFindStereoCorrespondenceGC( CvArr* left,  CvArr* right, CvArr* disparityLeft, CvArr* disparityRight, CvStereoGCState* state, int useDisparityGuess)
{
	typedef void(*_FindStereoCorrespondenceGC) ( CvArr*,  CvArr*, CvArr*, CvArr*, CvStereoGCState*, int ); 
	GPUCV_FUNCNAME("cvFindStereoCorrespondenceGC");
	CvArr* SrcARR[] = { (CvArr*) left,  (CvArr*) right};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&disparityLeft)paramsobj->AddParam("option", "MASK");
//Mask has been found, add it to params.
	 if(paramsobj &&disparityRight)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) left, ( CvArr*) right, (CvArr*) disparityLeft, (CvArr*) disparityRight, (CvStereoGCState*) state, (int) useDisparityGuess), _FindStereoCorrespondenceGC, ); 
	SWITCH_STOP_OPR();
}


/*====================================*/
void cvgswReprojectImageTo3D( CvArr* disparityImage, CvArr* _3dImage,  CvMat* Q, int handleMissingValues)
{
	typedef void(*_ReprojectImageTo3D) ( CvArr*, CvArr*,  CvMat*, int ); 
	GPUCV_FUNCNAME("cvReprojectImageTo3D");
	CvArr* SrcARR[] = { (CvArr*) disparityImage,  (CvArr*) Q};
	CvArr** DstARR = NULL;
	SWITCH_START_OPR(NULL); 
//Mask has been found, add it to params.
	 if(paramsobj &&_3dImage)paramsobj->AddParam("option", "MASK");
	RUNOP((( CvArr*) disparityImage, (CvArr*) _3dImage, ( CvMat*) Q, (int) handleMissingValues), _ReprojectImageTo3D, ); 
	SWITCH_STOP_OPR();
}

/*........End Code.............*/

