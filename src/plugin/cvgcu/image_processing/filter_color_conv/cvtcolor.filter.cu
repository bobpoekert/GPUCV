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
/*
* The function cvCvtColor converts input image from one color space to another
*/
#include <cvgcu/config.h>

#if _GPUCV_COMPILE_CUDA
 
#include <typeinfo>
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <cvgcu/image_processing/filter_color_conv/cvtColor.filter.h>


//from opencv/cv/include/cv.h
/* Constants for color conversion */
#define  CV_BGR2BGRA    0
#define  CV_RGB2RGBA    CV_BGR2BGRA

#define  CV_BGRA2BGR    1
#define  CV_RGBA2RGB    CV_BGRA2BGR

#define  CV_BGR2RGBA    2
#define  CV_RGB2BGRA    CV_BGR2RGBA

#define  CV_RGBA2BGR    3
#define  CV_BGRA2RGB    CV_RGBA2BGR

#define  CV_BGR2RGB     4
#define  CV_RGB2BGR     CV_BGR2RGB

#define  CV_BGRA2RGBA   5
#define  CV_RGBA2BGRA   CV_BGRA2RGBA

#define  CV_BGR2GRAY    6
#define  CV_RGB2GRAY    7

#define  CV_GRAY2BGR    8
#define  CV_GRAY2RGB    CV_GRAY2BGR

#define  CV_GRAY2BGRA   9
#define  CV_GRAY2RGBA   CV_GRAY2BGRA

#define  CV_BGRA2GRAY   10
#define  CV_RGBA2GRAY   11

#define  CV_BGR2BGR565  12
#define  CV_RGB2BGR565  13

#define  CV_BGR5652BGR  14

#define  CV_BGR5652RGB  15

#define  CV_BGRA2BGR565 16

#define  CV_RGBA2BGR565 17

#define  CV_BGR5652BGRA 18

#define  CV_BGR5652RGBA 19

#define  CV_GRAY2BGR565 20

#define  CV_BGR5652GRAY 21

#define  CV_BGR2BGR555  22

#define  CV_RGB2BGR555  23

#define  CV_BGR5552BGR  24

#define  CV_BGR5552RGB  25

#define  CV_BGRA2BGR555 26

#define  CV_RGBA2BGR555 27

#define  CV_BGR5552BGRA 28

#define  CV_BGR5552RGBA 29



#define  CV_GRAY2BGR555 30
#define  CV_BGR5552GRAY 31



#define  CV_BGR2XYZ     32
#define  CV_RGB2XYZ     33

#define  CV_XYZ2BGR     34
#define  CV_XYZ2RGB     35

#define  CV_BGR2YCrCb   36
#define  CV_RGB2YCrCb   37

#define  CV_YCrCb2BGR   38
#define  CV_YCrCb2RGB   39

#define  CV_BGR2HSV     40
#define  CV_RGB2HSV     41

#define  CV_BGR2Lab     44
#define  CV_RGB2Lab     45

#define  CV_BayerBG2BGR 46
#define  CV_BayerGB2BGR 47

#define  CV_BayerRG2BGR 48
#define  CV_BayerGR2BGR 49

#define  CV_BayerBG2RGB CV_BayerRG2BGR
#define  CV_BayerGB2RGB CV_BayerGR2BGR
#define  CV_BayerRG2RGB CV_BayerBG2BGR
#define  CV_BayerGR2RGB CV_BayerGB2BGR



#define  CV_BGR2Luv     50

#define  CV_RGB2Luv     51

#define  CV_BGR2HLS     52

#define  CV_RGB2HLS     53



#define  CV_HSV2BGR     54

#define  CV_HSV2RGB     55



#define  CV_Lab2BGR     56

#define  CV_Lab2RGB     57

#define  CV_Luv2BGR     58

#define  CV_Luv2RGB     59

#define  CV_HLS2BGR     60

#define  CV_HLS2RGB     61
#define  CV_COLORCVT_MAX  100
//=======================================================
/**\todo Only BGR/RGB to Grey is done and working. Add other format conversions
*/
_GPUCV_CVGCU_EXPORT_CU
void gcuCvtColor(CvArr* srcArr, CvArr* dstArr, int code)
{
	//cudaArray * Array=NULL;
	unsigned int height	= gcuGetHeight(srcArr);
	unsigned int width	= gcuGetWidth(srcArr);
	unsigned int depth	= gcuGetGLDepth(srcArr);

	void * d_dst = gcuPreProcess(dstArr, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
	void * d_src = gcuPreProcess(srcArr, GCU_INPUT, CU_MEMORYTYPE_DEVICE);

	dim3 threads(16,16,1);
	dim3 blocks = dim3(iDivUp(width,threads.x),
		iDivUp(height, threads.y),
		1);


	float scale = 1.;
	float shift = 0.;



#define GCU_CVTCOLOR_BGR2GRAY_SWITCH_FCT(CHANNELS, DST_TYPE, SRC_TYPE)\
	gcudaKernel_CvtColor_BGR2GRAY<DST_TYPE, SRC_TYPE##CHANNELS> <<<blocks, threads>>> ((SRC_TYPE##CHANNELS*)d_src,\
	(DST_TYPE*)		d_dst,\
	width, height, scale, shift);

#define GCU_CVTCOLOR_RGB2GRAY_SWITCH_FCT(CHANNELS, DST_TYPE, SRC_TYPE)\
	gcudaKernel_CvtColor_RGB2GRAY<DST_TYPE, SRC_TYPE##CHANNELS> <<<blocks, threads>>> ((SRC_TYPE##CHANNELS*)d_src,\
	(DST_TYPE*)		d_dst,\
	width, height, scale, shift);

#define GCU_CVTCOLOR_BGR2YCRCB_SWITCH_FCT(CHANNELS, DST_TYPE, SRC_TYPE)\
	gcudaKernel_CvtColor_BGR2YCrCb<DST_TYPE##CHANNELS, SRC_TYPE##CHANNELS> <<<blocks, threads>>> ((SRC_TYPE##CHANNELS*)d_src,\
	(DST_TYPE##CHANNELS*)		d_dst,\
	width, height, scale, shift);

#define GCU_CVTCOLOR_RGB2YCRCB_SWITCH_FCT(CHANNELS, DST_TYPE, SRC_TYPE)\
	gcudaKernel_CvtColor_RGB2YCrCb<DST_TYPE##CHANNELS, SRC_TYPE##CHANNELS> <<<blocks, threads>>> ((SRC_TYPE##CHANNELS*)d_src,\
	(DST_TYPE##CHANNELS*)		d_dst,\
	width, height, scale, shift);

	switch (code)
	{
#if 0
		case CV_RGB2XYZ:
		case CV_BGR2XYZ:
		case CV_XYZ2RGB:
		case CV_XYZ2BGR:
 
		case CV_YCrCb2RGB:
		case CV_YCrCb2BGR:
		case CV_RGB2HSV:
		case CV_RGB2Lab:
		case CV_BGR2Lab:
#endif
		//case CV_RGB2YCrCb:	GCU_MULTIPLEX_1CHANNELS_ALLFORMAT(GCU_CVTCOLOR_RGB2YCRCB_SWITCH_FCT, 3, depth);break;
		//case CV_BGR2YCrCb:	GCU_MULTIPLEX_1CHANNELS_ALLFORMAT(GCU_CVTCOLOR_BGR2YCRCB_SWITCH_FCT, 3, depth);break;
		case CV_RGB2GRAY:	GCU_MULTIPLEX_1CHANNELS_ALLFORMAT(GCU_CVTCOLOR_RGB2GRAY_SWITCH_FCT, 3, depth);break;
		case CV_BGR2GRAY:	GCU_MULTIPLEX_1CHANNELS_ALLFORMAT(GCU_CVTCOLOR_BGR2GRAY_SWITCH_FCT, 3, depth);break;

		//	case  CV_BGR2YCrCb: 
		//	case  CV_RGB2YCrCb: 
		default:
		printf("gcuCvtColor(): color conversion requested not performed.\n");
	}

	gcuPostProcess(srcArr); 
	gcuPostProcess(dstArr);
}
#endif //_GPUCV_COMPILE_CUDA

