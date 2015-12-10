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




/** \brief C++ File containg definitions GPU equivalents for openCV functions
\author Jean-Philippe Farrugia, Yannick Allusse
*/
#include "StdAfx.h"
#include <cvg/cvg.h>
#include <cxcoreg/cxcoreg.h>
#include <highguig/highguig.h>
#include <GPUCV/misc.h>

using namespace GCV;

_GPUCV_CVG_EXPORT_C LibraryDescriptor *modGetLibraryDescriptor(void)
{
	static LibraryDescriptor * pLibraryDescriptor=NULL;
	if(pLibraryDescriptor ==NULL)//init
	{
		pLibraryDescriptor = new LibraryDescriptor();
		pLibraryDescriptor->SetVersionMajor("1");
		pLibraryDescriptor->SetVersionMinor("0");
		pLibraryDescriptor->SetSvnRev("570");
		pLibraryDescriptor->SetSvnDate("");
		pLibraryDescriptor->SetWebUrl(DLLINFO_DFT_URL);
		pLibraryDescriptor->SetAuthor(DLLINFO_DFT_AUTHOR);
		pLibraryDescriptor->SetDllName("cvg");
		pLibraryDescriptor->SetImplementationName("GLSL");
		pLibraryDescriptor->SetBaseImplementationID(GPUCV_IMPL_GLSL);
		pLibraryDescriptor->SetUseGpu(true);
		pLibraryDescriptor->SetStartColor(GPUCV_IMPL_GLSL_COLOR_START);
		pLibraryDescriptor->SetStopColor(GPUCV_IMPL_GLSL_COLOR_STOP);
	}
	return pLibraryDescriptor;
}


_GPUCV_CVG_EXPORT_C int cvgDLLInit(bool InitGLContext, bool isMultiThread)
{
	return GpuCVInit(InitGLContext, isMultiThread);
}
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
//CVG_IMGPROC__GRAD_EDGE_CORNER_GRP
//=================================================================
#include <cv.hpp>
#include <cxcore.hpp>
void cvgSobel(CvArr* src, CvArr* dst, int xorder, int yorder, int aperture_size/*=3*/)
{
	GPUCV_START_OP(cvSobel(src, dst, xorder, yorder,aperture_size),
		"cvgSobel",
		dst,
		GenericGPU::HRD_PRF_3);

	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");

	float params[18];// = {threshold/255., maxValue/255.};

	//calcul offset values:
	GLint i, j;
	GLfloat xInc, yInc;
	GLuint width  = ((IplImage*)dst)->width;
	GLuint height = ((IplImage*)dst)->height;




#if 0
	windowWidth = textureWidth = w;
	windowHeight = textureHeight = h;
	// find largest power-of-two texture smaller than window
	if (!npotTexturesAvailable)
	{
		// Find the largest power of two that will fit in window.

		// Try each width until we get one that's too big
		i = 0;
		while ((1 << i) <= textureWidth)
			i++;
		textureWidth = (1 << (i-1));

		// Now for height
		i = 0;
		while ((1 << i) <= textureHeight)
			i++;
		textureHeight = (1 << (i-1));
	}

	if (textureWidth > maxTexSize)
	{
		textureWidth = maxTexSize;
	}
	if (textureHeight > maxTexSize)
	{
		textureHeight = maxTexSize;
	}
#endif

	xInc = 1.0f / (GLfloat)width;
	yInc = 1.0f / (GLfloat)height;

	//calculate pixel offsets...
	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			//original
			//params[(((i*3)+j)*2)+0] = (-1.0f * xInc) + ((GLfloat)i * xInc);
			//params[(((i*3)+j)*2)+1] = (-1.0f * yInc) + ((GLfloat)j * yInc);
			//new
			params[(((i*3)+j)*2)+0] = (-1.0f * xInc) + ((GLfloat)j * xInc);
			params[(((i*3)+j)*2)+1] = (-1.0f * yInc) + ((GLfloat)i * yInc);
		
		}
	}
#if 0//try
	//create temporary texture:
	IplImage * TempDst_Ilp = cvgCreateImage(cvGetSize(dst), GetCVDepth(dst), GetnChannels(dst));
	DataContainer * TempDest_DC = GPUCV_GET_TEX(TempDst_Ilp);//, DataContainer::LOC_GPU, false);

	//first pass
	std::string MetaTags;
	MetaTags = "$SOBEL_X_ORDER=";
	MetaTags += SGE::ToCharStr(xorder);
	MetaTags += "//$SOBEL_Y_ORDER=";
	MetaTags += SGE::ToCharStr(0);
	MetaTags += "//$SOBEL_APERTURE_SIZE=";
	MetaTags += SGE::ToCharStr(aperture_size);
	MetaTags += "//";

	TemplateOperator("cvgSobel", "FShaders/sobel", "",
		src, NULL, NULL,
		TempDst_Ilp, params, 18, TextureGrp::TEXTGRP_SAME_ALL, MetaTags);
		
	//second pass
	MetaTags = "$SOBEL_X_ORDER=";
	MetaTags += SGE::ToCharStr(0);
	MetaTags += "//$SOBEL_Y_ORDER=";
	MetaTags += SGE::ToCharStr(yorder);
	MetaTags += "//$SOBEL_APERTURE_SIZE=";
	MetaTags += SGE::ToCharStr(aperture_size);
	MetaTags += "//";

	TemplateOperator("cvgSobel", "FShaders/sobel", "",
		TempDst_Ilp, NULL, NULL,
		dst, params, 18, TextureGrp::TEXTGRP_SAME_ALL, MetaTags);
	
#else
	
	//first pass
	std::string MetaTags;
	MetaTags = "$SOBEL_X_ORDER=";
	MetaTags += SGE::ToCharStr(xorder);
	MetaTags += "//$SOBEL_Y_ORDER=";
	MetaTags += SGE::ToCharStr(yorder);
	MetaTags += "//$SOBEL_APERTURE_SIZE=";
	MetaTags += SGE::ToCharStr(aperture_size);
	MetaTags += "//";

	TemplateOperator("cvgSobel", "FShaders/sobel", "",
		src, NULL, NULL,
		dst, params, 18, TextureGrp::TEXTGRP_SAME_ALL, MetaTags);
#endif

	GPUCV_STOP_OP(
		cvSobel(src, dst, xorder, yorder,aperture_size),
		dst, src, NULL, NULL
			);
}
//=================================================================
//=================================================================
void cvgLaplace(CvArr* src, CvArr* dst, int aperture_size/*=3*/ )
{
	GPUCV_START_OP(cvLaplace(src, dst, aperture_size),
		"cvgLaplace",
		dst,
		GenericGPU::HRD_PRF_3);

	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");
	GCV_OPER_ASSERT(aperture_size==1 || aperture_size==3, "apperture size must be [1|3]");

	float params[18];// = {threshold/255., maxValue/255.};

	//calcul offset values:
	GLint i, j;
	GLfloat xInc, yInc;
	GLuint width  = ((IplImage*)dst)->width;
	GLuint height = ((IplImage*)dst)->height;

	xInc = 1.0f / (GLfloat)width;
	yInc = 1.0f / (GLfloat)height;

	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			params[(((i*3)+j)*2)+0] = (-1.0f * xInc) + ((GLfloat)i * xInc);
			params[(((i*3)+j)*2)+1] = (-1.0f * yInc) + ((GLfloat)j * yInc);
		}
	}

	std::string MetaTags;
	if(aperture_size==1)
	{
		MetaTags = "$APERTURE_SIZE_1=1//";
	}
	else if(aperture_size==3)
	{
		MetaTags = "$APERTURE_SIZE_3=1//";
		MetaTags += "//$APERTURE_SIZE=3//";
	}
	else
	{
		MetaTags += "$APERTURE_SIZE";
		MetaTags += SGE::ToCharStr(aperture_size);
		MetaTags += "//";
	}

	TemplateOperator("cvgLaplace", "FShaders/laplacian", "",
		src, NULL, NULL,
		dst, params, 18, TextureGrp::TEXTGRP_SAME_ALL, MetaTags);

	GPUCV_STOP_OP(
		cvLaplace(src, dst, aperture_size),
		dst, src, NULL, NULL
		);
}
//=================================================================
void cvgCanny(CvArr* src, CvArr* dst, double threshold1, double threshold2, int aperture_size)
{
	GPUCV_START_OP(cvCanny(src, dst, threshold1, threshold2, aperture_size),
		"cvgCanny",
		dst,
		GenericGPU::HRD_PRF_3);

	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");

	float params[2+2];// = {threshold/255., maxValue/255.};

	//calcul offset values:
	/*
	GLint i, j;
	GLfloat xInc, yInc;
	GLuint width  = ((IplImage*)dst)->width;
	GLuint height = ((IplImage*)dst)->height;


	xInc = 1.0f / (GLfloat)width;
	yInc = 1.0f / (GLfloat)height;

	for (i = 0; i < 3; i++)
	{
	for (j = 0; j < 3; j++)
	{
	params[(((i*3)+j)*2)+0] = (-1.0f * xInc) + ((GLfloat)i * xInc);
	params[(((i*3)+j)*2)+1] = (-1.0f * yInc) + ((GLfloat)j * yInc);
	}
	}
	*/
	params[0]=1.0f / (GLfloat)GetWidth(src);
	params[1]=1.0f / (GLfloat)GetHeight(src);
	params[2]=threshold1;
	params[3]=threshold2;


	std::string MetaTags;
	MetaTags = "//$APERTURE_SIZE=";
	MetaTags += SGE::ToCharStr(aperture_size);
	MetaTags += "//";
	//get a temporary destination image
	IplImage * TmpDsc = cvgCreateImage(cvGetSize(dst), IPL_DEPTH_16U, 3);

	TemplateOperator("cvgCanny", "FShaders/canny", "",
		src, NULL, NULL,
		TmpDsc, params, 20, TextureGrp::TEXTGRP_SAME_ALL, MetaTags);

	TemplateOperator("cvgCanny", "FShaders/canny_threshold", "",
		TmpDsc, NULL, NULL,
		dst, params, 20, TextureGrp::TEXTGRP_SAME_ALL, MetaTags);



	GPUCV_STOP_OP(
		cvCanny(src, dst, threshold1, threshold2, aperture_size),
		src, dst, NULL, NULL
		);
}
//CVG_IMGPROC__GRAD_EDGE_CORNER_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
//CVG_IMGPROC__SAMPLE_INTER_GEO_GRP
void cvgResize(  CvArr* src, CvArr* dst, int interpolation/*=CV_INTER_LINEAR*/)
{
	GPUCV_START_OP(cvResize(src, dst, interpolation),
		"cvgResize",
		dst,
		GenericGPU::HRD_PRF_2);

	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");

	std::string filterName = "";

	switch (interpolation)
	{
	case CV_INTER_NN  :
		filterName = "FShaders/resize";
		break;
	case CV_INTER_LINEAR :
		cvgResizeGLSLFct(src,dst, "CV_INTER_LINEAR");//,NULL);
		//cvResize(src,dst,interpolation);
		break;
	case CV_INTER_AREA  : cvResize(src,dst,interpolation);break;
	case CV_INTER_CUBIC : cvResize(src,dst,interpolation);break;
	}

	//	float params[4] = {src->width, src->height,dst->width, dst->height};
	//	GetFilterManager()->SetParams(chosen_filter, params, 4);
	if (filterName!="")
	{
		TemplateOperator("cvgResize", filterName, "",
			src, NULL, NULL,
			dst, NULL, 0);
		//   GpuFilter::FilterSize size = GetSize(dst);
		//	GetFilterManager()->Apply(chosen_filter, ,dst, &size);
	}

	GPUCV_STOP_OP(
		cvResize(src, dst, interpolation),
		src, dst, NULL, NULL
		);
}
//CVG_IMGPROC__SAMPLE_INTER_GEO_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
//CVG_IMGPROC__MORPHO_GRP
#define GPUCV_ERODE_OP	0
#define GPUCV_DILATE_OP 1
#define GPUCV_DEBUG_MORPHO 0
void cvgMorphOp(CvArr* src, CvArr* dst, IplConvKernel* element, int iterations, int mode)
{
	std::string FctName;
	switch(mode)
	{
	case GPUCV_ERODE_OP: FctName= "cvgErode()";break;
	case GPUCV_DILATE_OP: FctName= "cvgDilate()";break;
	}

	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");

	if (iterations > 1)
	{
		SG_Assert(CV_IS_IMAGE_HDR(src), FctName << ", iteration >1 => case not done for Cv_MAT");
	}

	if (!element)
		element = cvCreateStructuringElementEx( 3, 3, 1, 1, CV_SHAPE_RECT, 0 );

	GpuFilter::FilterSize size = GetSize(dst);

	//Test compatiblity, square structuring elements with odd-numbered size
	SG_Assert(element->nCols == element->nRows, "IplConvKernel size is currently limited to square size");
	SG_Assert(!IS_MULTIPLE_OF(element->nCols, 2), "IplConvKernel size is currently limited to odd-numbered size");
	//------------------

	//format meta-tags
#if 1
	std::string filterName = "FShaders/morpho";
	filterName += "($DEF_FILTER_WIDTH=";
	filterName += SGE::ToCharStr(element->nCols);
	filterName += "//$DEF_FILTER_HEIGHT=";
	filterName += SGE::ToCharStr(element->nRows);
	filterName += "$DEF_FCT=";
	filterName += (mode==GPUCV_ERODE_OP)?"min//)":"max//)";
	filterName += ".frag";
#else
	//
	std::string filterName = "FShaders/copy.frag";
#endif
	GpuFilter * CurrentFilter = GetFilterManager()->GetFilterByName(filterName);


	//set parameters
	int ParamsNbr = element->nCols * element->nRows +2;
	float *params = new float [ParamsNbr];
	params[0] = (float)GetWidth(src);
	params[1] = (float)GetHeight(src);

	for(int i=0;i< ParamsNbr - 2;i++)
		params[2+i] = (float)(element->values[i]);
	CurrentFilter->SetParams(params, ParamsNbr);
	delete params;
	//================

	DataContainer * SrcTex = GPUCV_GET_TEX(src);
	DataContainer * DstTex = GPUCV_GET_TEX(dst);

	SrcTex->PushSetOptions(DataContainer::UBIQUITY, true);
	SrcTex->PushSetOptions(DataContainer::DEST_IMG, false);
	DstTex->PushSetOptions(DataContainer::UBIQUITY, false);
	DstTex->PushSetOptions(DataContainer::DEST_IMG, true);
	DstTex->SetLocation<DataDsc_GLTex>(false);
	//run erode
	if (iterations == 1)
	{//only one loop, no temp tex
		CurrentFilter->Apply(SrcTex, DstTex, &size);
	}
	else
	{//multiple Loop, using temp tex
		DataContainer * TempSrcTex = NULL;
		DataContainer * TempDstTex = new DataContainer();
		DataContainer * TempSwitch = NULL;


		//we create a temp container that will contain a copy of Destination DataDsc_GLTex
		//copy textures properties but no data
		//TempSrcTex->_CopyProperties(*SrcTex);
		TempDstTex->_CopyProperties(*DstTex);//copy global properties of the container (ex:options...)
		DstTex->SetLocation<DataDsc_GLTex>(false);
		TempDstTex->CopyDataDsc<DataDsc_GLTex>(DstTex, false);
		TempDstTex->SetOption(DataContainer::CPU_RETURN|DataContainer::UBIQUITY, false);

#if _GPUCV_DEBUG_MODE
		SrcTex->PushSetOptions(CL_Options::LCL_OPT_DEBUG
			|DataContainer::DBG_IMG_LOCATION, true);
		DstTex->PushSetOptions(CL_Options::LCL_OPT_DEBUG
			|DataContainer::DBG_IMG_LOCATION, true);
		TempDstTex->PushSetOptions(CL_Options::LCL_OPT_DEBUG
			|DataContainer::DBG_IMG_LOCATION, true);
#endif

#if GPUCV_DEBUG_MORPHO
		cvNamedWindow("src", 1);
		cvNamedWindow("dst", 1);
#endif

		//perform first iteration out of the loop with real src texture
		CurrentFilter->Apply(SrcTex, TempDstTex, &size);

#if GPUCV_DEBUG_MORPHO
		cvgShowImage("src", SrcTex);
		cvgShowImage("dst", TempDstTex);
		cvWaitKey(0);
#endif
		//TempSrcTex = new DataContainer();
		//TempSrcTex->_CopyProperties(*TempDstTex);
		TempSrcTex = DstTex;// //new DataContainer();
		for (int i=0; i < iterations-1 ; i++)
		{
			TempSwitch = TempSrcTex;
			TempSrcTex = TempDstTex;
			TempDstTex = TempSwitch;
			TempSwitch = NULL;
			CurrentFilter->Apply(TempSrcTex, TempDstTex, &size);
#if GPUCV_DEBUG_MORPHO
			cvgShowImage("src", TempSrcTex);
			cvgShowImage("dst", TempDstTex);
			cvWaitKey(0);
#endif
		}

		//affect temp destination texture ID to real destination texture
		if(TempDstTex!=DstTex)
		{
			DstTex->SwitchDataDsc<DataDsc_GLTex>(TempDstTex);
		}


#if _GPUCV_DEBUG_MODE
		SrcTex->PopOptions();
		DstTex->PopOptions();
#endif
		SrcTex->PopOptions();
		DstTex->PopOptions();
		//release temp data
		if(TempDstTex!=DstTex)
			delete TempDstTex;
		if(TempSrcTex!=DstTex)
			delete TempSrcTex;

		//	cvDestroyWindow("src");
		//	cvDestroyWindow("dst");

	}
}
//===========================================================
void cvgDilate(CvArr* src, CvArr* dst, IplConvKernel* element, int iterations )
{
	GPUCV_START_OP(cvDilate(src, dst, element, iterations),
		"cvgDilate",
		dst,
		GenericGPU::HRD_PRF_3);

	cvgMorphOp(src, dst, element, iterations, GPUCV_DILATE_OP);

	GPUCV_STOP_OP(
		cvDilate(src,dst,element,iterations),
		src, dst, NULL, NULL
		);
}
//=================================================================
void cvgErode(CvArr* src, CvArr* dst, IplConvKernel* element, int iterations)
{
	GPUCV_START_OP(cvErode(src, dst, element, iterations),
		"cvgErode",
		dst,
		GenericGPU::HRD_PRF_3);

	cvgMorphOp(src, dst, element, iterations, GPUCV_ERODE_OP);

	GPUCV_STOP_OP(
		cvErode(src,dst,element,iterations),
		src, dst, NULL, NULL
		);
}
//=================================================================
void cvgMorphologyEx(CvArr* src, CvArr* dst, CvArr* temp, IplConvKernel* B, CvMorphOp op, int iterations )
{
	GPUCV_START_OP(cvMorphologyEx( src, dst, temp, B, op, iterations ),
		"cvgMorphologyEx",
		dst,
		GenericGPU::HRD_PRF_3);


	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");
	GCV_OPER_ASSERT(temp, "no temp images!");
	GCV_OPER_ASSERT(B, "no IplConvKernel!");
	
	bool cpuA = (cvgGetOptions(src, DataContainer::CPU_RETURN))?true:false;
	bool cpuC = (cvgGetOptions(dst, DataContainer::CPU_RETURN))?true:false;
	bool cputemp =  (cvgGetOptions(temp, DataContainer::CPU_RETURN))?true:false;


	cvgPushSetOptions(src, DataContainer::CPU_RETURN, false);
	cvgPushSetOptions(dst, DataContainer::CPU_RETURN, false);
	cvgPushSetOptions(temp, DataContainer::CPU_RETURN, false);

	bool execute=true;

	switch (op)
	{
	case CV_MOP_OPEN:
		{
			//dst=open(A,B)=dilate(erode(A,B),B)
			cvgErode(src, temp, B, iterations);
			cvgDilate(temp, dst, B, iterations);
			break;
		}

	case CV_MOP_CLOSE:
		{
			//dst=close(src,B)=erode(dilate(src,B),B)
			cvgDilate(src, temp, B, iterations);
			cvgErode(temp, dst, B, iterations);
			break;
		}

	case CV_MOP_GRADIENT:
		{
			//dst=morph_grad(src,B)=dilate(src,B)-erode(src,B),  if op=CV_MOP_GRADIENT
			cvgDilate(src, temp, B, iterations);
			cvgErode(src, dst, B, iterations);
			cvgSub(temp, dst, dst);

			if (cputemp)
			{
				//GPUCV_GET_TEX_ON_LOC(src,DataContainer::LOC_CPU, true); // necessary ??
				cvgSynchronize(src);
			}
			break;
		}

	case CV_MOP_TOPHAT:
		{
			//dst=tophat(src,B)=src-erode(src,B),   if op=CV_MOP_TOPHAT
			cvgErode(src, dst, B, iterations);
			cvgSub(src, dst, dst);
			break;
		}

	case CV_MOP_BLACKHAT:
		{
			//dst=blackhat(src,B)=dilate(src,B)-src,   if op=CV_MOP_BLACKHAT
			cvgDilate(src, dst, B, iterations);
			cvgSub(dst, src, dst);
			break;
		}

	default : execute = false; break;

	}

	cvgPopOptions(src);
	cvgPopOptions(dst);
	cvgPopOptions(temp);

	if (execute)
		if (cpuC)
		{
			//	GPUCV_GET_TEX_ON_LOC(dst, DataContainer::LOC_CPU, true);
			cvgSynchronize(dst);
		}

		GPUCV_STOP_OP(
			cvMorphologyEx( src, dst, temp, B, op, iterations ),
			src, dst, temp, NULL
			);
}
//CVG_IMGPROC__MORPHO_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
//CVG_IMGPROC__FILTERS_CLR_CONV_GRP
#if _GPUCV_DEVELOP_BETA
void cvgSmooth(CvArr* src, CvArr* dst, int smoothtype/*=CV_GAUSSIAN*/, int param1/*=3*/, int param2/*=0*/, double param3/*=0*/, double param4/*=0*/)
{
	GPUCV_START_OP(cvSmooth(src, dst, smoothtype,param1, param2, param3, param4),
		"cvgSmooth",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src, "no input images!");
	GCV_OPER_ASSERT(dst, "no destination image!");

	float params[18];// = {threshold/255., maxValue/255.};

	//calcul offset values:
	GLint i, j;
	GLfloat xInc, yInc;
	GLuint width  = ((IplImage*)dst)->width;
	GLuint height = ((IplImage*)dst)->height;


	xInc = 1.0f / (GLfloat)width;
	yInc = 1.0f / (GLfloat)height;

	for (i = 0; i < 3; i++)
	{
		for (j = 0; j < 3; j++)
		{
			params[(((i*3)+j)*2)+0] = (-1.0f * xInc) + ((GLfloat)i * xInc);
			params[(((i*3)+j)*2)+1] = (-1.0f * yInc) + ((GLfloat)j * yInc);
		}
	}
	//---

	std::string MetaTags;
	MetaTags = "$SOBEL_X_ORDER=";
	MetaTags += SGE::ToCharStr(xorder);
	MetaTags += "//$SOBEL_Y_ORDER=";
	MetaTags += SGE::ToCharStr(yorder);
	MetaTags += "//$SOBEL_APERTURE_SIZE=";
	MetaTags += SGE::ToCharStr(aperture_size);
	MetaTags += "//";

	TemplateOperator("cvgSobel", "FShaders/sobel", "",
		src, NULL, NULL,
		dst, params, 18, TextureGrp::TEXTGRP_SAME_ALL, MetaTags);



	cvgSetOptions(dst, DataContainer::DEST_IMG, false);


	GPUCV_STOP_OP(
		cvSmooth(src, dst, smoothtype,param1, param2, param3, param4),
		src, dst, NULL, NULL
		);

}
#endif//_GPUCV_DEVELOP_BETA
//====================================================
void cvgCvtColor(  CvArr* src, CvArr* dst, int code )
{
	GPUCV_START_OP(cvCvtColor(src, dst, code ),
		"cvgCvtColor",
		dst,
		GenericGPU::HRD_PRF_2);

		GCV_OPER_ASSERT(src, "no input images!");
		GCV_OPER_ASSERT(dst, "no destination image!");

#if 0//_USE_IMG_FORMAT
	string Seqbackup;//used only if _USE_IMG_FORMAT is defined
	//Yann.A. 30/11/2005 : patch to correct image format conversion...
	//not working yet...
	Seqbackup = dst->channelSeq;
	switch(cvgConvertCVTexFormatToGL(dst))
	{
	case GL_RGB: strcpy(dst->channelSeq, "RGB");break;
	case GL_BGR: strcpy(dst->channelSeq, "BGR");break;
	default : GPUCV_WARNING("\nWarning:cvgCvtColor -> Unknown channelSeq");
		Seqbackup=="";
	}
	printf("\n Dst seq : %s => new Seq : %s",Seqbackup.data(), dst->channelSeq);
#endif
	std::string FilterName;

	switch (code)
	{
	case  CV_RGB2XYZ: FilterName="FShaders/RGB2XYZ"; break;
	case  CV_BGR2XYZ: FilterName="FShaders/BGR2XYZ"; break;
	case  CV_XYZ2RGB: FilterName="FShaders/XYZ2RGB"; break;
	case  CV_XYZ2BGR: FilterName="FShaders/XYZ2BGR"; break;
	case  CV_RGB2YCrCb: FilterName="FShaders/RGB2YCrCb"; break;
	case  CV_BGR2YCrCb: FilterName="FShaders/BGR2YCrCb"; break;
	case  CV_YCrCb2RGB: FilterName="FShaders/YCrCb2RGB"; break;
	case  CV_YCrCb2BGR: FilterName="FShaders/YCrCb2BGR"; break;
	case  CV_RGB2HSV: FilterName="FShaders/RGB2HSV"; break;
	case  CV_BGR2HSV: FilterName="FShaders/BGR2HSV"; break;
	case  CV_RGB2Lab: FilterName="FShaders/RGB2Lab"; break;
	case  CV_BGR2Lab: FilterName="FShaders/BGR2Lab"; break;
	case  CV_RGB2GRAY:
		SG_Assert(GetnChannels(dst) == 1, "RGB to GRAY conversion : nb of destination image channels is not correct.\n");
		FilterName="FShaders/RGB2GRAY";break;
	case  CV_BGR2GRAY:
		SG_Assert(GetnChannels(dst) == 1, "BGR to GRAY conversion : nb of destination image channels is not correct.\n");
		FilterName="FShaders/BGR2GRAY";break;
	default :
		SG_Assert(0, "Case not done for code = " << code);
		break;
	}

	TemplateOperator("cvgCvtColor", FilterName, "",
		src, NULL, NULL,
		dst, NULL, 0);

#if 0// _USE_IMG_FORMAT
	//Yann.A. 30/11/2005 : patch to correct image format conversion...
	if ( Seqbackup!="")
		strcpy(dst->channelSeq, Seqbackup.data());
#endif

	GPUCV_STOP_OP(
		cvCvtColor(src, dst, code ),
		dst, src, NULL, NULL
		);
}
//=================================================================
void cvgThreshold(  CvArr* src, CvArr* dst, double threshold, double maxValue, int thresholdType )
{
	GPUCV_START_OP(cvThreshold(src, dst, threshold, maxValue, thresholdType),
		"cvgThreshold",
		dst,
		GenericGPU::HRD_PRF_3);

		GCV_OPER_ASSERT(src, "no input images!");
		GCV_OPER_ASSERT(dst, "no destination image!");

	string chosen_filter;
	string filterName;
	switch(thresholdType)
	{
	case CV_THRESH_BINARY :		filterName="FShaders/threshold_binary";break;
	case CV_THRESH_BINARY_INV : filterName="FShaders/threshold_binary_inv";break;
	case CV_THRESH_TRUNC :		filterName="FShaders/threshold_trunc";break;
	case CV_THRESH_TOZERO :		filterName="FShaders/threshold_tozero";break;
	case CV_THRESH_TOZERO_INV :	filterName="FShaders/threshold_tozero_inv";break;
	default : SG_Assert(0, "Unkown type " << thresholdType); break;
	}

	float params[2] = {threshold/255., maxValue/255.};

	TemplateOperator("cvgThreshold", filterName, "",
		src, NULL, NULL,
		dst, params, (thresholdType == CV_THRESH_BINARY || thresholdType == CV_THRESH_BINARY_INV)? 2:1);

	GPUCV_STOP_OP(
		cvThreshold(src, dst, threshold, maxValue, thresholdType),
		src, dst, NULL, NULL
		);
}
//=================================================================
//CVG_IMGPROC__FILTERS_CLR_CONV_GRP
//_______________________________________________________________
//_______________________________________________________________
