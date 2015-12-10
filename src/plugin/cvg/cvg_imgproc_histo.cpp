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



/** \brief C++ File containg definitions GPU equivalents for openCV functions: Image Processing -> Histograms
\author Jean-Philippe Farrugia, Yannick Allusse
*/
#include "StdAfx.h"
#include <cvg/cvg.h>
#include <cxcoreg/cxcoreg.h>
#include <highguig/highguig.h>
#include <GPUCV/misc.h>


using namespace GCV;

//_______________________________________________________________
//CVG_IMGPROC__HISTOGRAM_GRP
#if 1 // use ARB_imaging
void cvgCalcHist(IplImage ** _src, CvHistogram* hist, int accumulate/*=0*/, const CvArr* mask/*=NULL*/)
{
	GPUCV_START_OP(,
		"cvgQueryHistValue",
		*_src,
		GenericGPU::HRD_PRF_3);

	if(GLEW_ARB_imaging)
	{//call OpenGL histogram
		//Get image on GPU
		CvgArr * gpuImg = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(*_src));
		gpuImg->SetOption(DataContainer::UBIQUITY, true);
		DataDsc_GLTex * glImg = gpuImg->SetLocation<DataDsc_GLTex>();

		//init OpenGL context
		GLuint GLFormat = GL_LUMINANCE;
		int Bins = hist->mat.dim[0].size;
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glHistogram(GL_HISTOGRAM, Bins, GLFormat, GL_FALSE);
		glEnable(GL_HISTOGRAM);

		glClearColor(0.5, 0.5, 0.5, 0.0);

		int width = 512;
		int height = 512;
#if 1
		//glImg->InitGLView();
		//glEnable(GL_SCISSOR_TEST);
		//glScissor(0,0,width,height);
		glDrawBuffer(GL_BACK);
		glViewport(0, 0, width, height);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		//gluOrtho2D(-1, 1, -1, 1);
		glOrtho(0, 512, 0, 512, -100.0, 100.0);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		//glLoadIdentity();
		//glOrtho(0, 512, 0, 512, -1.0, 1.0);
		//gluPerspective(30., 4/3., 0.1, 10000);
		// glMatrixMode(GL_MODELVIEW);

#else
		glViewport(0, 0, (GLsizei) 512, (GLsizei) 512);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0, 256, 0, 10000, -1.0, 1.0);
		glMatrixMode(GL_MODELVIEW);
#endif

		//draw image to compute
		glClear(GL_COLOR_BUFFER_BIT);
#if 1
		//glRasterPos2i(256, 1);
		glTranslatef(256,256,0);
		glScalef(256,256,1);
		//glImg->_Bind();
		glImg->DrawFullQuad( width,height);
		//glDrawBuffer(GL_AUX0);
		glDrawBuffer(GL_FRONT);
		glReadBuffer(GL_BACK);
		glCopyPixels(0, 0, 512, 512, GL_COLOR);
		//glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);
#else
		glRasterPos2i(1, 1);
		glDrawPixels(512,512, GL_RGB, GL_UNSIGNED_BYTE, (*_src)->imageData);
#endif


		//get back result
		GLushort *values = NULL;
		if(GLFormat == GL_LUMINANCE)
			values = new GLushort[Bins];
		else
			values = new GLushort[Bins*3];
		glFlush();
		glFinish();
#if _GPUCV_GL_USE_GLUT
#if 0//_DEBUG
		glutSwapBuffers();
#endif
#endif

		glGetHistogram(GL_HISTOGRAM, GL_TRUE, GLFormat, GL_UNSIGNED_SHORT, values);

		_GPUCV_GL_ERROR_TEST();

		if(GLFormat == GL_LUMINANCE)
		{
			for(int i = 0; i  < Bins;i++)
			{
				//	printf("\n i:%d val:%d",i,values[i]);
				((CvMatND*)(&hist->mat))->data.i[i] = values[i];
			}
		}
		else
		{
			for(int i = 0; i  < Bins;i++)
			{
				//	printf("\n i:%d val:%d",i,values[i*3]);
				((CvMatND*)(&hist->mat))->data.i[i] = values[i*3];
			}
		}

		//clean everything
		glDisable(GL_HISTOGRAM);
	}

	GPUCV_STOP_OP(,
		NULL, NULL, NULL, NULL
		);
}
#else
void cvgCalcHist(IplImage** src,CvHistogram* hist,int accumulate/*=0*/, const CvArr* mask/*=NULL*/)//GLuint liste2, GLuint RGBA_Texture/*=0*/)
{
	GPUCV_START_OP(cvCalcHist(src, hist, accumulate, mask),
		"cvgCalcHist2",
		*src,
		GenericGPU::HRD_PRF_3);

	SG_Assert(*src, "No SourceImage");
	SG_Assert(hist, "No histogram");
	SG_Assert(accumulate==0, "cvgCalcHist do not support accumulate flag");
	SG_Assert(mask==NULL, "cvgCalcHist do not support mask image");


	getGLContext()->PushAttribs();
	GLfloat HistoRange[2], HistoStep;
	HistoRange[0] = 0.;
	HistoRange[1] = 256.;
	HistoStep = 1;
	CvSize HistoSize;
	HistoSize.height = 2;
	HistoSize.width  = int(HistoRange[1]/HistoStep);
	//for debug
	HistoSize.height = HistoSize.width;

	CvArr * IplDest = cvgCreateImage(HistoSize, IPL_DEPTH_8U, 3);

	//for now we start with texture on CPU.
	DataContainer * CvgSrc = GPUCV_GET_TEX(*src);//, DataContainer::LOC_CPU, true);
	CvgSrc->SetLocation<DataDsc_CPU>(true);
	CvgSrc->SetLabel("HistoSource");

	//create destination texture
	DataContainer * TempDest = GPUCV_GET_TEX(IplDest);
	TempDest->SetLabel("HistoDest");

	// params[2] = {threshold/255., maxValue/255.};
	float params[2];
	params[0] =  src[0]->width;
	params[1] =  src[0]->height;

#if _GPUCV_DEBUG_MODE
	CvgSrc->PushSetOptions(CL_Options::LCL_OPT_DEBUG, 1);
	TempDest->PushSetOptions(CL_Options::LCL_OPT_DEBUG, 1);
#endif
	//new DataContainer(GL_RGB_FLOAT16_ATI, HistoRange[1]/HistoStep, 1., GL_LUMINANCE, GL_FLOAT, NULL, false);

	//buffer size
	//GLuint buffSize = CvgSrc->_GetWidth() * CvgSrc->_GetHeight() * CvgSrc->_GetNChannels() * sizeof(GLubyte);
	//create and bind vbo
	//GLBuffer VertexSrcBuffer(GL_ARRAY_BUFFER, GLBuffer::STREAM_DRAW, buffSize, (GLuint*)CvgSrc->_GetPixelsData());


	/*TempDest->_ForceLocation(DataContainer::LOC_GPU);
	//TempDest->SetRenderToTexture();
	InitGLView(TempDest->_GetWidth(), TempDest->_GetHeight());
	VertexSrcBuffer.DrawArrays(GL_POINTS, 0, buffSize/sizeof(GLubyte));
	TempDest->UnsetRenderToTexture();
	*/
#if 1
	TemplateOperator("cvgCalcHist",
		"FShaders/histogram_vs","VShaders/histogram_vs",
		//"","",//"FShaders/histogram_yannick","VShaders/histogram_yannick",
		*src, NULL, NULL,
		IplDest,
		params, 2,
		//NULL, 0,
		TextureGrp::TEXTGRP_NO_CONTROL, "",
		//NULL);
		CalcHisDrawFct);
#endif
	/*
	cvNamedWindow("cvgHisto()",1);
	cvgShowImage("cvgHisto()", IplDest);
	cvWaitKey(0);
	cvDestroyWindow("cvgHisto()");
	*/

#if _GPUCV_DEBUG_MODE
	CvgSrc->PopOptions();
	TempDest->PopOptions();
#endif

	getGLContext()->PopAttribs();

	GPUCV_STOP_OP(
		cvCalcHist(src, hist, accumulate, mask),//in case if error this operator is called
		src, NULL, NULL, NULL //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
#endif
//===========================================================
float cvgQueryHistValue(CvArr* src,int color)
{
	GPUCV_START_OP(return 0,
		"cvgQueryHistValue",
		src,
		GenericGPU::HRD_PRF_3);

	GLuint query;
	unsigned int result;
	CvgArr * temp_Arr = new CvgArr(&src);
	DataContainer * cvgSrc =  GPUCV_GET_TEX(src);

	// Setting occlusion query
	glGenQueriesARB(1, &query);
	glEnable (GL_CULL_FACE);
	glBeginQueryARB(GL_SAMPLES_PASSED_ARB, query);

	float params[2] = {(color)/256., (color+1)/256.};

	TemplateOperator("cvgQueryHistValue", "FShaders/query_hist_value.frag", "",
		cvgSrc, NULL, NULL,
		temp_Arr, params, 2);

	glFlush();
	glEndQueryARB(GL_SAMPLES_PASSED_ARB);
	// Getting result
	glGetQueryObjectuivARB(query, GL_QUERY_RESULT_ARB, &result);

	glDisable (GL_CULL_FACE);
	glDeleteQueriesARB(1, &query);
	return((float)result);
	GPUCV_STOP_OP(return 0 ,
		src, NULL, NULL, NULL
		);
	return 0;
}


//CVG_IMGPROC__HISTOGRAM_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
