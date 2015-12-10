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
/**
Jean-Philippe Farrugia
Yannick Allusse
*/
#include "StdAfx.h"
#include <GPUCV/CvgOperators.h>
#include <GPUCVHardware/GLContext.h>
//---------------------
#if _GPUCV_DEBUG_MODE
#define _DEBUG_HISTVALUE 0
#endif

using namespace GCV;


void TestGeoDrawFunction(TextureGrp*_grpIn,TextureGrp*_grpOut, GLuint a, GLuint b)
{
	glPushMatrix();
	glScalef(0.5,0.5,1.);
	glBegin(GL_LINE_STRIP);
	{

		for (int j = 0; j < 4; j++)
		{
			glColor4f(1.,1.,1.,1.);
			glScalef(0.8,0.8,1.);
			for (int i = 0; i < 4; i++)
				glVertex3fv(&GPUCVDftQuadVertexes[i*3]);
		}
		/*
		if (m_textureType ==  GL_TEXTURE_2D || m_textureType == GL_TEXTURE_RECTANGLE_ARB)
		{
		if(GlobTexCoord)
		{
		for(int i = 0; i<4; i++)
		{
		glTexCoord2dv(GlobTexCoord->operator [](i*2));
		glVertex3fv(&GPUCVDftQuadVertexes[i*3]);
		//glVertex3f(Vertexes[i].x,Vertexes[i].y,Vertexes[i].z);
		}
		}
		else if(m_mainAttachement == DataDsc_GLTex::NO_ATTACHEMENT)
		{
		for(int i = 0; i<4; i++)
		{
		glTexCoord2dv(&GPUCVDftTextCoord[i*2]);
		glVertex3fv(&GPUCVDftQuadVertexes[i*3]);
		//glVertex3f(Vertexes[i].x,Vertexes[i].y,Vertexes[i].z);
		}
		}
		else
		{

		for(int i = 0; i<4; i++)
		{
		DataDsc_GLTex * CurTexGL =NULL;
		TEXTURE_GRP_INTERNE_DO_FOR_ALL(
		TEXT,
		CurTexGL = TEXT->GetDataDsc<DataDsc_GLTex>();
		CurTexGL->_GlMultiTexCoordARB(i);
		);
		}
		}
		}
		else
		{
		SG_Assert(0, "Unknown texture type, ID: " << m_textureType);
		}
		*/
		/*else if (m_textureType ==  GL_TEXTURE_RECTANGLE_ARB)
		{
		itObj itText;
		for(int i = 0; i<4; i++)
		{
		for( itText = m_TextVect.begin(); itText != m_TextVect.end(); itText++)
		(*itText)->GlMultiTexCoordARB(i);
		glVertex3f(Vertexes[i].x,Vertexes[i].y,Vertexes[i].z);
		}


		if (x!=0 && y!=0)
		{
		glTexCoord2f(0, 0); glVertex3f(-1, -1, -0.5f);
		glTexCoord2f(x, 0); glVertex3f( 1, -1, -0.5f);
		glTexCoord2f(x, y); glVertex3f( 1,  1, -0.5f);
		glTexCoord2f(0, y); glVertex3f(-1,  1, -0.5f);
		}
		else
		{   string Msg = "Critical : drawQuad()=> Using GL_TEXTURE_RECTANGLE_ARB and image size is equal to O\n";
		GPUCV_ERROR(Msg.data());
		}

		}
		*/
	}
	glEnd();
	glPopMatrix();
}

void cvgTestGeometryShader(CvArr* src, CvArr* dst)
{
	GPUCV_START_OP(,
		"cvgTestGeometryShader",
		dst,
		GenericGPU::HRD_PRF_2);

	float Params [2] = {0,0};//{scale*256., shift};
	std::string MetaOptions = "$DEF_SCALE=1//";
	//set groups
	TextureGrp InGrp;
	TextureGrp OutGrp;
	InGrp.SetGrpType(TextureGrp::TEXTGRP_INPUT);
	OutGrp.SetGrpType(TextureGrp::TEXTGRP_OUTPUT);
	InGrp.AddTexture(GPUCV_GET_TEX(src));
	OutGrp.AddTexture(GPUCV_GET_TEX(dst));
	//=======
	//Set shader files
	ShaderProgramNames ShFiles;
	ShFiles.m_ShaderNames[0] = "FShaders/test_geo.frag";
	ShFiles.m_ShaderNames[1] = "VShaders/test_geo.vert";
	ShFiles.m_ShaderNames[2] = "GShaders/test($GEO_INPUT=GL_POINTS$GEO_OUTPUT=GL_POINTS).geo";

	CvgArr * dstCvg			 = (CvgArr *)GPUCV_GET_TEX(dst);
	DataDsc_GLTex * dstGLTex = dstCvg->GetDataDsc<DataDsc_GLTex>();
	//			dstGLTex->_SetTexType(GL_TEXTURE_2D_ARRAY_EXT);

#if 0
	glColor4f(1.,1.,0.,0.);
	TemplateOperatorGeo("cvgTestGeometryShader", ShFiles,
		&InGrp, &OutGrp, NULL, 0, TextureGrp::TEXTGRP_NO_CONTROL, "", TestGeoDrawFunction);

#else
	ShaderObject * GeoShader = new ShaderObject(NULL);
	GeoShader->cvLoadShaders(ShFiles);
	_GPUCV_GL_ERROR_TEST();
	GLuint GeoShHandle = GeoShader->GetShaderHandle();
	_GPUCV_GL_ERROR_TEST();

	dstGLTex->SetRenderToTexture();
	glViewport( 0, 0, 256, 256);
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective( 40.0f, 1.0f, 0.5f, 15.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef( 0.0f, 0.0f, -8.0f);

	glUseProgramObjectARB(GeoShHandle);
	_GPUCV_GL_ERROR_TEST();
	glColor4f(1.,0.,1.,0.);
	glBegin(GL_POINTS);
	glVertex3d(-1.0,-1.0,0.0);
	glVertex3d(1.0,0.0,0.0);

	glVertex3d(-1.0,1.0,0.0);
	glVertex3d(-1.0,-1.0,0.0);

	glVertex3d(-1.0,-1.0,0.0);
	glVertex3d(1.0,0.0,0.0);
	glEnd();
	glUseProgramObjectARB(0);
	_GPUCV_GL_ERROR_TEST();
	glFlush();
	dstGLTex->UnsetRenderToTexture();
	//		glutSwapBuffers();
	//		glutSwapBuffers();
	//		glutSwapBuffers();
	_GPUCV_GL_ERROR_TEST();
#endif

	GPUCV_STOP_OP(
		,//in case if error this operator is called
		src, dst, NULL, NULL//in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}

#if 0//_GPUCV_DEVELOP_BETA
/*!
*	\author Yannick Allusse
*/
void cvgXor(CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask/*=NULL*/ )
{
	SetThread();
	string Msg;
	string FctName="cvgXor";

	//test images properties----------------------------------------------------
	if (!ImgSizeComp3(src1, src2, dst, true,FctName))           return;
	else if(!ImgnChannelsComp3(src1, src2, dst, true,FctName))  return;
	else if ( !ImgDepthComp3(src1,src2,dst, true,FctName))      return;
	//--------------------------------------------------------------------------

	string Filtername = "FShaders/XorPixel.frag";
	string chosen_filter = GetFilterManager()->GetFilterName(Filtername);
	if (mask==NULL)
	{
		CvArr ** tab ;
		int nb = 0;
		tab = new CvArr*[1];
		tab[0] = (CvArr*)src2;
		nb = 1;

		GetFilterManager()->Apply(chosen_filter, src1, dst,
			src1->width, src1->height,
			tab, nb);
		delete [] tab;
	}
	else
		cvXor(src1,src2,dst,mask);
}



void AcqAddRed(CvArr * A, CvArr * B, CvArr* C)
{
	if (!GetHardProfile()->IsCompatible(GenericGPU::HRD_PRF_2))
	{
		cvAdd(A, B, C, NULL);
		return;
	}

	SetThread();
	string chosen_filter;

	chosen_filter = GetFilterManager()->GetFilterName("FShaders/AcqAddRed.frag");

	if ( !ImgSizeComp3(A,B,C))
	{
		GPUCV_ERROR("Critical : can't apply cvgAdd, Image size doesn't match !\n");
	}
	else
	{
		CvArr ** tab ;
		int nb = 0;
		tab = new CvArr*;
		(*tab) = (CvArr*)B;
		nb = 1;

		GetFilterManager()->Apply(chosen_filter,
			A,C,
			A->width,A->height,
			tab,nb);
		delete [] tab;
	}
	UnsetThread();
}



/*!
*	\author Yannick Allusse
*/
CvScalar AcqDistanceAvg(CvArr * carte, CvArr * mask)
{
	//check image format
	/*	if (src==NULL)
	{
	GPUCV_ERROR("\nCritical : cvgAvg() => src is NULL.");
	return cvAvg(src, mask);
	}
	else if ((src->nChannels > 1) || (src->depth != 8))
	{
	GPUCV_NOTICE("\nNotice : cvgAvg() => Source image is not single channel or not 8-bits.");
	return cvAvg(src, mask);
	}
	*/

	//----------------------------------
	SetThread();
	//	unsigned int Pixel;
	int Ipixel=0;
	CvScalar Result;
	//do it on GPU
	CvSize Size1= {32,32};//{src->width, src->height};

	if (carte->width > 1024*1024)
		Size1.width = 64;
	if (carte->height > 1024*1024)
		Size1.height = 64;

	//CvSize Size2= {1,1};
	long unsigned int GPUaverage=0;
	//unsigned int GPUMax=0;
	//unsigned int GPUMin=100000;
	CvArr * dstGPU=NULL;
	dstGPU=cvgCreateImage(Size1, 8, 3);
	cvgResizeGLSLFct(carte,dstGPU, "AVG_ACQ", mask);
	//reads Average
	Ipixel = (Size1.width * Size1.height*3) -1;
	unsigned long int PixelNbr = 0;
	unsigned long int TotalPixelNbr = 0;
	unsigned long int Sum=0;

	GPUaverage = 0;
	Result.val[0] = Result.val[1] = Result.val[2] = Result.val[3] = 0;
	int R=0, B=0, G=0;
	while(Ipixel>=0)
	{
		R=(unsigned char)dstGPU->imageData[Ipixel];
		G=(unsigned char)dstGPU->imageData[Ipixel-1];
		B=(unsigned char)dstGPU->imageData[Ipixel-2];
		PixelNbr = G*256+B;
		Sum +=R*PixelNbr;
		//	printf("\nAverage Data Pix %d: pixel nbr: %d | %d sum : %d (%d)",Ipixel,G, B, Sum, R);//, data[Ipixel]);
		Ipixel-=3;
		TotalPixelNbr+= PixelNbr;
	}
	if (TotalPixelNbr)
		Result.val[0] = ((float)Sum)/((float)TotalPixelNbr);
	else
		Result.val[0] = 0;

	cvgReleaseImage(&dstGPU);
	UnsetThread();

	return Result;





}
#endif//beta






#if _GPUCV_DEVELOP_BETA
//#endif
/*!
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvHoughCircles" target=new>cvDistTransform</a> function using geometric objects.
*  Detects circles in grayscale image using Hough transform
*	\param image -> Source 8-bit single-channel (grayscale) image.
*	\param circle_storage -> pointer to memory storage for detected circles. In openCV, this could be a cvSeq* sequence or a single cvMat* row/column matrix.
*	\param method -> Currently, the only implemented method is CV_HOUGH_GRADIENT, which is basically 21HT, described in
*	\param dp -> Resolution of the accumulator used to detect centers of the circles. For example, if it is 1, the accumulator will have the same resolution as the input image, if it is 2 - accumulator will have twice smaller width and height, etc.
*	\param min_dist -> Minimum distance between centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
*	\param param1 -> The first method-specific parameter. In case of CV_HOUGH_GRADIENT it is the higher threshold of the two passed to Canny edge detector (the lower one will be twice smaller).
*	\param param2 -> The second method-specific parameter. In case of CV_HOUGH_GRADIENT it is accumulator threshold at the center detection stage. The smaller it is, the more false circles may be detected. Still, circles, corresponding to the larger accumulator values, will be returned first.
*	\return sequence containing detected circles in source image.
*	\warning All features are disabled for now, gpuCV implementation of HoughCircles is barely in alpha version. More to come...
*       \author Jean-Philippe Farrugia
*/
CvSeq* cvgHoughCircles(CvArr* image, void* circle_storage,
					   int method, double dp, double min_dist,
					   double param1, double param2)
{

	// Error processing
	if(method != CV_HOUGH_GRADIENT)
	{
		GPUCV_ERROR("Error : currently, only CV_HOUGH_GRADIENT is supported. Switching to OpenCV...\n");
		cvHoughCircles(image, circle_storage, method, dp, min_dist,param1, param2);
	}


	SetThread();
	// Getting (eventually)back source image on CPU
	GetTextureManager()->GetIplImage(image);
	// Creating Image for storing blending circles
	CvArr* result_circles = cvCreateImage(cvSize(image->width,image->height),8,1);

	// The following treatment is a test : it is subject to heavy changes
	// Ideally Finding contours on source image. TODO : check method
	// For testing purpose : grayscale image is actually  binary
	// Reading source image and Drawing circles for each white pixel in image


	// Loop on testing radiuses : from 100 to 200
	for(int radius = 20;radius<=100;radius++)
	{
		unsigned char* result_tab = new unsigned char[image->width*image->height];

		// Activating Blending
		// TODO : activate Color and disable Texture mapping

		// Forcing rendering in result_circles
		RenderBufferManager()->Force(GetTextureManager()->GetGpuImage(result_circles));
		glClear(GL_COLOR_BUFFER_BIT);
		glDisable(GL_TEXTURE_2D);
		glBlendFunc(GL_ONE, GL_ONE);
		glEnable(GL_BLEND);
		glColor3f(1.0/255.0,1.0/255.0,1.0/255.0);
		// Loop on source pixels and drawing circles with the current radius
		for(int x=0;x<image->width;x++)
			for(int y=0;y<image->height;y++)
			{
				long int offset = y*image->width+x;
				if(image->imageData[offset] != 0) drawIntCircle(radius,x,y,image->width,image->height,10);
			}
			// Displaying circles image for DEBUG
			cvgShowFrameBufferImage("cvgHoughCircles Result", image->width,image->height,GL_RGB, GL_UNSIGNED_BYTE);
			glEnable(GL_TEXTURE_2D);
			glDisable(GL_BLEND);
			RenderBufferManager()->UnForce();

			// We now need to find the coordinates of bright pixels.
			// Getting back circles image and displaying it for debug

			// If the pixel lum is greater than a fixed threshold, then it is a circle center
			// Clearing buffer when done
	}
	return(NULL);
}


#endif

#if _GPUCV_DEVELOP_BETA
/*!
*	\brief compute a erosion on src and store it on dst, kernel is a quare

*	\param src -> source image
*	\param dst -> destination image
*	\param element -> kenel (not used)
*	\param iterations -> number of iterations (not used)
*	\return none
*/
void cvgErodeSquare(  CvArr* src, CvArr* dst, IplConvKernel* element, int iterations)
{
	SetThread();

	string chosen_filter = GetFilterManager()->GetFilterName("FShaders/erodeSquare.frag");

	// Image Dimensions
	float params[4] = {GetWidth(src), GetHeight(src),element->nCols,element->nRows};
	GetFilterManager()->SetParams(chosen_filter, params, 4);

	GpuFilter::FilterSize size = GetSize(dst);
	GetFilterManager()->Apply(chosen_filter, src,dst, &size);

	UnsetThread();
}
#endif
//=============================================================
float cvgIsDiff(CvArr* src1, CvArr* src2)
{
	GPUCV_START_OP(return false,
		"cvgIsDiff",
		src1,
		GenericGPU::HRD_PRF_3);

	GLuint query;
	unsigned int result;


	DataContainer *Src1_tex = GPUCV_GET_TEX(src1);
	DataContainer *Src2_tex = GPUCV_GET_TEX(src2);

	//Get temp texture for result
	DataContainer *temp_tex = NULL;
#if _GPUCV_USE_TEMP_FBO_TEX
	temp_tex = RenderBufferManager()->SelectInternalTexture(TextureRenderBuffer::RENDER_FP16_RGBA);
#else
	temp_tex = new DataContainer();
	DataDsc_GLTex * temp_GLtex = temp_tex->GetDataDsc<DataDsc_GLTex>();
	temp_GLtex->_SetSize(GetWidth(src1), GetHeight(src1));
	temp_GLtex->SetFormat(GL_RGBA, GL_UNSIGNED_SHORT);
#endif
	temp_tex->SetLabel("cvgIsDiff tempTex");

	// Setting occlusion query
	glEnable(GL_CULL_FACE);
	glGenQueriesARB(1, &query);

	TemplateOperator("cvgIsDiff", "FShaders/diff_tf.frag", "",
		Src1_tex, Src2_tex, NULL,
		temp_tex);
	glEndQueryARB(GL_SAMPLES_PASSED_ARB);

	// Getting result : if == 0, images are identical
	glGetQueryObjectuivARB(query, GL_QUERY_RESULT_ARB, &result);
	glDisable (GL_CULL_FACE);
	glDeleteQueriesARB(1, &query);

#if _GPUCV_USE_TEMP_FBO_TEX

#else
	delete temp_tex;
#endif
	return ( 100.*result /(float)(GetWidth(src1)*GetHeight(src1)));

	GPUCV_STOP_OP(return 0.;,
		src1, src2, NULL, NULL
		);
	return 0;
}
//=============================================================
#if _GPUCV_DEVELOP_BETA
#define _DEBUG_CONNECTEDCOMP
void cvgConnectedComp(  CvArr* src, CvArr* dst)
{
	CvArr* temp, *temp1, *temp2, *temp_swap;
	string chosen_filter;

	// Testing Geforce6
	if (!GetHardProfile()->IsCompatible(GenericGPU::HRD_PRF_3))
	{
		GPUCV_WARNING("This operator needs GeForce6\n");
		exit(EXIT_FAILURE);
	}

#ifdef _DEBUG_CONNECTEDCOMP
	//Creating windows
	cvNamedWindow("Source Image",1);
	cvNamedWindow("Temp1",1);
	cvNamedWindow("Temp2",1);
#endif
	SetThread();
	temp = cvgCloneImage(dst);

	// Loading shader
	chosen_filter = GetFilterManager()->GetFilterName("FShaders/connected_comp1.frag");

	// Disabling CPU memory return
	cvgUnsetCpuReturn(dst);
	cvgUnsetCpuReturn(temp);

	// First pass : pixel tagging
	if(GetFilterManager()->GetNbParams(chosen_filter) == 0)
		GetFilterManager()->AddParam(chosen_filter, (float)(src->width));
	else
		GetFilterManager()->SetParamI(chosen_filter, 0, (float)(src->width));
	GetFilterManager()->Apply(chosen_filter,src,temp,dst->width,dst->height);

	// Second pass : aggregation
	chosen_filter = GetFilterManager()->GetFilterName("FShaders/connected_comp2.frag");

	// Looping until the two images are identical
	temp1 = temp;
	temp2 = dst;

	int PassNbr = 0;
	int LocalPassNbr = 5;
	do
	{
		LocalPassNbr = 5;
		//	do
		//	{
		GetFilterManager()->Apply(chosen_filter,temp1,temp2,dst->width,dst->height);
		temp_swap = temp1;
		temp1 = temp2;
		temp2 = temp_swap;
		PassNbr++;

		//	}while(LocalPassNbr-->0);
#ifdef _DEBUG_CONNECTEDCOMP
		cvShowImage("Temp1",temp1);
		cvShowImage("Temp2",temp2);
		GPUCV_DEBUG("Log: cvgConnectedComp update temp1 & temp2");
		cvWaitKey();
#endif
	}
	while(cvgIsDiff(temp1,temp2));

	char Msg[64];
	sprintf(Msg,"\nConnected Component pass number %d\n",PassNbr);
	GPUCV_DEBUG(Msg);

	// Returning to CPU memory for display
	cvgSetCpuReturn(dst);
	cvgSetCpuReturn(temp);
	UnsetThread();
	cvgReleaseImage(&temp);
#ifdef _DEBUG_CONNECTEDCOMP
	//Creating windows
	cvDestroyWindow("Source Image");
	cvDestroyWindow("Temp1");
	cvDestroyWindow("Temp2");
#endif
}


/*
void cvgShowTextureImage( char* name, GLuint width, GLuint height, GLuint Format, GLuint PixelType)
{
//DebugImgData = new unsigned char[3*width*height];
int FormatCV = cvgConvertGLTexFormatToCV(Format);
int TypeCV   = cvgConvertGLPixTypeToCV(PixelType);

CvArr* test_debug = cvCreateImage(cvSize(width, height),TypeCV,FormatCV);
glReadPixels(0,0,width, height,Format,PixelType,test_debug->imageData);

//	memcpy(test_debug->imageData,DebugImgData,3*width*height*sizeof(unsigned char));
cvNamedWindow(name,1);
cvShowImage(name,test_debug);
cvWaitKey();
cvReleaseImage(&test_debug);
cvDestroyWindow(name);
//delete DebugImgData;
}
*/
//#define _DEBUG_CONNECTEDCOMP
void cvgConnectedComp2(  CvArr* src, CvArr* dst)
{
	//CvArr *temp1, *temp2;//, *temp,*temp_swap;
	string chosen_filter;

	// Testing Geforce6
	if (!GetHardProfile()->IsCompatible(GenericGPU::HRD_PRF_3))
	{
		GPUCV_WARNING("This operator needs GeForce6\n");
		exit(EXIT_FAILURE);
	}

	//
	//	GLuint temptex_1 = RenderBufferManager()->TexFP32;
	//	GLuint temptex_2 = RenderBufferManager()->TexFP32bis;
	DataContainer* temptexOrig = new DataContainer(GL_RGBA_FLOAT32_ATI,
		src->width,
		src->height,
		GL_LUMINANCE,
		GL_UNSIGNED_BYTE,
		src->imageData);
	DataContainer* temptex_1 = new DataContainer(GL_RGBA_FLOAT32_ATI,
		src->width,
		src->height,
		GL_RGBA,
		GL_UNSIGNED_BYTE,
		0);
	DataContainer* temptex_2 = new DataContainer(GL_RGBA_FLOAT32_ATI,
		src->width,
		src->height,
		GL_RGBA,
		GL_UNSIGNED_BYTE,
		0);

	//cvgShowFrameBufferImage("FrameBuffer 1",src->width,src->height,cvgConvertCVTexFormatToGL(1), cvgConvertCVPixTypeToGL(IPL_DEPTH_8U));
	//cvgShowFrameBufferImage("FrameBuffer 1",src->width,src->height,cvgConvertCVTexFormatToGL(2), cvgConvertCVPixTypeToGL(IPL_DEPTH_8U));
	//cvgShowFrameBufferImage("FrameBuffer 1",src->width,src->height,cvgConvertCVTexFormatToGL(3), cvgConvertCVPixTypeToGL(IPL_DEPTH_8U));


	// Loading shader, labeling
	float Params[2] = {(float)(src->width),(float)(src->height)};

	chosen_filter = GetFilterManager()->GetFilterName("FShaders/connected_comp1y.frag");
	GetFilterManager()->SetParams(chosen_filter, Params, 2);
	RenderBufferManager()->Force(temptex_1, dst->width,dst->height);
	GetFilterManager()->Apply(chosen_filter,temptexOrig,temptex_1,dst->width,dst->height);
	RenderBufferManager()->GetResult();
	//GetTextureManager()->LoadImageIntoFrameBuffer(src,temptex_1);
	// cvgShowFrameBufferImage("Labeling",src->width,src->height,cvgConvertCVTexFormatToGL(1), cvgConvertCVPixTypeToGL(IPL_DEPTH_8U));
	//  cvgShowFrameBufferImage("Labeling",src->width,src->height,cvgConvertCVTexFormatToGL(2), cvgConvertCVPixTypeToGL(IPL_DEPTH_8U));
	cvgShowFrameBufferImage("Labeling",src->width,src->height,cvgConvertCVTexFormatToGL(3, "BGR"), cvgConvertCVPixTypeToGL(IPL_DEPTH_8U));
	cvgShowFrameBufferImage("Labeling",src->width,src->height,cvgConvertCVTexFormatToGL(4, "BGRA"), cvgConvertCVPixTypeToGL(IPL_DEPTH_8U));
	RenderBufferManager()->UnForce();
	//--------------------------

	// Second pass : aggregation looking for max
	chosen_filter = GetFilterManager()->GetFilterName("FShaders/connected_comp2ybismin.frag");
	GetFilterManager()->SetParams(chosen_filter, Params, 2);

	int PassNbr = 0;
	GPUCV_TEXT_TYPE TemtexSwap;
	do
	{
		RenderBufferManager()->Force(temptex_2, dst->width,dst->height);
		GetFilterManager()->Apply(chosen_filter,temptex_1,temptex_2,dst->width,dst->height);
		RenderBufferManager()->GetResult();

		TemtexSwap = temptex_1;
		temptex_1 = temptex_2;
		temptex_2 = TemtexSwap;
		PassNbr++;
#ifdef _DEBUG_CONNECTEDCOMP
		GPUCV_DEBUG("Passnbr :" << PassNbr);
		//	GetTextureManager()->LoadImageIntoFrameBuffer(src,temptex_2);
		//    cvgShowFrameBufferImage("Connected Comp",src->width,src->height,cvgConvertCVTexFormatToGL(1), cvgConvertCVPixTypeToGL(IPL_DEPTH_8U));
		//	cvgShowFrameBufferImage("Connected Comp",src->width,src->height,cvgConvertCVTexFormatToGL(2), cvgConvertCVPixTypeToGL(IPL_DEPTH_8U));
		cvgShowFrameBufferImage("Connected Comp",src->width,src->height,cvgConvertCVTexFormatToGL(3, "BGR"), cvgConvertCVPixTypeToGL(IPL_DEPTH_8U));
		//	cvgShowFrameBufferImage("Connected Comp",src->width,src->height,cvgConvertCVTexFormatToGL(4, "BGRA"), cvgConvertCVPixTypeToGL(IPL_DEPTH_8U));
#endif
		RenderBufferManager()->UnForce();
	}
	while( (PassNbr<50));//cvgIsDiff(temp1,temp2));//&&




#if 0
	GLuint temptex_1 = TexCreate(GL_LUMINANCE,
		src->width,
		src->height,
		GL_LUMINANCE,
		GL_UNSIGNED_BYTE,
		src->imageData,0);
	GLuint temptex_2 = TexCreate(GL_LUMINANCE,
		src->width,
		src->height,
		GL_LUMINANCE,
		GL_UNSIGNED_BYTE,
		src->imageData,0);

	/*	GLuint temptex_1 = RenderBufferManager()->TexFP32;
	GLuint temptex_2 = RenderBufferManager()->TexFP32bis;
	*/
	GetTextureManager()->LoadImageIntoFrameBuffer(src,temptex_1);


	SetThread();
	temp1 = cvgCreateImage(cvGetSize(dst),8, 1);//cvgCloneImage(dst);
	//temp2 = cvCreateImage(cvGetSize(dst),8, 3);
	//	temp2 = cvgCloneImage(temp);//cvgCloneImage(dst);

	// Loading shader
	chosen_filter = GetFilterManager()->GetFilterName("FShaders/connected_comp1y.frag");
	//GetFilterManager()->Apply(chosen_filter,src,temp1,dst->width,dst->height);
	//	RenderBufferManager()->SetContext(temptex_2, dst->width,dst->height);
	GetFilterManager()->Apply(chosen_filter,temptex_1,temptex_2,dst->width,dst->height);
	//    RenderBufferManager()->GetResult();

#ifdef _DEBUG_CONNECTEDCOMP
	//Creating windows
	cvNamedWindow("Source Image",1);
	cvShowImage("Source Image",src);
	//	cvvWaitKey(0);
	cvNamedWindow("Temp1",1);
	//	cvNamedWindow("Temp2",1);
	//    GetTextureManager()->LoadTextureIntoFBO(src,temptex_1);
	GetTextureManager()->GrabGLBuffer(temp1, GL_RGBA, GL_RGBA_FLOAT32_ATI);
	//	cvgSetCpuReturn(temp1);
	cvgShowImage("Temp1",temp1);
	cvgShowFrameBufferImage("temptex_1", src->width, src->height, GL_RGBA, GL_RGBA_FLOAT32_ATI);
#endif

	//	RenderBufferManager()->UnSetContext();
	//	GetFilterManager()->Apply(chosen_filter,src,temp,dst->width,dst->height);
	//	RenderBufferManager()->GetResult();
	//	RenderBufferManager()->UnForce();
	//	cvgSetCpuReturn(temp);
	//	cvShowImage("Source Image",temp);
	//cvvWaitKey(0);
	//==============================================

	// Second pass : aggregation looking for max
	chosen_filter = GetFilterManager()->GetFilterName("FShaders/connected_comp2ybismin.frag");

	if(GetFilterManager()->GetNbParams(chosen_filter) == 0)
	{
		GetFilterManager()->AddParam(chosen_filter, (float)(src->width));
		GetFilterManager()->AddParam(chosen_filter, (float)(src->height));
	}
	else
	{
		GetFilterManager()->SetParamI(chosen_filter, 0, (float)(src->width));
		GetFilterManager()->SetParamI(chosen_filter, 0, (float)(src->height));
	}

	// Looping until the two images are identical
	//	temp1 = temp;
	//temp2 = dst;

	//UnsetCpuReturn(temp1);
	//UnsetCpuReturn(temp2);
	int PassNbr = 0;
	/*

	do
	{
	cvgUnsetCpuReturn(temp1);
	cvgUnsetCpuReturn(temp2);
	GetFilterManager()->Apply(chosen_filter,temptex_1,temptex_2,dst->width,dst->height);
	temp_swap = temp1;
	temp1 = temp2;
	temp2 = temp_swap;
	PassNbr++;
	#ifdef _DEBUG_CONNECTEDCOMP
	//   GetTextureManager()->GrabGLBuffer(dst, GL_RGBA);
	cvgSetCpuReturn(temp1);
	cvgSetCpuReturn(temp2);
	cvShowImage("Temp1",temp1);
	cvShowImage("Temp2",temp2);
	GPUCV_ERROR(("\nLog: cvgConnectedComp update temp1 & temp2");
	printf("\nPassnbr %d", PassNbr);
	cvWaitKey(0);
	//	cvgCopy(dst,temp);
	#endif
	}
	while(cvgIsDiff(temp1,temp2));//&& (PassNbr<5));

	*/
	// third pass : we are looking for pixels that have changed
	/*	int reduceFactor = 4;
	CvSize NewSize;
	NewSize.width = temp1->width;
	NewSize.height = temp1->height;

	CvArr *MipMap1 = cvgCreateImage(NewSize,temp1->depth,temp1->nChannels);
	chosen_filter = GetFilterManager()->GetFilterName("FShaders/connected_comp3y.frag");

	if(GetFilterManager()->GetNbParams(chosen_filter) == 0)
	{
	GetFilterManager()->AddParam(chosen_filter, (float)(src->width));
	GetFilterManager()->AddParam(chosen_filter, (float)(src->height));
	GetFilterManager()->AddParam(chosen_filter, (float)(reduceFactor));
	}
	else
	{
	GetFilterManager()->SetParamI(chosen_filter, 0, (float)(src->width));
	GetFilterManager()->SetParamI(chosen_filter, 0, (float)(src->height));
	GetFilterManager()->SetParamI(chosen_filter, 0, (float)(reduceFactor));
	}


	GetFilterManager()->Apply(chosen_filter,temp1,MipMap1,MipMap1->width,MipMap1->height);

	PassNbr++;
	#ifdef _DEBUG_CONNECTEDCOMP
	cvgSetCpuReturn(temp1);
	cvgSetCpuReturn(MipMap1);
	cvShowImage("Temp1",temp1);
	cvShowImage("Temp2",MipMap1);
	GPUCV_ERROR(("\nLog: Reducing Image Size");
	printf("\nPassnbr %d", PassNbr);
	cvWaitKey(0);
	#endif
	*/
	//get MipMapping...?

	/*
	glEnable(GL_DEPTH_TEST);
	glShadeModel(GL_FLAT);//alphaChannel
	//glTexParameteri(GetHardProfile()->GetTextType(), GL_TEXTURE_WRAP_S, GL_REPEAT);
	//glTexParameteri(GetHardProfile()->GetTextType(), GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GetHardProfile()->GetTextType(), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GetHardProfile()->GetTextType(), GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);

	//createTExture

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glEnable(GetHardProfile()->GetTextType());
	*/


	char Msg[64];
	sprintf(Msg,"\nConnected Component pass number %d\n",PassNbr);
	GPUCV_ERROR((Msg);

	// Returning to CPU memory for display
	//	cvgSetCpuReturn(dst);
	//	cvgSetCpuReturn(temp);
	UnsetThread();
	//	cvgReleaseImage(&temp);
#endif //0


#ifdef _DEBUG_CONNECTEDCOMP
	//Creating windows
	//	cvDestroyWindow("Source Image");
	//	cvDestroyWindow("Temp1");
	//	cvDestroyWindow("Temp2");
#endif
}
//=============================================================
void cvgConnectedComp3(  CvArr* src, CvArr* dst)
{
	CvArr* temp, *temp1, *temp2, *temp_swap;
	string chosen_filter;

	// Testing Geforce6
	if (!GetHardProfile()->IsCompatible(GenericGPU::HRD_PRF_3))
	{
		GPUCV_ERROR(("This operator needs GeForce6\n");
		exit(EXIT_FAILURE);
	}

	/*	GLuint temptex_1 = TexCreate(GL_RGBA,
	src->width,
	src->height,
	GL_LUMINANCE,
	GL_UNSIGNED_BYTE,
	src->imageData,0);
	GLuint temptex_2 = TexCreate(GL_RGBA,
	src->width,
	src->height,
	GL_LUMINANCE,
	GL_UNSIGNED_BYTE,
	src->imageData,0);

	GetTextureManager()->LoadTextureIntoFBO(src,temptex_1);
	*/
	SetThread();
	temp1 = cvCreateImage(cvGetSize(dst),8, 3);//cvgCloneImage(dst);
	temp2 = cvCreateImage(cvGetSize(dst),8, 3);
	//	temp2 = cvgCloneImage(temp);//cvgCloneImage(dst);

	// Loading shader
	chosen_filter = GetFilterManager()->GetFilterName("FShaders/connected_comp1y.frag");
	GetFilterManager()->Apply(chosen_filter,src,temp1,dst->width,dst->height);

#ifdef _DEBUG_CONNECTEDCOMP

	//Creating windows
	cvNamedWindow("Source Image",1);
	//	cvShowImage("Source Image",src);
	//	cvvWaitKey(0);
	cvNamedWindow("Temp1",1);
	cvNamedWindow("Temp2",1);

	//	GetTextureManager()->GrabGLBuffer(temp1, GL_RGBA);
	cvgSetCpuReturn(temp1);
	cvShowImage("Temp1",temp1);
	cvvWaitKey(0);
#endif
	//	GetFilterManager()->Apply(chosen_filter,src,temp,dst->width,dst->height);
	//	RenderBufferManager()->GetResult();
	//	RenderBufferManager()->UnForce();
	//	cvgSetCpuReturn(temp);
	//	cvShowImage("Source Image",temp);
	//cvvWaitKey(0);
	//==============================================

	// Second pass : aggregation looking for max
	chosen_filter = GetFilterManager()->GetFilterName("FShaders/connected_comp2ybismin.frag");

	if(GetFilterManager()->GetNbParams(chosen_filter) == 0)
	{
		GetFilterManager()->AddParam(chosen_filter, (float)(src->width));
		GetFilterManager()->AddParam(chosen_filter, (float)(src->height));
	}
	else
	{
		GetFilterManager()->SetParamI(chosen_filter, 0, (float)(src->width));
		GetFilterManager()->SetParamI(chosen_filter, 0, (float)(src->height));
	}

	// Looping until the two images are identical
	//	temp1 = temp;
	//temp2 = dst;

	//UnsetCpuReturn(temp1);
	//UnsetCpuReturn(temp2);

	int PassNbr = 0;
	do
	{
		cvgUnsetCpuReturn(temp1);
		cvgUnsetCpuReturn(temp2);
		GetFilterManager()->Apply(chosen_filter,temp1,temp2,dst->width,dst->height);
		temp_swap = temp1;
		temp1 = temp2;
		temp2 = temp_swap;
		PassNbr++;
#ifdef _DEBUG_CONNECTEDCOMP
		//   GetTextureManager()->GrabGLBuffer(dst, GL_RGBA);
		cvgSetCpuReturn(temp1);
		cvgSetCpuReturn(temp2);
		cvShowImage("Temp1",temp1);
		cvShowImage("Temp2",temp2);
		GPUCV_ERROR(("\nLog: cvgConnectedComp update temp1 & temp2");
		printf("\nPassnbr %d", PassNbr);
		cvWaitKey(0);
		//	cvgCopy(dst,temp);
#endif
	}
	while(cvgIsDiff(temp1,temp2));//&& (PassNbr<5));


	// third pass : we are looking for pixels that have changed
	/*	int reduceFactor = 4;
	CvSize NewSize;
	NewSize.width = temp1->width;
	NewSize.height = temp1->height;

	CvArr *MipMap1 = cvgCreateImage(NewSize,temp1->depth,temp1->nChannels);
	chosen_filter = GetFilterManager()->GetFilterName("FShaders/connected_comp3y.frag");

	if(GetFilterManager()->GetNbParams(chosen_filter) == 0)
	{
	GetFilterManager()->AddParam(chosen_filter, (float)(src->width));
	GetFilterManager()->AddParam(chosen_filter, (float)(src->height));
	GetFilterManager()->AddParam(chosen_filter, (float)(reduceFactor));
	}
	else
	{
	GetFilterManager()->SetParamI(chosen_filter, 0, (float)(src->width));
	GetFilterManager()->SetParamI(chosen_filter, 0, (float)(src->height));
	GetFilterManager()->SetParamI(chosen_filter, 0, (float)(reduceFactor));
	}


	GetFilterManager()->Apply(chosen_filter,temp1,MipMap1,MipMap1->width,MipMap1->height);

	PassNbr++;
	#ifdef _DEBUG_CONNECTEDCOMP
	cvgSetCpuReturn(temp1);
	cvgSetCpuReturn(MipMap1);
	cvShowImage("Temp1",temp1);
	cvShowImage("Temp2",MipMap1);
	GPUCV_ERROR(("\nLog: Reducing Image Size");
	printf("\nPassnbr %d", PassNbr);
	cvWaitKey(0);
	#endif
	*/
	//get MipMapping...?

	/*
	glEnable(GL_DEPTH_TEST);
	glShadeModel(GL_FLAT);//alphaChannel
	//glTexParameteri(GetHardProfile()->GetTextType(), GL_TEXTURE_WRAP_S, GL_REPEAT);
	//glTexParameteri(GetHardProfile()->GetTextType(), GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GetHardProfile()->GetTextType(), GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GetHardProfile()->GetTextType(), GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);

	//createTExture

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glEnable(GetHardProfile()->GetTextType());
	*/


	char Msg[64];
	sprintf(Msg,"\nConnected Component pass number %d\n",PassNbr);
	GPUCV_ERROR((Msg);

	// Returning to CPU memory for display
	cvgSetCpuReturn(dst);
	cvgSetCpuReturn(temp);
	UnsetThread();
	cvgReleaseImage(&temp);
#ifdef _DEBUG_CONNECTEDCOMP
	//Creating windows
	cvDestroyWindow("Source Image");
	cvDestroyWindow("Temp1");
	cvDestroyWindow("Temp2");
#endif
}
#endif
//=============================================================

#if 0
void cvgCalcHistVBO( IplImage** img, CvHistogram* hist, int doNotClear=0,  CvArr* mask=0 )
{
	GPUCV_START_OP(cvgCalcHistVBO(img, hist, doNotClear, mask),
		"cvCalcHist",
		*img,
		GenericGPU::HRD_PRF_3);

	DataContainer * HistoDest = new DataContainer();








	GPUCV_STOP_OP(
		cvgCalcHistVBO(img, hist, doNotClear, mask),
		*img, NULL, NULL, NULL
		);
}
#endif
#if _GPUCV_DEVELOP_BETA
/*!
*	\brief GPU histogram computation by using VertexShader
*	\param src -> source image
*	\param hist -> [MORE_HERE]
*	\param liste2 -> [MORE_HERE]
*	\return none
*	\author Jean-Philippe Farrugia
*/
// Passing display list in function FOR DEBUG PURPOSE ONLY
#if _GPUCV_DEBUG_MODE
//	#define _DEBUG_HISTO
#endif
#define _DEBUG_FP16

//#define _GPUCV_PROFILE_HISTO
void cvgCalcHist2(CvArr** src,CvHistogram* hist,GLuint liste2, GLuint RGBA_Texture/*=0*/)
{
	unsigned char* DebugImgData=NULL ;
	CvArr* test_debug=NULL;

	float params[2];
	string chosen_filter;

	// Testing Geforce6
	if (!GetHardProfile()->IsCompatible(GenericGPU::HRD_PRF_3))
	{
		GPUCV_ERROR(("This operator needs GeForce6, switching to OpenCV...\n");
		cvCalcHist(src,hist);
	}
	SetThread();

	char* image_data = 0;

	//Init static textures
	static GLuint StaticFP32_1 = TexCreate(GL_RGBA_FLOAT32_ATI,
		src[0]->width,
		src[0]->height,
		GL_LUMINANCE,
		GL_UNSIGNED_BYTE,
		0, RenderBufferManager()->TexFP32);
	static GLuint StaticFP16 = TexCreate(GL_RGBA_FLOAT16_ATI,
		512,
		512,
		GL_LUMINANCE,
		GL_UNSIGNED_BYTE,
		0, RenderBufferManager()->TexFP16);
	static GLuint StaticFP32_2 = TexCreate(GL_RGBA_FLOAT32_ATI,
		512,
		512,
		GL_LUMINANCE,
		GL_UNSIGNED_BYTE,
		0, RenderBufferManager()->TexFP32);




	// Testing if src[0] is cpureturn or not
	if(GetTextureManager()->IsCpuReturn(src[0]))
		image_data = src[0]->imageData;

	GLuint temptex_1 =StaticFP32_1;
	GLuint temptex_2 =StaticFP16;
	GLuint temptex_3 =StaticFP32_2;

	//	RenderBufferManager()->CopyToFP32Buffer(src[0]->imageData, src[0]->width, src[0]->height, GL_LUMINANCE, GL_UNSIGNED_BYTE);
	// First texture containing src image
	// First Texture : fp32 for reading from vertex processor
#ifdef _DEBUG_FP16
	if (RGBA_Texture) temptex_1 = RGBA_Texture;
	else
	{
		printf("\nRGBA_Texture is NULL");
		if(image_data == 0)
		{

			/*	temptex_1 = RenderBufferManager()->TexFP32;
			RenderBufferManager()->Force(temptex_1, src[0]->width, src[0]->height);
			GetTextureManager()->LoadImageIntoFrameBuffer(src[0],temptex_1);
			//cvgShowFrameBufferImage("FrameBuffer",src[0]->width,src[0]->height,cvgConvertCVTexFormatToGL(src[0]->nChannels), cvgConvertCVPixTypeToGL(src[0]->depth));
			cvgShowFrameBufferImage("FrameBuffer",src[0]->width,src[0]->height,GL_LUMINANCE, GL_FLOAT);
			RenderBufferManager()->UnForce();
			*/
			/*	temptex_1 = TexCreate(GL_RGBA_FLOAT32_ATI,
			src[0]->width,
			src[0]->height,
			GL_LUMINANCE,
			GL_UNSIGNED_BYTE,
			image_data, RenderBufferManager()->TexFP32);
			*/
			RenderBufferManager()->Force(temptex_1, src[0]->width, src[0]->height);
			// gl initialisation
			InitGLView(src[0]->width, src[0]->height);

			// Displaying quad
			glActiveTextureARB(GL_TEXTURE0);
			glBindTexture(GetHardProfile()->GetTextType(), GetTextureManager()->GetGpuImage(src[0]));
			//drawQuad(src[0]->width, src[0]->height);//05/10/05 Yann.A, is it also depending on the GetHardProfile()->GetTextType().???yes->pass image Width and Height...
			drawQuad();
#ifdef _DEBUG_HISTO
			cvgShowFrameBufferImage("Hist src Window 1", src[0]->width, src[0]->height,GL_RGBA, GL_FLOAT);
#endif
			RenderBufferManager()->UnForce();
		}
		else
		{
			temptex_1 = TexCreate(GL_RGBA_FLOAT32_ATI,
				src[0]->width,
				src[0]->height,
				GL_LUMINANCE,
				GL_UNSIGNED_BYTE,
				image_data, temptex_1);
		}

	}
#else

	if(image_data != 0)
	{
		// First texture containing src image
		// First Texture : fp32 for reading from vertex processor
		temptex_1 = TexCreate(GL_RGBA_FLOAT32_ATI,
			src[0]->width,
			src[0]->height,
			GL_LUMINANCE,
			GL_UNSIGNED_BYTE,
			image_data);
	}
	else
	{
		// Copying src0 text into tempTex1

		temptex_1 = TexCreate(GL_RGBA_FLOAT32_ATI,
			src[0]->width,
			src[0]->height,
			GL_LUMINANCE,
			GL_UNSIGNED_BYTE,
			image_data);
		RenderBufferManager()->Force(temptex_1, src[0]->width, src[0]->height);
		// gl initialisation
		InitGLView(src[0]->width, src[0]->height);

		// Displaying quad
		glActiveTextureARB(GL_TEXTURE0);
		glBindTexture(GetHardProfile()->GetTextType(), GetTextureManager()->GetGpuImage(src[0]));
		//drawQuad(src[0]->width, src[0]->height);//05/10/05 Yann.A, is it also depending on the GetHardProfile()->GetTextType().???yes->pass image Width and Height...
		drawQuad();
		RenderBufferManager()->UnForce();


		// DEBUG
		//-------------------------
#ifdef _DEBUG_HISTO
		cvgShowFrameBufferImage("Hist src Window 1", src[0]->width, src[0]->height,GL_RGBA, GL_FLOAT);
#endif
	}
#endif //debugFP16



#ifdef _DEBUG_FP16
	//	temptex_2 = StaticFP16;//RenderBufferManager()->TexFP16;
	//	glBindTexture(GetHardProfile()->GetTextType(), temptex_2);
	//	temptex_3 = StaticFP32_2;//RenderBufferManager()->TexFP32bis;

	// Second Texture : fp16 for blending
	/*	temptex_2 = TexCreate(GL_RGBA_FLOAT16_ATI,
	512,
	512,
	GL_RGB, //GL_LUMINANCE,
	GL_FLOAT,//GL_UNSIGNED_BYTE,
	0, RenderBufferManager()->TexFP16);
	*/
	// Third Texture : fp32 for precision
	/*	temptex_3 = TexCreate(GL_RGBA_FLOAT32_ATI,
	512,
	512,
	GL_LUMINANCE,
	GL_UNSIGNED_BYTE,
	0, RenderBufferManager()->TexFP32bis);
	*/
#else
	// Second Texture : fp16 for blending
	temptex_2 = TexCreate(GL_RGBA_FLOAT16_ATI,
		512,
		512,
		GL_LUMINANCE,
		GL_UNSIGNED_BYTE,
		0);
	// Third Texture : fp32 for precision

	temptex_3 = TexCreate(GL_RGBA_FLOAT32_ATI,
		512,
		512,
		GL_LUMINANCE,
		GL_UNSIGNED_BYTE,
		0);
#endif

	glFlush();
	glFinish();


	// First pass : vertex displacement
	// Loading shader
	chosen_filter = GetFilterManager()->GetFilterName("VShaders/histogram_vs.vert","FShaders/histogram_vs.frag");

	// Force context to keep temptex_1, for glReadPixels
	RenderBufferManager()->Force(temptex_2, 512,512);

	// Parameters
	//float params[2] = {threshold/255., maxValue/255.};
	params[0] = params[1] = 512.;
	GetFilterManager()->SetParams(chosen_filter, params, 2);
	GetFilterManager()->SetDisplayList(chosen_filter, liste2);

	// Enabling blending
	glBlendFunc(GL_ONE, GL_ONE);
	glEnable(GL_BLEND);
	// Computation
	GetFilterManager()->Apply(chosen_filter,temptex_1,temptex_2,512,512);
	glDisable(GL_BLEND);

#ifdef _DEBUG_HISTO
	cvgShowFrameBufferImage("Hist dst Window",(GLuint)params[0],(GLuint)params[1],GL_RGB,GL_UNSIGNED_BYTE);
#endif
	//--------------------------
	// Eventuellement

	RenderBufferManager()->UnForce();
	glFlush();
	glFinish();


#ifdef _DEBUG_FP16
	//temptex_3 = RenderBufferManager()->TexFP32;
	/*TexCreate(GL_RGBA_FLOAT32_ATI,
	512,
	512,
	GL_LUMINANCE,
	GL_UNSIGNED_BYTE,
	0, RenderBufferManager()->TexFP32);
	*/
#endif
	// Force context to keep temp_tex1, for glReadPixels
	//	RenderBufferManager()->Force(temptex_3, src[0]->width, src[0]->height);
	params[0] = params[1] = 512.;
	RenderBufferManager()->Force(temptex_3,	(GLuint)params[0], (GLuint)params[1]);



	// Second pass : aggregation;
	chosen_filter = GetFilterManager()->GetFilterName("FShaders/histogram_vs2.frag");
	GetFilterManager()->SetParams(chosen_filter, params, 2);
	GetFilterManager()->Apply(chosen_filter,temptex_2,temptex_3,(GLuint)params[0], (GLuint)params[1]);
	//	GetFilterManager()->Apply(chosen_filter,temptex_2,temptex_3,src[0]->width,src[0]->height);

	glFlush();
	glFinish();

	// Reading histogram data
	float histodata[256];
	_BENCH_GL("glReadPixels",glReadPixels(256,254,256,1,GL_RED,GL_FLOAT,histodata),"",256,1);

	// Putting histogram data in histogram structure
	for (int i=0; i<256; i++)
		((CvMatND*)(&hist->mat))->data.fl[i] = histodata[i];//*16 is for test only..Yann.A..

	// Eventuellement
	RenderBufferManager()->UnForce();
#ifndef _DEBUG_FP16
	GetTextureManager()->AddFreeTexture(temptex_1);
	GetTextureManager()->AddFreeTexture(temptex_2);
	GetTextureManager()->AddFreeTexture(temptex_3);
#endif

	// End
	UnsetThread();

}
#else
//=============================================================
void CalcHisDrawFct(TextureGrp*_inputTexGrp, TextureGrp*_outputTexGrp, GLuint _width, GLuint _height)
{
#if 0
	SG_Assert(_inputTexGrp, "Empty");
	_GPUCV_GL_ERROR_TEST();
	//SG_Assert(_outputTexArr, "Empty");
	DataContainer *Src = _inputTexGrp->operator [](0);

	//buffer size
	GLsizeiptr VertexNbr = Src->_GetWidth() * Src->_GetHeight()/4;
	GLuint PositionSize	= VertexNbr* Src->_GetNChannels() * sizeof(GLubyte);
	GLuint ColorSize	= VertexNbr* Src->_GetNChannels() * sizeof(GLubyte);
	GLvoid * data = Src->_GetPixelsData();

	//create and bind vbo
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);

	GLBuffer VertexSrcBuffer(GLBuffer::ARRAY_BUFFER, GLBuffer::STATIC_DRAW, PositionSize, data);
	VertexSrcBuffer.SetPointer(GLBuffer::VERTEX_POINTER, 2, GL_SHORT, 0,  0);

	GLBuffer ColorSrcBuffer(GLBuffer::ARRAY_BUFFER, GLBuffer::STATIC_DRAW, ColorSize, data);
	ColorSrcBuffer.SetPointer(GLBuffer::COLOR_POINTER, 3, GL_SHORT, 0,  0);
	glClearColor(0, 255, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	glTranslatef(0., 0., -0.5);
	glScalef(1/256., 1/256., 1/256.);
	glScalef(1/200., 1/200., 1/200.);
	glColor4f(255., 255., 255., 255.);

	VertexSrcBuffer.Bind();
	VertexSrcBuffer.DrawArrays(GL_POINTS, 0, PositionSize/sizeof(GLubyte)/100);
	VertexSrcBuffer.UnBind();
	//TempDest->UnsetRenderToTexture();

	VertexSrcBuffer.Disable();
	ColorSrcBuffer.Disable();
	_GPUCV_GL_ERROR_TEST();
#endif
}
//=============================================================


#endif

//=============================================================
#if _GPUCV_DEVELOP_BETA
/*!
*	\brief GPU histogram computation by using OcclusionQuery extension
*	\param img -> images (only the first one is used yet)
*	\param hist -> OpenCV histogram structure used to store data
*	\param doNotClear -> donotclear flag
*	\param mask -> mask image (not avaible yet)
*	\return none
*	\author Jean-Philippe Farrugia
*/
void cvgCalcHist( CvArr** img, CvHistogram* hist, int doNotClear,  CvArr* mask)
{

	if (!GetHardProfile()->IsCompatible(GenericGPU::HRD_PRF_3))
	{
		cvCalcHist(img, hist, doNotClear, mask);
		return;
	}

	SetThread();
	if (!GLEW_ARB_occlusion_query) GPUCV_WARNING("Warning : your card doesn't support occlusion query GL extension, openCV operator will be used.\n");
	if (CV_IS_SPARSE_HIST( hist ) || !GLEW_ARB_occlusion_query) cvCalcHist(img, hist, doNotClear, mask);
	else
	{
		// flag if range is non uniform
		bool dynamic_range = !(CV_IS_UNIFORM_HIST( hist ));

		// number of buckets
		int dim = ((CvMatND*)(&hist->mat))->dim[0].size;

		// Storage for occlusion queries
		GLuint    nb_pixels_query;

		// Allocating Histogram
		GLuint* histo_data = new GLuint[dim];

		// Occlusion query setting
		glGenQueriesARB(1, &nb_pixels_query);

		// result texture wich will not be used but is necessary to build a correct FBO
		GLuint textmp = TexCreate(GL_RGB, img[0]->width, img[0]->height, GL_LUMINANCE, GL_UNSIGNED_BYTE, 0 );

		// source texture
		GLuint src  = GetTextureManager()->GetGpuImage(img[0]);

		// binding and forcing use of FBO or PBuffer
		// we need to force it because we don't want to desactivate the rendering context
		// between evry filter pass and between shader application and occlusion query count
		//if (GetHardProfile()->IsFBOCompatible()) FBOManager()->ForceRender(img[0]->width, img[0]->height);

		RenderBufferManager()->Force(textmp, img[0]->width, img[0]->height);
		// range of values
		float range0=0;
		float range1=0;
		float min = 0, step = 0;

		string chosen_filter;
		// loading shader, 2 differents
		if (dynamic_range)
			chosen_filter = GetFilterManager()->GetFilterName("FShaders/histogram_dr.frag");
		else {
			range0 = hist->thresh[0][0];
			range1 = hist->thresh[0][1];

			if (range1 < range0)
				GPUCV_WARNING("Warning : ranges are not in ascending order.\n");

			min = range0/255.;
			step = ((range1 - range0 +1)/(float)dim)/256.;

			string filename = "FShaders/histogram.frag";

			chosen_filter = GetFilterManager()->GetFilterName(filename);
		}
		// parameters initialisation
		float params[2];
		if (dynamic_range)
		{
			params[0] = params[1] = 0;
			GetFilterManager()->SetParams(chosen_filter, params, 2);
		}
		else
		{
			params[0] = params[1] = (float)(0);
			GetFilterManager()->SetParams(chosen_filter, params, 2);
		}
		//loop on buckets
		for(int i=0;i<dim;i++)
		{

			if (dynamic_range)
			{
				params[0] = range0 = hist->thresh2[0][i];
				params[1] = range1 = hist->thresh2[0][i+1];
				GetFilterManager()->SetParams(chosen_filter, params, 2);
			}
			else
			{
				params[0] = range0 = min + (float)i * step;
				params[1] = range1 = range0+step;
				GetFilterManager()->SetParams(chosen_filter, params, 2);
			}

			// Occlusion query, counting pixels...
			glBeginQueryARB(GL_SAMPLES_PASSED_ARB, nb_pixels_query);
			GetFilterManager()->Apply(chosen_filter, src,textmp,img[0]->width,img[0]->height);

			glFlush();

			// Filling histogram
			glEndQueryARB(GL_SAMPLES_PASSED_ARB);
			glGetQueryObjectuivARB(nb_pixels_query, GL_QUERY_RESULT_ARB, &histo_data[i]);
		}


		// store datas in cvHistogram structure
		if (doNotClear)
		{
			for (int i=0; i<dim; i++)
				((CvMatND*)(&hist->mat))->data.fl[i] += (float)histo_data[i];
		}
		else
		{
			for (int i=0; i<dim; i++)
				((CvMatND*)(&hist->mat))->data.fl[i] = (float)histo_data[i];
		}

		glDeleteQueriesARB(1, &nb_pixels_query);

		delete [] histo_data;


		RenderBufferManager()->UnForce();

		if (GetTextureManager()->IsCpuReturn(img[0])) GetTextureManager()->ForceCpuSetting(img[0]);

		GetTextureManager()->AddFreeTexture(textmp);


		/*
		float max_val_histo, val_histo, val_graph;
		CvArr * hist_img = cvgCreateImage(cvSize(255,255),8,3);
		cvGetMinMaxHistValue(hist,0,&max_val_histo, 0, 0 );
		cvZero(hist_img);

		float w = 256./(float)dim;
		for(int i=0;i<dim;i++)
		{
		val_histo = cvQueryHistValue_1D(hist,i);
		val_graph = cvRound(255-(val_histo/(max_val_histo)*255.0));
		//		cvLine(hist_img,cvPoint(i,255),cvPoint(i,val_graph),CV_RGB(255,0,0));
		cvRectangle(hist_img, cvPoint(i*w,255), cvPoint((i+1)*w,val_graph),CV_RGB(255,0,0));
		}

		// Displaying Images
		cvNamedWindow("MyHistogram",1);
		cvShowImage("MyHistogram", hist_img);

		//cvReleaseImage(&dst);
		//cvgReleaseImage(&red_img);
		cvgReleaseImage(&hist_img);
		*/

	}

	UnsetThread();
}
#endif
//=============================================================

