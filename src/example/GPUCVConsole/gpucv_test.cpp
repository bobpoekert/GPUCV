#include "StdAfx.h"
#include "mainSampleTest.h"
#include "commands.h"
#include <GPUCV/cv_new.h>
#include <GPUCV/cv_new.h>

#if 0
void runRenderBufferTest(IplImage * src1, IplImage * src2)
{
	static int count = -1;
	
	count ++;
	
	IplImage * dest_test	= cvgCreateImage(cvGetSize(src1),8,3);
	//IplImage * sourcetemp	= cvLoadImage("data/pictures/test.jpg",1);
	
	CvgArr * SrcImage		=	GetTextureManager()->Get(src1);
	//CvgArr * SrcImage2 =	GetTextureManager()->Get(src2);
	
	CvgArr * _cvgImageDest = GetTextureManager()->Get(dest_test);
	SrcImage->SetLocation(DataContainer::LOC_GPU);
	
	//GPUCV_DEBUG("\nRendering to DataContainer");
	//GPUCV_DEBUG("\nRendering to IplImage");
	//=================
	
	cvNamedWindow("src",1);\
	cvNamedWindow("renderBuffer CvgArr",1);\
	
#if 1
	GPUCV_DEBUG("\nRendering to CvgArr(USING SETCONTEXT)");
	
	_cvgImageDest->SetLocation(DataContainer::LOC_GPU);
	SrcImage->SetLocation(DataContainer::LOC_GPU);
	glFlush();
	
	_cvgImageDest->SetRenderToTexture();
	InitGLView(dest_test->width, dest_test->height);
	DataContainer::DrawFullQuad(dest_test->width, dest_test->height, SrcImage->GetGpuImage());
	glFlush();
	_cvgImageDest->UnsetRenderToTexture();
	
	cvgShowImage("src", src1);\
	cvgShowImage("renderBuffer CvgArr",  _cvgImageDest->GetIplImage());
	cvWaitKey(0);
	//#else if 0
	GPUCV_DEBUG("\nRendering to CvgArr(USING FORCE)");
	_cvgImageDest->SetLocation(DataContainer::LOC_GPU);
	SrcImage->SetLocation(DataContainer::LOC_GPU);
	glFlush();
	
	_cvgImageDest->ForceRenderToTexture();
	InitGLView(dest_test->width, dest_test->height);
	DataContainer::DrawFullQuad(dest_test->width, dest_test->height, SrcImage->GetGpuImage());
	glFlush();
	_cvgImageDest->UnForceRenderToTexture();
	
	cvgShowImage("src", src1);\
	cvgShowImage("renderBuffer CvgArr",  _cvgImageDest->GetIplImage());
	cvWaitKey(0);
	
	
	//#else //if 1
	GPUCV_DEBUG("\nRendering to CvgArr(USING FBO SETCONTEXT)");
	DataContainer * TextureDest =NULL;
	TextureDest = _cvgImageDest->GetGpuImage();
	
	TextureDest->SetLocation(DataContainer::LOC_GPU);
	SrcImage->SetLocation(DataContainer::LOC_GPU);
	
	//	glFlush();
	RenderBufferManager()->SetContext(TextureDest, TextureDest->_GetWidth(), TextureDest->_GetHeight());//, GL_DEPTH_ATTACHMENT_EXT);
	InitGLView(dest_test->width, dest_test->height);
	/*		switch(count)
	 {
	 case 0:	glClearColor(1., 0., 0., 1);break;
	 case 1:	glClearColor(0., 1., 0., 1);break;
	 case 2:	glClearColor(0., 0., 1., 1);break;
	 case 3:	glClearColor(1., 1., 1., 1);
	 count = -1;
	 break;
	 }
	 *///		glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT|GL_ACCUM_BUFFER_BIT);
	//		glScalef(0.5, 0.5, 0.5);
	
	
	
	
	DataContainer::DrawFullQuad(dest_test->width, dest_test->height, SrcImage->GetGpuImage());
	
	/*
	 glEnable(GL_LIGHTING);
	 glEnable(GL_LIGHT0);
	 GLfloat Color [4] = {1., 1., 1., 1.};
	 glLightfv(GL_LIGHT0, GL_AMBIENT, Color);
	 */
	
	//glTranslatef(0., 0., +1.);
	/*	glBegin(GL_QUADS);
	 glColor4f(1., 1., 0., 1.);
	 glVertex3f(-1, -1, -0.5f);
	 glVertex3f( 1, -1, -0.5f);
	 glVertex3f( 1,  1, -0.5f);
	 glVertex3f(-1,  1, -0.5f);
	 glEnd();
	 */
	//		glFlush();
	
	RenderBufferManager()->UnSetContext();
	//getGLContext()->SetEnv();
	
	TextureDest->SetLocation(DataContainer::LOC_CPU);
	SrcImage->SetLocation(DataContainer::LOC_CPU);
	
	//if (count == 0)
	cvShowImage("src", SrcImage->GetIplImage());
	//else
	//	cvShowImage("src", src2);
	cvShowImage("renderBuffer CvgArr",  _cvgImageDest->GetIplImage());
	cvWaitKey(0);
#endif
	
	cvDestroyWindow("src");\
	cvDestroyWindow("renderBuffer CvgArr");\
	//=================
	//GPUCV_DEBUG("\nRendering to ImageManager");
	
	GetTextureManager()->Delete(dest_test);
}



void runRenderDepth(IplImage * src1, IplImage * src2)
{
#define TestDepthImg 1
#define USE_SwapBUFFER 1
	static int count = -1;
	count ++;
#if TestDepthImg
	IplImage * dest_Depth	= cvgCreateImage(cvGetSize(src1),IPL_DEPTH_32F,1);
#else
	IplImage * dest_Depth	= cvgCreateImage(cvGetSize(src1),IPL_DEPTH_8U,3);
#endif
	IplImage * dest_test	= cvgCreateImage(cvGetSize(src1),IPL_DEPTH_8U,3);
	
	IplImage * sourcetemp	= cvLoadImage("data/pictures/test.jpg",1);
	
	CvgArr * SrcImage		=	GetTextureManager()->Get(sourcetemp);
	//CvgArr * SrcImage2 =	GetTextureManager()->Get(src2);
	
	CvgArr * _cvgImageDest		= GetTextureManager()->Get(dest_test);
	CvgArr * _cvgImageDestDepth	= GetTextureManager()->Get(dest_Depth);
	SrcImage->SetLocation(DataContainer::LOC_GPU);
	
	//=================
	
	cvNamedWindow("src",1);\
	cvNamedWindow("renderBuffer CvgArr",1);\
	
	
	GPUCV_DEBUG("\nRendering to CvgArr(USING FORCE)");
	
	
	GPUCV_DEBUG("\nRendering to CvgArr(USING FBO SETCONTEXT)");
	_cvgImageDest->SetLocation(DataContainer::LOC_GPU, false);//move without transfer
	_cvgImageDestDepth->SetLocation(DataContainer::LOC_GPU, false);//move without transfer
	DataContainer * TextureDest		= _cvgImageDest->GetGpuImage();
	DataContainer * TextureDestDepth	= _cvgImageDestDepth->GetGpuImage();
	
	TextureDest->SetLabel("Control Color Image");
	TextureDestDepth->SetLabel("Depth Image");
	
	
	//TextureDest->SetLocation(DataContainer::LOC_GPU);
	//SrcImage->SetLocation(DataContainer::LOC_GPU);
	
	glFlush();
	
	getGLContext()->PushAttribs();
#if TestDepthImg
	TextureDestDepth->_SetPixelFormat(GL_DEPTH_COMPONENT32F_NV, GL_DEPTH_COMPONENT32F_NV, GL_FLOAT);
#endif
	TextureGrp * TestGroup = new TextureGrp();
	TestGroup->AddTexture(TextureDest);
	
	
	
#if TestDepthImg
	TestGroup->AddTextures(&TextureDestDepth, 1, DataContainer::DEPTH_ATTACHMENT_EXT);
	glDepthFunc(GL_LEQUAL);
	glDepthRange(0., 1.);
	glEnable(GL_DEPTH_TEST);
	glClearDepth(1.);
#endif
	
	switch(count)
	{
		case 0:	glClearColor(1., 0., 0., 1);break;
		case 1:	glClearColor(0., 1., 0., 1);break;
		case 2:	glClearColor(0., 0., 1., 1);break;
		case 3:	glClearColor(1., 1., 1., 1);
			count = -1;
			break;
	}
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	
	
	
#if !USE_SwapBUFFER
	RenderBufferManager()->Force(TestGroup, TextureDest->_GetWidth(), TextureDest->_GetHeight());
#endif
	InitGLViewPerspective(dest_test->width, dest_test->height, 0.1, 10);
	
	//glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
	glScalef(0.5, 0.5, 0.5);
	//glTranslatef(0., 0., -5.);
	//glDisable(GL_TEXTURE_2D);
	
	glTranslatef(0., 0., -5.);
	DataContainer::DrawFullQuad(dest_test->width, dest_test->height, SrcImage->GetGpuImage());
	glTranslatef(0., 0., 5.);
	
	if(glIsEnabled(GL_TEXTURE_2D))
		std::cout << "GL_Texture2D enable" << std::endl;
	else
		std::cout << "GL_Texture2D disable" << std::endl;
	
	if(glIsEnabled(GL_LIGHTING))
		std::cout << "GL_LIGHTING enable" << std::endl;
	else
		std::cout << "GL_LIGHTING disable" << std::endl;
	
	/*
	 glEnable(GL_LIGHTING);
	 glEnable(GL_LIGHT0);
	 GLfloat Color [4] = {1., 1., 1., 1.};
	 glLightfv(GL_LIGHT0, GL_
	 AMBIENT, Color);
	 */
	
	
	glScalef(0.5, 0.5, 0.5);
	//glTranslatef(0., 0., -5.);
	glRotatef(30, 45., 0., 0.);
	glBegin(GL_QUADS);
	glColor4f(1., 1., 0., 1.);
	glVertex3f(-1, -1, 0.f);
	glVertex3f( 1, -1, 0.5f);
	glVertex3f( 1,  1, -0.5f);
	glVertex3f(-1,  1, -1.f);
	glEnd();
	
	glFlush();
	
	//cvgShowFrameBufferImage("Apply Result(RGBA*UNSIGNED_BYTE)", dest_test->width, dest_test->height,_GPUCV_FRAMEBUFFER_DFLT_FORMAT, GL_UNSIGNED_BYTE);
	
#if !USE_SwapBUFFER
	RenderBufferManager()->UnForce();
#else
	glutSwapBuffers();
#endif
	glDisable(GL_DEPTH_TEST);
	
	
	TextureDest->SetLocation(DataContainer::LOC_CPU);
	SrcImage->SetLocation(DataContainer::LOC_CPU);
	
#if TestDepthImg
	TextureDestDepth->SetLocation(DataContainer::LOC_CPU);
	cvShowImage("src",			  _cvgImageDestDepth->GetIplImage());
#else
	cvShowImage("src",			  SrcImage->GetIplImage());
#endif
	
	cvShowImage("renderBuffer CvgArr",  _cvgImageDest->GetIplImage());
	cvWaitKey(0);
	cvDestroyWindow("src");\
	cvDestroyWindow("renderBuffer CvgArr");\
	//=================
	GetTextureManager()->Delete(dest_test);
	delete TestGroup;
}

#endif
/*
 void runMatrixAvg(void)
 {
 IplImage *src3 = cvLoadImage("data/pictures/1x.jpg",1);
 IplImage *src4;IplImage *src5;IplImage * src6;IplImage *src7;IplImage * src8;IplImage * src9;IplImage * src10;
 src4 = cvLoadImage("data/pictures/2x.jpg",1);
 src5 = cvLoadImage("data/pictures/3.jpg",1);
 src6 = cvLoadImage("data/pictures/4.jpg",1);
 src7 = cvLoadImage("data/pictures/5.jpg",1);
 src8 = cvLoadImage("data/pictures/6.jpg",1);
 src9 = cvLoadImage("data/pictures/7.jpg",1);
 src10 = cvLoadImage("data/pictures/8.jpg",1);
 
 
 __CreateImages__(cvGetSize(src1) ,DepthSize, src1->nChannels, OperOpenCV|OperGLSL);
 
 __CreateWindows__();
 
 // _CV_benchLoop("Add",cvAdd(src1,src2, destCV), "");
 _GPU_benchLoop("MatrixAvg",cvgMatrixAvg(src3,src4,src5, src6,src7,src8,src9,src10),destGLSL , "");
 __ShowImages__();
 __ReleaseImages__();
 
 }
 */

void runVBO(IplImage * src1)
{
#if 0
	//IplImage * src2=NULL;
	IplImage * destGLSL=cvgCreateImage(cvGetSize(src1), IPL_DEPTH_8U, 3);
	//we render the src1 image into destGLSL using VBO
	
	DataContainer	*cvgSrc		= GPUCV_GET_TEX_ON_LOC(src1, DataContainer::LOC_GPU, true);
	DataContainer  *cvgDest	= GPUCV_GET_TEX_ON_LOC(destGLSL, DataContainer::LOC_GPU, false);
	
	
	//create VBO...
	GLsizeiptr PositionSize, ColorSize;
	GLsizeiptr VertexNbr = cvgSrc->_GetWidth() * cvgSrc->_GetHeight();
	GLuint buffSize = VertexNbr * cvgSrc->_GetNChannels() * sizeof(GLubyte);
	
#if 1
	PositionSize	= buffSize;
	ColorSize		= buffSize;
	
	GPUCV_DEBUG("VertexArray=========");
	GLubyte * data = (GLubyte *)*cvgSrc->_GetPixelsData();
	for (int i = 0; i < 16; i+=2)
	{
		GPUCV_DEBUG(i << ":" << (int)data[i] << "/" << (int)data[i+1]);
	}
	GPUCV_DEBUG("====================");
#else
	GLsizeiptr TextSize;
	VertexNbr = 4;
	PositionSize = 4 * 3 * sizeof(GLfloat);
	TextSize = 4 * 2 * sizeof(GLfloat);
	ColorSize = 4 * 3 * sizeof(GLfloat);
	GLfloat PositionData[] =
	{
		-1.0f,-1.0f, -0.5f,
		1.0f,-1.0f, -0.5f,
		1.0f, 1.0f, -0.5f,
		-1.0f, 1.0f, -0.5f
	};
	GLfloat ColorData[] =
	{
		255., 255., 255,
		255., 255., 255,
		255., 255., 255,
		255., 255., 255
	};
	GLfloat TextData[] =
	{
		0., 0.,
		1., 0.,
		1., 1.,
		0., 1.
	};
#endif
	
	
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
#if 0
	
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	
	GLBuffer VertexSrcBuffer(GLBuffer::ARRAY_BUFFER, GLBuffer::STATIC_DRAW, PositionSize, PositionData);
	VertexSrcBuffer.Bind();
	VertexSrcBuffer.SetPointer(GLBuffer::VERTEX_POINTER, 3, GL_FLOAT, 0,  0);
	
	
	
	GLBuffer TextureSrcBuffer(GLBuffer::ARRAY_BUFFER, GLBuffer::STATIC_DRAW, TextSize, TextData);
	TextureSrcBuffer.Bind();
	TextureSrcBuffer.SetPointer(GLBuffer::TEXTURE_POINTER, 2, GL_FLOAT, 0,  0);
#else
	
	GLBuffer VertexSrcBuffer(GLBuffer::ARRAY_BUFFER, GLBuffer::STATIC_DRAW, PositionSize, *cvgSrc->_GetPixelsData());
	VertexSrcBuffer.Bind();
	VertexSrcBuffer.SetPointer(GLBuffer::VERTEX_POINTER, 2, GL_SHORT, 1,  0);
	
	GLBuffer ColorSrcBuffer(GLBuffer::ARRAY_BUFFER, GLBuffer::STATIC_DRAW, ColorSize, *cvgSrc->_GetPixelsData());
	ColorSrcBuffer.Bind();
	ColorSrcBuffer.SetPointer(GLBuffer::COLOR_POINTER, 3, GL_FLOAT, 0,  0);
	
#endif
	
	
	_GPUCV_GL_ERROR_TEST();
	
	cvgDest->SetRenderToTexture();
	InitGLView(*cvgDest);
	glClearColor(0, 255, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	glTranslatef(0., 0., -0.5);
	glScalef(1/256., 1/256., 1/256.);
	glScalef(1/200., 1/200., 1/200.);
#if 0   //test render to Text
	
	DataContainer::DrawFullQuad(cvgDest->_GetWidth(), cvgDest->_GetHeight(), cvgSrc);
#else   //test VBO
	//glTranslatef(0., 0., -5);
	glColor4f(255., 255., 255., 255.);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	//glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	
	VertexSrcBuffer.Bind();
	//cvgSrc->_Bind();
	VertexSrcBuffer.DrawArrays(GL_TRIANGLE_STRIP, 0, 16);
	cvgSrc->_UnBind();
	VertexSrcBuffer.UnBind();
	VertexSrcBuffer.Disable();
	ColorSrcBuffer.Disable();
	//TextureSrcBuffer.Disable();
#endif
	cvgDest->UnsetRenderToTexture();
	
	VertexSrcBuffer.Disable();
	ColorSrcBuffer.Disable();
	//TextureSrcBuffer.Disable();
	
	_GPUCV_GL_ERROR_TEST();
	
	cvNamedWindow("VBOTest",1);
	cvgShowImage("VBOTest", destGLSL);
	cvWaitKey(0);
	cvDestroyWindow("VBOTest");
#endif
}







void runDist(IplImage * src1)
{
#if 0
	IplImage * src2=NULL;
	GPUCV_FUNCNAME("runDist");
	
	__CreateImages__(cvGetSize(src1) ,IPL_DEPTH_32F, 1, OperALL);
	
	IplImage *srcGray1 = NULL;
	IplImage *srcGray2 = NULL;
	
	
	if (src1->nChannels !=1)
	{
		srcGray1 = 	cvgCreateImage(cvGetSize(src1),IPL_DEPTH_8U,1);
		cvCvtColor(src1, srcGray1, CV_BGR2GRAY);
	}
	else
	{
		srcGray1 = cvCloneImage(src1);
	}
	//srcGray2 = 	cvgCreateImage(cvGetSize(src1),IPL_DEPTH_8U,2);
	
	//convert Scr image into contour image with sobel.
	//cvSobel(srcGray1, srcGray1, 1,1,3);
	cvThreshold(srcGray1,srcGray1, 180, 255, CV_THRESH_BINARY);
	cvThreshold(src1,src1, 180, 255, CV_THRESH_BINARY);
	
	__CreateWindows__();
	_CV_benchLoop("DistTransform", cvDistTransform(srcGray1,destCV, CV_DIST_L2), "");
	//	_GPU_benchLoop("DistTransform", cvgDistTransform(srcGray1,destGLSL, CV_DIST_L2,3,NULL,NULL), destGLSL, "");
	
	cvgSynchronize(destGLSL);
	//	cvScale(destCV, destCV, 10);
	//	cvScale(destGLSL, destGLSL, 10);
	
	int width = 15;
	for (int i = 0 ; i < 2 ; i ++)
	{for (int j = 0 ; j < 32 ; j ++)
	{printf ("\n pixel value opencv -- %d",destCV[i*width+j]);
		printf ("\t pixel value gpucv -- %d",destGLSL[i*width+j]);
		printf ("\t pixel value sOURCE -- %d",src1[i*width+j]);
	}
	}
	__ShowImages__();
	__ReleaseImages__();
	cvgReleaseImage(&srcGray1);
	
	
	
	
#endif
}




void runTestGeometry(IplImage * src1,IplImage * src2)
{
#if 0
	GPUCV_FUNCNAME("runTestGeometry");
	__CreateImages__(cvGetSize(src1) ,IPL_DEPTH_8U, 3, OperGLSL);
	
	_GPU_benchLoop("runTestGeometry",cvgTestGeometryShader(src1,destGLSL),destGLSL, "");
	
	__CreateWindows__();
	__ShowImages__();
	__ReleaseImages__();
#endif
}
void runStats(IplImage * src1)
{
	//IplImage * src2=NULL;
	/*
	 if( (src1 = cvLoadImage("data/pictures/stats4x4.bmp",1)) == 0 )
	 if( (src1 = cvLoadImage("../GPUCVExport/data/pictures/stats4x4.bmp",1)) == 0 )//for debugging under MS Visual C++
	 exit(0);
	 */
	//resizeImage(src1, 512,512);
	/*
	 IplImage *srcGray1 = cvgCreateImage(cvGetSize(src1),8,1);
	 if (src1->nChannels !=1)
	 {
	 printf("\nConverting Src image to gray\n");
	 cvCvtColor(src1, srcGray1, CV_BGR2GRAY);
	 }
	 else
	 {
	 printf("\nCloning Src images\n");
	 srcGray1 = cvCloneImage(src1);
	 }
	 //=====================================
	 
	 // _CV_benchLoop("cvMax", cvMax(srcGray1,srcGray2, destCV), "");
	 _GPU_benchLoop("cvgImageStat", cvgImageStat(srcGray1), NULL, "");
	 
	 
	 cvgReleaseImage(&srcGray1);
	 */
	
}

#if _GPUCV_DEPRECATED
void localS(std::string & Command)
{		
	std::string Curmd2;
	SGE::GetNextCommand(Command, Curmd2);
	std::istringstream b(Curmd2);
	float scale=0;
	b >> scale;
	SGE::GetNextCommand(Command, Curmd2);
	std::istringstream c(Curmd2);
	float shift=0;
	c >> shift;
	runLocalSum(GlobSrc1,scale,shift);
}
#endif

#if _GPUCV_DEVELOP_BETA
void runHamed(IplImage * src1)
{
	IplImage * src2 = NULL;
	__CreateImages__(cvGetSize(src1) ,DepthSize, src1->nChannels, OperOpenCV|OperGLSL);
	
	__CreateWindows__();
	_GPU_benchLoop("Hamed",cvgHamed(src1,destGLSL),destGLSL , "");
	__ShowImages__();
	cvgReleaseImage(&destGLSL);
}
#endif

