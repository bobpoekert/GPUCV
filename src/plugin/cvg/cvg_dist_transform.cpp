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
#include <GPUCV/CvgOperators.h>
#include <cxcoreg/cxcoreg.h>
#include <cvg/cvg.h>
#include <highguig/highguig.h>

//---------------------

using namespace GCV;

void cvgDistTransform( CvArr* src, CvArr* dst, int distance_type/*=CV_DIST_L2*/, int mask_size/*=3*/,  float* mask/*=NULL*/, CvArr* labels/*=NULL*/ )
{
	SetThread();

	SG_Assert(GetCVDepth(dst) == IPL_DEPTH_32F, "depth != IPL_DEPTH_32F");
	//SG_Assert((GetnChannels(dst) == 1)&&(GetnChannels(src) == 1), "nChannels != 1");

	//case CVG_DIST_GEO
	//get Image on CPU...
	CvgArr *Input  = (CvgArr*)GetTextureManager()->Get<CvgArr>(src);
	CvgArr *Output = (CvgArr*)GetTextureManager()->Get<CvgArr>(dst);
	//temp image
	IplImage *	TmpDstIpl = cvgCreateImage(cvGetSize(src), IPL_DEPTH_32F, 3);
	CvgArr	*	TmpOutput = (CvgArr*)GetTextureManager()->Get<CvgArr>(TmpDstIpl);

	IplImage *	DepthIpl = cvgCreateImage(cvGetSize(src), IPL_DEPTH_32F, 1);
	CvgArr	*	DepthCvg = (CvgArr*)GetTextureManager()->Get<CvgArr>(DepthIpl);
	//-------
	DataDsc_CPU *	Input_CPU	= Input->SetLocation<DataDsc_CPU>(true);
	DataDsc_GLTex * Output_GLTex= Output->SetLocation<DataDsc_GLTex>(false);
	DataDsc_GLTex * TmpOutput_GLTex= TmpOutput->SetLocation<DataDsc_GLTex>(false);
	DataDsc_GLTex * Depth_GLTex= DepthCvg->SetLocation<DataDsc_GLTex>(false);

	//init context
	TextureGrp TexGrp;
	TexGrp.SetGrpType(TextureGrp::TEXTGRP_OUTPUT);
	TmpOutput_GLTex->_SetColorAttachment(DataDsc_GLTex::COLOR_ATTACHMENT0_EXT);
	TexGrp.AddTexture(TmpOutput);
	Depth_GLTex->_SetColorAttachment(DataDsc_GLTex::DEPTH_ATTACHMENT_EXT);
	TexGrp.AddTexture(DepthCvg);
	RenderBufferManager()->SetContext(&TexGrp, GetWidth(src), GetHeight(src));


	//TmpOutput_GLTex->SetRenderToTexture();
	//RenderBufferManager()->Force(th_d, dst->width,dst->height);

#if 1
	InitGLViewPerspective(*Output_GLTex,0.01,10000.0);
	//glDisable(GL_DEPTH_TEST);
#else
	//	InitGLView(dst->width, dst->height); 
	glEnable(GL_SCISSOR_TEST);
	glScissor(0,0,Output_GLTex->_Getwidth,dst->height);
	glViewport(0, 0, dst->width,dst->height);
	glMatrixMode(GL_PROJECTION);    
	glLoadIdentity();               
	///gluOrtho2D(-1, 1, -1, 1);       
	glMatrixMode(GL_MODELVIEW);     
	glLoadIdentity();
	//glEnable(GetHardProfile()->GetTextType());
#endif	


	_GPUCV_GL_ERROR_TEST();
	/*
	glViewport(0,0,dst->width,dst->height);						// Reset The Current Viewport
	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection Matrix
	gluOrtho2D(-1, 1, -1, 1);      
	// Calculate The Aspect Ratio Of The Window
	//	gluPerspective(45.0f,(GLfloat)dst->width/(GLfloat)dst->height,0.1f,100.0f);

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview Matrix
	glLoadIdentity();			

	//glShadeModel(GL_SMOOTH);							// Enable Smooth Shading
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);				// Black Background
	glClearDepth(1.0f);									// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
	glDepthFunc(GL_LEQUAL);								// The Type Of Depth Testing To Do
	//glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);	// R
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	
	//init depth buffer
	*/

	//draw geometry
	int Ipix=0;
	float QuadSize = 1.;///GetHeight(src);


	GLUquadricObj *coneObj = gluNewQuadric();
	gluQuadricDrawStyle(coneObj,GLU_FILL);
	GLfloat PI = 3.14;

	CvScalar ColorCenter = {0,0,0,0};
	CvScalar ColorSlices = {1,0,0,1};
	GLboolean backupMask;
	glGetBooleanv(GL_DEPTH_WRITEMASK, &backupMask);
	if (glIsEnabled(GL_DEPTH_TEST))
		printf("\nGL_DEPTH_TEST enable");
	else printf("\nGL_DEPTH_TEST disable");
	if (backupMask)
		printf("\nDepth mask enabled");
	else	printf("\nDepth mask disabled");

	_GPUCV_GL_ERROR_TEST();

	glClearColor(0.,0.,0.,0.);
	glClearDepth(0.); 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	for (int j=0; j<4; j++)
	{
		glPushMatrix();
		glTranslatef(0.,0.,-0.5);
		float ScaleVal=0.1;
		glScalef(ScaleVal, ScaleVal,ScaleVal);
		glTranslatef(0.5,0.5,0.);
		glDepthFunc(GL_ALWAYS);
		switch (j)
		{
		case 0:
			Ipix=0;
			glTranslatef(-1.,-1.,0);
			glRotatef(45,1,0,0);
			//glTranslatef(-0.5,-0.5,0.);
			break;
		case 1:
			Ipix=0;
			//glDepthFunc(GL_EQUAL);
			glTranslatef(-1.,0.,0);
			glRotatef(-45,1,0,0);
			break;
		case 2:
			Ipix=0;
			//glDepthFunc(GL_GREATER);
			glTranslatef(0.,-1.,0);
			break;
		case 3:
			Ipix=0;
			glTranslatef(0.,0.,0.);
			//glDepthFunc(GL_NEVER);
			break;
		}
		ScaleVal=0.2;
		glScalef(ScaleVal, ScaleVal,ScaleVal);

		uchar * SrcBuffer =	(uchar*) *Input_CPU->_GetPixelsData();

#if 1		
		for (unsigned int PixY =0; PixY < GetHeight(src); PixY++)
			for (unsigned int PixX =0; PixX < GetHeight(src); PixX++)
			{
				if (SrcBuffer[Ipix]>1)
				{//draw cone...
					glPushMatrix();
					glTranslatef((float)PixX/GetWidth(src), (float)PixY/GetHeight(src), 0.);
					glColor3f(1-(float)PixY/GetWidth(src),1-(float)PixX/GetHeight(src),0.);
					//glColor3f(0,1,1.);
					//drawIntCircle(1., PixX, PixY, GetHeight(src), GetHeight(src), 10);
					//glScalef(1.,1.,-1.);
					gluCylinder(coneObj, QuadSize/GetHeight(src), 1,1, 10, 3 );
					//	glTranslatef(0,0, -0.2);
					//	glColor3f(1,1,1);
					//	gluCylinder(coneObj, QuadSize/GetHeight(src)*2, QuadSize/GetHeight(src)*2,0.5, 10, 3 );
					glPopMatrix();
				}
				Ipix +=1;
			}
			Ipix=0;
			for (unsigned int PixY =0; PixY < GetHeight(src); PixY++)
				for (unsigned int PixX =0; PixX < GetHeight(src); PixX++)
				{
					if (SrcBuffer[Ipix]>1)
					{//draw cone...
						glPushMatrix();
						glTranslatef((float)PixX/GetHeight(src), (float)PixY/GetHeight(src), 0.);
						glTranslatef(0,0, -0.2);
						glColor3f(1,1,1);
						gluCylinder(coneObj, QuadSize/GetHeight(src)*2, QuadSize/GetHeight(src)*2,0.5, 10, 3 );
						glPopMatrix();
					}
					Ipix +=1;
				}
#endif
				//Input_CPU->_SetPixelsData(DataDsc_CPU** SrcBuffer );
				glPopMatrix();
	}



	// Done Drawing The Quad
	//TmpOutput_GLTex->UnsetRenderToTexture();
	RenderBufferManager()->UnSetContext();
	//	glPopMatrix();
	glFlush();
	glFinish();
	//==================================



	cvNamedWindow("TempImg", 1);
	cvgShowImage("TempImg",TmpDstIpl);
	cvNamedWindow("TempDepth", 1);
	cvgShowImage("TempDepth",DepthIpl);
	cvWaitKey(0);

	/*	
	//RenderBufferManager()->UnForce();

	cvvWaitKey(0);
	cvgReadPixels(0,0,GetWidth(dst),GetHeight(dst),
	GL_LUMINANCE,
	GL_FLOAT,            
	DstBuffer->imageData); 

	//debug
	//	cvNamedWindow("Src Filter Image",1);
	RenderBufferManager()->GetResult();
	cvNamedWindow("Src Filter Image",1);
	//cvgShowFrameBufferImage("Dest Filter Image",GetWidth(dst),GetHeight(dst),cvgConvertCVTexFormatToGL(GetnChannels(dst)), GL_UNSIGNED_BYTE);//cvgConvertCVPixTypeToGL(dst->depth)
	//cvgShowFrameBufferImage("Dest Filter Image",GetWidth(dst),GetHeight(dst),cvgConvertCVTexFormatToGL(4,"RGBA"), GL_FLOAT);//cvgConvertCVPixTypeToGL(dst->depth)
	//cvDestroyWindow("Src Filter Image");*/ 
	/*cvgShowImage("Src Filter Image",src);
	/get back result into Dest*/

	UnsetThread();

#if 0	

	if  (distance_type==CV_DIST_L1)
		TemplateOperator("cvgDistTransform", "FShaders/distt_city", "",
		src, NULL, NULL,
		dst,NULL,0,
		TextureGrp::TEXTGRP_SAME_ALL_FORMAT,"");
	if (distance_type==CV_DIST_L2) 
		TemplateOperator("cvgDistTransform", "FShaders/distt_euclid", "",
		TmpDstIpl, NULL, NULL,
		dst,NULL,0,
		TextureGrp::TEXTGRP_SAME_ALL_FORMAT,"");
	if (distance_type==CV_DIST_C) 
		TemplateOperator("cvgDistTransform", "FShaders/distt_chess", "",
		src, NULL, NULL,
		dst,NULL,0,
		TextureGrp::TEXTGRP_SAME_ALL_FORMAT,"");
#endif	
}

