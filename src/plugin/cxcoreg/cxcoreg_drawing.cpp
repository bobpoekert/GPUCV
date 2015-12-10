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

/** \file cxcoreg_drawing.cpp
\author Yannick Allusse, Songbo SONG
\brief Contain the GPUCV function correspondance of cxarray.cpp

Drawing functions for curves and shapes, texts, Point Sets and Contours
correspond in Cxcore documentation to
"Drawing functions"
*/
#include "StdAfx.h"
#include <GPUCV/config.h>
#include <GPUCV/misc.h>
#include <cxcoreg/cxcoreg.h>


using namespace GCV;

//=====================================================================================================

void unitCircle(int slices);
void drawCircle(double radius, double x, double y,int slices);
void drawIntCircle(int i_radius, int i_x, int i_y,int width, int height,int slices);
void drawCircleCenter(double radius, double x, double y,int slices, CvScalar SlicesColor, CvScalar CenterColor);
void unitCircleCenter(int slices, CvScalar SlicesColor, CvScalar CenterColor);

/*!
*	\brief Draw a circle the old fashioned way : Discretised loop on angles
*	\brief All the slices are connected to the center of the circle
*	\param slices -> circle's slices number
*	\param SlicesColor -> Color for the Slices.
*	\param CenterColor -> Color for the Center.
*	\return none
*/
void unitCircleCenter(int slices, CvScalar SlicesColor, CvScalar CenterColor)
{
	float angle=0;
	static int LastSlicesNBR=0;
	static GLuint DisplayListCircle=0;

	//test if list exists...
	if (DisplayListCircle!=0)
	{
		if (	slices != LastSlicesNBR)
		{
			glDeleteLists(DisplayListCircle, 1);
			DisplayListCircle=0;
			LastSlicesNBR=slices;
		}
	}

	//create it if needed
	if (DisplayListCircle==0)
	{
		GPUCV_DEBUG("\nunitCircleCenter => Create Display List");
		DisplayListCircle=1;
		glNewList(DisplayListCircle, GL_COMPILE);
		glBegin(GL_TRIANGLE_FAN);//GL_LINE_STRIP//GL_TRIANGLES
		glColor4f(CenterColor.val[0], CenterColor.val[1], CenterColor.val[2], CenterColor.val[3]);
		glVertex3f(0,0,0);//center of the circle
		glColor4f(SlicesColor.val[0], SlicesColor.val[1], SlicesColor.val[2], SlicesColor.val[3]);
		for(int i=0;i<=slices;i++)
		{
			//angle one
			angle = i*2*MON_PI/(slices);
			//glColor4f(SlicesColor.val[0], SlicesColor.val[1], SlicesColor.val[2], SlicesColor.val[3]);//1.,1.,1.,1.);
			//			glColor3f(SlicesColor.val[0], SlicesColor.val[1], SlicesColor.val[2]);//1.,1.,1.,1.);
			glVertex3f(cos(angle),sin(angle),0);
			/*			//angle two
			angle = (i+1)*2*MON_PI/(slices);
			//			glColor4f(1.,1.,1.,1.);
			glVertex3f(cos(angle),sin(angle),0);
			*/
		}
		glEnd();
		glEndList();
	}

	glCallList(DisplayListCircle);
}
//=================================================
/*!
*	\brief Draw a circle using unitCircle(), move it to (x,y) position and scale it by radius
*	\return none
*/
void drawCircleCenter(double radius, double x, double y,int slices, CvScalar SlicesColor, CvScalar CenterColor)
{
	//moving and scaling unit circle(TODO : display list)
	glPushMatrix();
	glTranslated(x,y,0);
	glScaled(radius,radius,1.0);
	unitCircleCenter(slices, SlicesColor, CenterColor);
	glPopMatrix();
}
//=================================================
/*!
*	\brief Draw a circle the old fashioned way : Discretised loop on angles
*	\param slices -> circle's slices number
*	\return none
*/
void unitCircle(int slices)
{
	glBegin(GL_LINE_LOOP);
	float angle=0;
	for(int i=0;i<slices;i++)
	{
		angle = i*2*MON_PI/(slices);
		glVertex3f(cos(angle),sin(angle),0);
	}
	glEnd();
}
//=================================================
/*!
*	\brief Draw a circle using unitCircle(), move it to (x,y) position and scale it by radius
*/
void drawCircle(double radius, double x, double y,int slices)
{
	//moving and scaling unit circle(TODO : display list)
	glPushMatrix();
	glTranslated(x,y,0);
	glScaled(radius,radius,1.0);
	unitCircle(slices);
	glPopMatrix();
}
//=================================================
void drawIntCircle(int i_radius, int i_x, int i_y,int width, int height,int slices)
{
	double d_radius,d_x,d_y;

	d_x = 2*i_x/(double)width;//double(2*i_x)/(double)width-1.0;
	d_y = 2*i_x/(double)height;//1.-double(2*i_y)/(double)height;
	d_radius=2.0*(double)i_radius/(double)width ; // Problem : how to deal with non square pictures ? gluDisk ?

	drawCircle(d_radius,d_x,d_y,slices);
}
//=================================================
//===========================================================================================================
void cvgLine( IplImage* img, CvPoint pt1, CvPoint pt2, CvScalar color,int thickness, int line_type, int shift )
{
	if(shift)
	{
		cvgSynchronize(img);
		cvLine(img, pt1, pt2, color,thickness, line_type, shift);
		return;
	}

	GPUCV_START_OP(cvLine(img, pt1, pt2, color,thickness, line_type, shift),
		"cvgLine",
		(IplImage*)NULL,
		GenericGPU::HRD_PRF_1);

	SG_Assert(img, "No src image");
	DataContainer * DC_Src = GPUCV_GET_TEX(img);
	SG_Assert(DC_Src, "Could not retrieve DataContainer");


	if(!DC_Src->DataDscHaveData<DataDsc_GLTex>() && DC_Src->DataDscHaveData<DataDsc_CPU>())
	{//if image is on CPU, we don't draw on GPU
		cvLine(img, pt1, pt2, color,thickness, line_type, shift);
		return;
	}
	//else we draw on GPU


	DataDsc_GLTex * imgDDTex = DC_Src->GetDataDsc<DataDsc_GLTex>();
	SG_Assert(DC_Src, "Could not retrieve DataDsc_GLTex");

	//start drawing
	imgDDTex->SetRenderToTexture();
	imgDDTex->InitGLView();
	glTranslatef(-1.,-1.,0.);
	glLineWidth( (float) thickness );
	glEnable( GL_LINE_SMOOTH);
	glBegin(GL_LINES);
	glColor4f( color.val[0], color.val[1], color.val[2], color.val[3]);
	glVertex2f(2.*pt1.x/img->width,2.*pt1.y/img->height);
	glVertex2f(2.*pt2.x/img->width,2.*pt2.y/img->height);
	glEnd();
	glDisable( GL_LINE_SMOOTH);
	imgDDTex->UnsetRenderToTexture();

	GPUCV_STOP_OP(
		cvLine(img, pt1, pt2, color,thickness, line_type, shift),
		img, NULL, NULL, NULL
		);
}
//=================================================
void cvgRectangle( IplImage* img, CvPoint pt1, CvPoint pt2, CvScalar color,int thickness, int line_type, int shift )
{
	if(shift)
	{
		GPUCV_WARNING("cvgRectangle(): shift option not supported, going back to CPU");
		cvgSynchronize(img);
		cvRectangle(img, pt1, pt2, color,thickness, line_type, shift);
		return;
	}

	GPUCV_START_OP(cvRectangle(img, pt1, pt2, color,thickness, line_type, shift),
		"cvgCreateImage",
		(IplImage*)NULL,
		GenericGPU::HRD_PRF_1);
	//--
	SG_Assert(img, "No src image");
	DataContainer * DC_Src = GPUCV_GET_TEX(img);
	SG_Assert(DC_Src, "Could not retrieve DataContainer");


	if(!DC_Src->DataDscHaveData<DataDsc_GLTex>()
		&& DC_Src->DataDscHaveData<DataDsc_CPU>())
	{//if image is on CPU, we don't draw on GPU
		cvRectangle(img, pt1, pt2, color,thickness, line_type, shift);
		return;
	}
	//else we draw on GPU
	DataDsc_GLTex * imgDDTex = DC_Src->GetDataDsc<DataDsc_GLTex>();
	SG_Assert(DC_Src, "Could not retrieve DataDsc_GLTex");

	//start drawing
	imgDDTex->SetRenderToTexture();
	imgDDTex->InitGLView();
	glTranslatef(-1.,-1.,0.);
	glLineWidth( (float) thickness );
	glEnable( GL_LINE_SMOOTH);
	glBegin(GL_LINE_STRIP);
	glColor4f( color.val[0], color.val[1], color.val[2], color.val[3]);
	glVertex2f(2.*pt1.x/img->width,2.*pt1.y/img->height);
	glVertex2f(2.*pt2.x/img->width,2.*pt1.y/img->height);
	glVertex2f(2.*pt2.x/img->width,2.*pt2.y/img->height);
	glVertex2f(2.*pt1.x/img->width,2.*pt2.y/img->height);
	glVertex2f(2.*pt1.x/img->width,2.*pt1.y/img->height);
	glEnd();
	glDisable( GL_LINE_SMOOTH);
	imgDDTex->UnsetRenderToTexture();
	//--
	GPUCV_STOP_OP(
		cvRectangle(img, pt1, pt2, color,thickness, line_type, shift),
		img, NULL, NULL, NULL
		);
}
//=======================================================================
void cvgCircle(IplImage* img, CvPoint center, int radius, CvScalar color,int thickness, int line_type, int shift)
{
	if(shift)
	{
		cvgSynchronize(img);
		cvCircle(img, center, radius, color,thickness, line_type, shift);
		return;
	}

	GPUCV_START_OP(cvCircle(img, center, radius, color,thickness, line_type, shift),
		"cvgCircle",
		(IplImage*)NULL,
		GenericGPU::HRD_PRF_1);

	SG_Assert(img, "No src image");
	DataContainer * DC_Src = GPUCV_GET_TEX(img);
	SG_Assert(DC_Src, "Could not retrieve DataContainer");


	if(!DC_Src->DataDscHaveData<DataDsc_GLTex>() && DC_Src->DataDscHaveData<DataDsc_CPU>())
	{//if image is on CPU, we don't draw on GPU
		cvCircle(img, center, radius, color,thickness, line_type, shift);
		return;
	}
	//else we draw on GPU
	DataDsc_GLTex * imgDDTex = DC_Src->GetDataDsc<DataDsc_GLTex>();
	SG_Assert(DC_Src, "Could not retrieve DataDsc_GLTex");

	//start drawing
	imgDDTex->SetRenderToTexture();
	imgDDTex->InitGLView();
	glTranslatef(-1.,-1.,0.);
	glLineWidth( (float) thickness );
	glEnable( GL_LINE_SMOOTH);
	int slices=200;
	glColor4f( color.val[0], color.val[1], color.val[2], color.val[3]);
	drawIntCircle(radius, center.x,center.y, img->width, img->height,slices);
	glDisable( GL_LINE_SMOOTH);
	imgDDTex->UnsetRenderToTexture();

	GPUCV_STOP_OP(
		cvCircle(img, center, radius, color,thickness, line_type, shift),
		img, NULL, NULL, NULL
		);
}
//===========================================================================================================

//GLuint disttex()
//
//{
//     int x,y;
//
//	 unsigned char * DistData = new unsigned char [5*5];
//
//    for ( x=0;x<5;x++ )
//      for( y=0;y<5;y++ )
//      {  DistData[x*5+y] = (char)(sqrt(double((x-2)*(x-2)+(y-2)*(y-2))*20)); //(x-2)*(x-2)+(y-2)*(y-2)
//
//        printf("  %d  %d  %d  \n",x,y, DistData[x*5+y]);
//
//        }
//
//    printf("  0  0  %d  \n", DistData[0]);
//	return TexCreate(GL_RGB,5,5, GL_LUMINANCE, GL_UNSIGNED_BYTE, DistData, 0);
//
//
//
//}
//
//void cvgDistTex( const IplImage* src, IplImage* dst, int distance_type/*=CV_DIST_L2*/, int mask_size/*=3*/, const float* mask/*=NULL*/, IplImage* labels/*=NULL*/ )
//{
//
//	  //dst = cvCreateImage( cvGetSize(src), 8,1 );
//      //IplImage* dst = cvCreateImage( cvGetSize(src), 8,1 );
//
//     // cvCvtColor(src,dst,CV_RGB2GRAY);
//
//
//	SetThread();
//
//	if (dst->depth != IPL_DEPTH_32F)
//	{//error, image must be 32bit float
//	   GPUCV_NOTICE("\ncvgDistTex : depth != IPL_DEPTH_32F");
//	  // Sleep(1000);
//	   cvDistTransform(src,dst,distance_type,mask_size,mask,labels);return;
//	}
//	else if ((dst->nChannels != 1)||(src->nChannels != 1))
//	{//error, single channel
//	   GPUCV_NOTICE("\ncvgDistTex : nChannels != 1");
//	  // Sleep(1000);
//	   cvDistTransform(src,dst,distance_type,mask_size,mask,labels);return;
//	}
//	else if ( (distance_type==CV_DIST_L1) || (distance_type==CV_DIST_L2) ||
//		 	 (distance_type==CV_DIST_C) || (distance_type==CV_DIST_USER))
//	{//we do only geometric distance transform on GPU yet...
//	   GPUCV_NOTICE("\ncvgDistTex : distance_type != CVG_DIST_GEO");
//	  // Sleep(1000);
//  	   cvDistTransform(src,dst,distance_type,mask_size,mask,labels);return;
//	}
//
//
//
////	cvvWaitKey(0);
//
//
//	//case CVG_DIST_GEO
//	//get Image on CPU...
//    GetTextureManager()->GetIplImage(src);
//	GLuint th_d = GetTextureManager()->CreateGpuImage(dst);
//
//	//init context
//	RenderBufferManager()->SetContext(th_d, dst->width,dst->height);
//	glClearColor(0.,0.,1.,0.);
//	InitGLView(dst->width, dst->height);
////init depth buffer
//
//
//	//draw geometry
////	if (display_list != -1) glCallList(display_list);
////   else drawQuad(imageWidth,imageHeight);
//
//    unsigned int ImageSize = src->width * src->height * src->nChannels * (src->depth/8);
//	int Ipix=0;
//	float QuadSize = 1./src->height*5*10;
//
//
//	GLuint DistTex =  disttex();
//
//	glBindTexture(GetHardProfile()->GetTextType(), DistTex);
//
//	glEnable(GL_TEXTURE_2D);
////	glEnable(GL_DEPTH_TEST);
//	//glDisable(GL_DEPTH_TEST);
////	glDepthFunc(GL_LESS);
//    glClearDepth(1);
//
//
//
//
//
//     glTranslatef(-1.,-1.,0.);
//   drawQuad();
//
//
//	/*	_BENCH_GL("DRAW_DIST",
//			for (int PixY =0; PixY < src->height; PixY++)
//		    	for (int PixX =0; PixX < src->width; PixX++)
//			{
//				if ( src->imageData[Ipix]!=0 )
//				{   	glPushMatrix();
//    				glTranslatef((float)PixX/src->width, (float)PixY/src->height, 0.);
//
//                    glBegin(GL_QUADS);
//					{
//				 	//	glColor3f(1., 1., 1.);
//						if (GetHardProfile()->GetTextType() ==  GL_TEXTURE_2D)
//						{
//
//                        glTexCoord2f(0,0);glVertex3f(-QuadSize, -QuadSize, -0.5f);
//					    glTexCoord2f(1,0);glVertex3f( QuadSize, -QuadSize, -0.5f);
//				        glTexCoord2f(1,1);glVertex3f( QuadSize,  QuadSize, -0.5f);
//					    glTexCoord2f(0,1);glVertex3f(-QuadSize,  QuadSize, -0.5f);
//						}
//					}
//					glEnd();
//				     	glPopMatrix();
//
//					}
//				//else
//				//	printf("\nno draw: %d %d", PixX, PixY);
//				Ipix +=1;
//			}
//			, "",0,0);//end of _BENCH_GL
//*/
//  /*  glBegin(GL_LINES);
//   // printf("%f=======================\n",color.val[0]);
//
//
//   //glColor4f(color.val[0],color.val[1],color.val[2],color.val[3]); glVertex2f(pt1.x,pt2.y);
//    //glColor4f(color.val[0],color.val[1],color.val[2],color.val[3]); glVertex2f(pt1.x,pt2.y);
//
//   glColor3f(1.0f,1.0f,1.0f); glVertex2f(0.,0.f);
//   glColor3f(1.0f,1.0f,1.0f); glVertex2f(1.f,1.f);
//
//    glEnd();
//
//    */
//
////glPopMatrix();
//    _BENCH_GL("glFlush",glFlush(), "", 0,0);
//	_BENCH_GL("glFinish",glFinish(), "", 0, 0);
////==================================
//
////	cvvWaitKey(0);
//	_BENCH_GL("glReadPixels",glReadPixels(0,0,dst->width,dst->height,
//				  GL_LUMINANCE,
//				  GL_FLOAT,
//                  dst->imageData),"",dst->width,dst->height);
//
//    //debug
////	cvNamedWindow("Src Filter Image",1);
//	RenderBufferManager()->GetResult();
//	cvNamedWindow("Src Filter Image",1);
////	cvShowImage("Src Filter Image",src);
//	cvgShowFrameBufferImage("Dest Filter Image",dst->width,dst->height,GL_LUMINANCE ,GL_UNSIGNED_BYTE );//GL_LUMINANCE  GL_FLOAT
////	cvgShowFrameBufferImage("Dest Filter Image",dst->width,dst->height,cvgConvertCVTexFormatToGL(3), cvgConvertCVPixTypeToGL(dst->depth));
////	cvgShowFrameBufferImage("Dest Filter Image",dst->width,dst->height,cvgConvertCVTexFormatToGL(2), cvgConvertCVPixTypeToGL(dst->depth));
////	cvgShowFrameBufferImage("Dest Filter Image",dst->width,dst->height,cvgConvertCVTexFormatToGL(3), cvgConvertCVPixTypeToGL(dst->depth));
//	cvgShowFrameBufferImage("Dest Filter Image",dst->width,dst->height,cvgConvertCVTexFormatToGL(4), cvgConvertCVPixTypeToGL(dst->depth));
//	cvgShowFrameBufferImage("Dest Filter Image",dst->width,dst->height,cvgConvertCVTexFormatToGL(1), cvgConvertCVPixTypeToGL(IPL_DEPTH_32F));
//	cvgShowFrameBufferImage("Dest Filter Image",dst->width,dst->height,cvgConvertCVTexFormatToGL(2), cvgConvertCVPixTypeToGL(IPL_DEPTH_32F));
//	cvgShowFrameBufferImage("Dest Filter Image",dst->width,dst->height,cvgConvertCVTexFormatToGL(3), cvgConvertCVPixTypeToGL(IPL_DEPTH_32F));
//	cvgShowFrameBufferImage("Dest Filter Image",dst->width,dst->height,cvgConvertCVTexFormatToGL(4), cvgConvertCVPixTypeToGL(IPL_DEPTH_32F));
//*/
//	cvDestroyWindow("Src Filter Image");
//	//get back result into Dest
//
//
//
//	RenderBufferManager()->UnSetContext();
//
//	UnsetThread();
//
//}

