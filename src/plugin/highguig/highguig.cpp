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
#include <highguig/highguig.h>
#include <cxcoreg/cxcoreg.h>
#include <GPUCV/misc.h>
#include <stdio.h>
#include <string>


using namespace GCV;

//=============================================
_GPUCV_HIGHGUIG_EXPORT_C LibraryDescriptor *modGetLibraryDescriptor(void)
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
		pLibraryDescriptor->SetDllName("highguig");
		pLibraryDescriptor->SetImplementationName("GLSL");
		pLibraryDescriptor->SetBaseImplementationID(GPUCV_IMPL_GLSL);
		pLibraryDescriptor->SetUseGpu(true);
	}
	return pLibraryDescriptor;
}
//=============================================
void cvgShowImage(const char* name, CvArr* image)//, GLuint _mipmapLevel/*=0*/)
{
	GPUCV_START_OP(cvShowImage(name,image),
		"cvgShowImage()",
		image,
		GenericGPU::HRD_PRF_0);

	//ShowOpenGLError(__FILE__, __LINE__);
	DataContainer * texImage = NULL;
	texImage = GetTextureManager()->Find(image);


	/*if(!texImage)	
	{//is image a DataContainer..., no opencv case, but usefull for debugging.
	texImage = dynamic_cast<DataContainer*> (image);
	}
	*/

	if(!texImage)//not a GPU Image
	{
		cvShowImage(name, image);
		return;
	}

	if (texImage->GetOption(DataContainer::VISUAL_DBG))
		cvgShowImage2(name,(IplImage *)image);

#if 0
	if(_mipmapLevel == 0)
	{
#endif		
		texImage->PushSetOptions(DataContainer::UBIQUITY, true);//set ubicuity flag...so we can keep GPU image
		texImage->SetLocation<DataDsc_IplImage>(true);//get to CPU
		CvgArr * cvgImg = dynamic_cast<CvgArr *> (texImage);
		if(cvgImg)
			cvShowImage(name, image);//show it
		else
			SG_Assert(0, "case not done=> conversion from DataContainer to CvgArr!!!");
		//print average...
		//CvScalar avg = cvAvg(image);
		//printf("\ncvAvg '%s': %lf, %lf %lf %lf", name, avg.val[0], avg.val[1], avg.val[2], avg.val[3]);

		//
		texImage->PopOptions();//restore previous options
#if 0
	}
	else
	{//we want to see some of the mipmap images...
		//reduce temp image size
		CvSize tempSize =  cvGetSize(image);
		tempSize.width  /= 1 << _mipmapLevel;
		tempSize.height /= 1 << _mipmapLevel;

		//create temp image
		IplImage * IplMipMap	= cvgCreateImage(tempSize, GetDepth(image), GetnChannels(image));
		DataContainer * TexMipMap	= GPUCV_GET_TEX(IplMipMap);//, DataContainer::LOC_GPU, false);//create texture on gpu
		DataDsc_GLTex * DD_GLTexMipMap = TexMipMap->SetLocation<DataDsc_GLTex>(false);

		//create small image that contains mipmap...
		glPushMatrix();
		texImage->SetLocation<DataDsc_GLTex>(true);
		//TexMipMap->_CreateLocation(DataContainer::LOC_CPU);
		DD_GLTexMipMap->SetRenderToTexture();
		DD_GLTexMipMap->InitGLView();
		glClearColor(0., 0., 1.,0);
		glClear(GL_COLOR_BUFFER_BIT);
		texImage->GetDataDsc<DataDsc_GLTex>()->DrawFullQuad(tempSize.width, tempSize.height);

		glFlush();
		glFinish();
		//TexMipMap->FrameBufferToCpu(true);
		DD_GLTexMipMap->UnsetRenderToTexture();
		//TexMipMap->_AddLocation(DataContainer::LOC_CPU);
		glPopMatrix();

		//create full size image that will show mipmap filter effects...
		IplImage * IplMipMapFull			= cvgCreateImage(cvGetSize(image), GetDepth(image), GetnChannels(image));
		DataContainer * TexMipMapFull		= GPUCV_GET_TEX(IplMipMapFull);//, DataContainer::LOC_GPU, false);//create texture on gpu
		DataDsc_GLTex * DD_GLTexMipMap_FULL = TexMipMapFull->SetLocation<DataDsc_GLTex>(false);

		glPushMatrix();
		TexMipMap->SetLocation<DataDsc_GLTex>(true);
		//TexMipMap->_CreateLocation(DataContainer::LOC_CPU);
		DD_GLTexMipMap_FULL->SetRenderToTexture();
		DD_GLTexMipMap_FULL->InitGLView();
		glClearColor(0., 0., 1.,0);
		glClear(GL_COLOR_BUFFER_BIT);
		TexMipMap->GetDataDsc<DataDsc_GLTex>()->DrawFullQuad(tempSize.width, tempSize.height);
		glFlush();
		glFinish();
		//TexMipMap->FrameBufferToCpu(true);
		DD_GLTexMipMap_FULL->UnsetRenderToTexture();
		//TexMipMap->_AddLocation(DataContainer::LOC_CPU);
		glPopMatrix();

		cvgShowImage(name, IplMipMapFull);//show temp image.
		cvgReleaseImage(&IplMipMap);
		cvgReleaseImage(&IplMipMapFull);
	}
#endif
	GPUCV_STOP_OP(
		cvShowImage(name,image),
		src, NULL, NULL, NULL
		);
}
//============================================================
void cvgShowImage2(const char* name, IplImage* image)
{	
	SG_Assert(image, "cvgShowImage2() no input image");
	int y_arb=10;
	cvgSynchronize(image);
	CvSize ImgSize = cvGetSize(image);
	CvArr* temp=cvCreateImage(ImgSize,image->depth,image->nChannels);
	cvCopy(image,temp,0);

	DataContainer * texImage = NULL;
	texImage = GetTextureManager()->Find(image);

	std::ostringstream _stream;
	std::ostringstream _stream2;

	//_stream << "=========================================================";
	if(texImage->GetLabel() != "")
	{
		_stream << "\tTexture Label:			"	<< texImage->GetValStr() << std::endl ;
	}
	_stream << std::endl;
	if(texImage->GetLastDataDsc())
	{
		DataDsc_Base* DDBase = texImage->GetLastDataDsc();
		DDBase->operator << (_stream2);
		//_stream2 << "\tTexture DD:			"	<< *texImage->GetLastDataDsc() << std::endl;
	}
	/*
	_stream << "\tDescription:			" << ;
	_stream << "=========================================================";
	_stream << "\tShwoing all locations informations:";

	for (int i = 0; i < m_textureLocationsArrLastID; i++)
	{
	_stream << texImage->GetLastDataDsc();
	/*}
	_stream << "=========================================================";*/


	CvFont f1;
	CvPoint start;
	start.x = 5;
	start.y = 13;

	//intialising and writing out the font
	cvInitFont(&f1,CV_FONT_HERSHEY_PLAIN,.81,.81,0,1,8);


	//printing label 
	std::string image_label = _stream.str();
	const char* h = image_label.c_str();
	CvSize text_size=cvSize(0,0);

	int* y =  &y_arb;
	cvGetTextSize(h,&f1,&text_size,y);
	int x2 = start.x+text_size.width;
	int y2 = start.y+text_size.height;
	cvRectangle(temp,start,cvPoint(x2,y2),cvScalar(255,255,255),
		CV_FILLED, 8, 0 );
	cvPutText(temp,h,start,&f1,CV_RGB( 0,0,0));


	//printing data descriptal
	std::string data_dsc_text = _stream2.str();
	std::string current;
	string::size_type loc; 
	const char* data_dsc_text_char;

	do{
		loc = data_dsc_text.find("\n",2);
		if( loc != string::npos ) {
			current = data_dsc_text.substr(0,loc);
			data_dsc_text = data_dsc_text.substr(loc);
		} 
		else
		{	
			current = data_dsc_text;
			data_dsc_text = "";
		}

		data_dsc_text_char = current.c_str();
		printf (data_dsc_text_char);

		start.y += 10;

		CvSize text_size_dd=cvSize(0,0);

		int* y_dd =  &y_arb;

		cvGetTextSize(data_dsc_text_char,&f1,&text_size_dd,y_dd);
		x2 = start.x + text_size_dd.width + 10;
		y2 = start.y + text_size_dd.height;
		cvRectangle(temp,start,cvPoint(x2,y2),cvScalar(255,255,255),
			CV_FILLED, 8, 0 );

		data_dsc_text_char = current.c_str();
		cvPutText(temp,data_dsc_text_char,start,&f1,CV_RGB( 0,0,0));
	}
	while (data_dsc_text.compare("")!=0);



	//displyaing the image

	cvShowImage(name, temp);
}
//====================================================================================
void cvgShowMatrix(const char* label,CvMat *mat,CvPoint start,int width ,int height)
{	
	if (width == 0 || height == 0)
	{
		GPUCV_WARNING("Box size unspecified , default 32x32 assumed");
		width = 32;
		height = 32;
	}
	GLuint InternalFormat=0;
	GLuint Format=0;
	GLuint PixType=0;
	int mat_type = mat->type;
	//cvgConvertCVMatrixFormatToGL(mat_type,InternalFormat,Format,PixType);
	GLuint Type;
	Type	= CV_MAT_TYPE(mat_type);

	DataContainer * DC_mat = GetTextureManager()->Find(mat);

	//if (texImage->GetOption(DataContainer::VISUAL_DBG))

	DC_mat->PushSetOptions(DataContainer::UBIQUITY, true);//set ubicuity flag...so we can keep GPU image
	DC_mat->SetLocation<DataDsc_CvMat>(true);//get to CPU
	CvgArr * cvgImg = dynamic_cast<CvgArr *> (DC_mat);
	SG_Assert(cvgImg, "case not done=> conversion from DataContainer to CvgArr!!!");
	
#define MATRIX_TYPE(TYPE,CHANNELS)\
	{cvgShowMatrixt<TYPE,CHANNELS>(label,mat,start,mat->rows,mat->cols,width,height);}

	switch(Type)
	{
	case CV_8UC1: MATRIX_TYPE(uchar,1);break;
	case CV_8UC2: MATRIX_TYPE(uchar,2);break;
	case CV_8UC3: MATRIX_TYPE(uchar,3);break;
	case CV_8UC4: MATRIX_TYPE(uchar,4);break;
	case CV_8SC1: MATRIX_TYPE(char,1);break;
	case CV_8SC2: MATRIX_TYPE(char,2);break;
	case CV_8SC3: MATRIX_TYPE(char,3);break;
	case CV_8SC4: MATRIX_TYPE(char,4);break;
	case CV_16SC1: MATRIX_TYPE(int,1);break;
	case CV_16SC2: MATRIX_TYPE(int,2);break;
	case CV_16SC3: MATRIX_TYPE(int,3);break;
	case CV_16SC4: MATRIX_TYPE(int,4);break;
		/*case CV_16UC1: MATRIX_TYPE(uint,1);break;
		case CV_16UC2: MATRIX_TYPE(uint,2);break;
		case CV_16UC3: MATRIX_TYPE(uint,3);break;
		case CV_16UC4: MATRIX_TYPE(uint,4);break;*/
	case CV_32FC1: MATRIX_TYPE(float,1);break;
	case CV_32FC2: MATRIX_TYPE(float,2);break;
	case CV_32FC3: MATRIX_TYPE(float,3);break;
	case CV_32FC4: MATRIX_TYPE(float,4);break;
	case CV_64FC1: MATRIX_TYPE(double,1);break;
	case CV_64FC2: MATRIX_TYPE(double,2);break;
	case CV_64FC3: MATRIX_TYPE(double,3);break;
	case CV_64FC4: MATRIX_TYPE(double,4);break;	
	}
	DC_mat->PopOptions();//restore previous options
}



void cvgShowImage3(const char* label,IplImage *name,  unsigned int start_x/*=0*/, unsigned int start_y/*=0*/, unsigned int width/*=0*/, unsigned int height/*=0*/)
{	
	CvPoint start=cvPoint (start_x, start_y);//to be C compliant.
	if (width == 0 || height == 0)
	{GPUCV_NOTICE ("Box size unspecified , default 32x32 assumed");
	width = 32;
	height = 32;
	}
	unsigned int Type = name->depth;
	unsigned int channels = name->nChannels;

#define IMAGE_TYPE(TYPE,CHANNELS,OFFSET,DISTYPE)\
	{cvgShowImaget<TYPE,CHANNELS,DISTYPE>(label,name,start,name->width,name->height,width,height,OFFSET);}
	switch(channels)
	{
	case 1:switch(Type)
		   {
	case IPL_DEPTH_8U: IMAGE_TYPE(char,1,0,int);break;
	case IPL_DEPTH_8S: IMAGE_TYPE(char,1,0,int);break;
	case IPL_DEPTH_16U: IMAGE_TYPE(unsigned int,1,0,unsigned int);break;
	case IPL_DEPTH_16S: IMAGE_TYPE(int,1,0,int);break;
	case IPL_DEPTH_32F: IMAGE_TYPE(float,1,0,float);break;	
	case IPL_DEPTH_32S: IMAGE_TYPE(signed int,1,0,signed int);break;	
	case IPL_DEPTH_64F: IMAGE_TYPE(double,1,0,double);break;	
		   };break;

	case 2:switch(Type)
		   {
	case IPL_DEPTH_8U: IMAGE_TYPE(char,2,0,int);break;
	case IPL_DEPTH_8S: IMAGE_TYPE(char,2,0,int);break;
	case IPL_DEPTH_16U: IMAGE_TYPE(unsigned int,2,0,unsigned int);break;
	case IPL_DEPTH_16S: IMAGE_TYPE(int,2,0,int);break;
	case IPL_DEPTH_32F: IMAGE_TYPE(float,2,0,float);break;	
	case IPL_DEPTH_32S: IMAGE_TYPE(signed int ,2,0,signed int);break;	
	case IPL_DEPTH_64F: IMAGE_TYPE(double,2,0,double);break;	
		   };break;
	case 3:switch(Type)
		   {
	case IPL_DEPTH_8U: IMAGE_TYPE(char,3,0,int);break;
	case IPL_DEPTH_8S: IMAGE_TYPE(char,3,0,int);break;
	case IPL_DEPTH_16U: IMAGE_TYPE(unsigned int,3,0,unsigned int);break;
	case IPL_DEPTH_16S: IMAGE_TYPE(int,3,0,int);break;
	case IPL_DEPTH_32F: IMAGE_TYPE(float,3,0,float);break;	
	case IPL_DEPTH_32S: IMAGE_TYPE(signed int,3,0,signed int);break;	
	case IPL_DEPTH_64F: IMAGE_TYPE(double,3,0,double);break;	
		   };break;
	case 4:switch(Type)
		   {
	case IPL_DEPTH_8U: IMAGE_TYPE(char,4,128,int);break;
	case IPL_DEPTH_8S: IMAGE_TYPE(char,4,0,int);break;
	case IPL_DEPTH_16U: IMAGE_TYPE(unsigned int,4,0,unsigned int);break;
	case IPL_DEPTH_16S: IMAGE_TYPE(int,4,0,int);break;
	case IPL_DEPTH_32F: IMAGE_TYPE(float,4,0,float);break;	
	case IPL_DEPTH_32S: IMAGE_TYPE(signed int ,4,0,signed int);break;	
	case IPL_DEPTH_64F: IMAGE_TYPE(double,4,0,double);break;	
		   };break;

	}
}
//==========================================================================================
int cvgSaveImage( const char* filename, CvArr* image )
{
	GPUCV_START_OP(return cvSaveImage(filename, image),
		"cvgSaveImage", 
		image,
		GenericGPU::HRD_PRF_1);

	cvgSynchronize(image);
	return cvSaveImage(filename, image);

	GPUCV_STOP_OP(
		return cvSaveImage(filename, image),
		image, NULL, NULL, NULL
		);
	return -1;
}
/*====================================*/
IplImage* cvgQueryFrame(CvCapture* capture)
{
	IplImage * NewFrame = NULL;
	GPUCV_START_OP(NewFrame=cvRetrieveFrame(capture),
		"cvgRetrieveFrame", 
		NewFrame,
		GenericGPU::HRD_PRF_0);

	NewFrame=cvQueryFrame((CvCapture*) capture);
	if (NewFrame)
		cvgSetDataFlag<DataDsc_IplImage>(NewFrame, true, true);

	GPUCV_STOP_OP(
		NewFrame=cvRetrieveFrame(capture),
		NewFrame, NULL, NULL, NULL
		);
	return NewFrame;
}
//============================================================
IplImage* cvgRetrieveFrame(CvCapture* capture, int streamIdx)
{
	IplImage * NewFrame = NULL;
	GPUCV_START_OP(NewFrame=cvRetrieveFrame(capture),
		"cvgRetrieveFrame", 
		NewFrame,
		GenericGPU::HRD_PRF_0);
#if CV_MAJOR_VERSION > 1
	NewFrame=cvRetrieveFrame(capture, streamIdx);
#else
	NewFrame=cvRetrieveFrame(capture);
#endif
	if (NewFrame)
		cvgSetDataFlag<DataDsc_IplImage>(NewFrame, true, true);

	GPUCV_STOP_OP(
		NewFrame=cvRetrieveFrame(capture),
		NewFrame, NULL, NULL, NULL
		);
	return NewFrame;
}
//============================================================
