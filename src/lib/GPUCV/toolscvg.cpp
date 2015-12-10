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
#include <GPUCV/toolscvg.h>
#include <GPUCVCore/fps.h>
#include <GPUCVCore/coretools.h>
#include <GPUCV/cvgArr.h>
#include <GPUCV/cxtypesg.h>


namespace GCV{

//=================================================
GLuint GetCVDepth(const CvArr * arr)
{
	GLuint pixtype;
	SG_Assert(arr, "Array empty");
	if(CV_IS_IMAGE_HDR(arr))
		pixtype = ((IplImage*)arr)->depth;
	else if (CV_IS_MAT(arr))
		pixtype = ((CvMat*)arr)->type;
	else
		SG_Assert(0, "GetCVDepth()Unknown type");

	return pixtype;
}
//=================================================
GLuint GetGLDepth(const CvArr * arr)
{
	GLuint internal_format;
	GLuint format;
	GLuint pixtype;

	SG_Assert(arr, "Array empty");
	if(CV_IS_IMAGE_HDR(arr))
		pixtype = cvgConvertCVPixTypeToGL((IplImage*)arr);
	else if (CV_IS_MAT(arr))
		cvgConvertCVMatrixFormatToGL((CvMat*)arr, internal_format, format, pixtype);
	else
		SG_Assert(0, "GetCVDepth()Unknown type");

	return pixtype;
}
//=================================================
GLuint GetWidth(const CvArr * arr)
{
	SG_Assert(arr, "Array empty");
	if(CV_IS_IMAGE_HDR(arr))
		return cvGetSize(arr).width;
	else if (CV_IS_MAT(arr))
		return ((CvMat*)arr)->cols;
	else
		SG_Assert(0, "GetWidth()Unknown type");
	return 0;
}
//=================================================
GLuint GetHeight(const CvArr * arr)
{
	SG_Assert(arr, "Array empty");
	if(CV_IS_IMAGE_HDR(arr))
		return cvGetSize(arr).height;
	else if (CV_IS_MAT(arr))
		return ((CvMat*)arr)->rows;
	else
		SG_Assert(0, "GetHeight()Unknown type");
	return 0;
}
//=================================================
GLuint GetnChannels(const CvArr * arr)
{
	SG_Assert(arr, "Array empty");
	if(CV_IS_IMAGE_HDR(arr))
		return ((IplImage*)arr)->nChannels;
	else if (CV_IS_MAT(arr))
		return 1;//SG_Assert(0, "GetnChannels() for Matrix not manage yet");
	else
		SG_Assert(0, "GetnChannels()Unknown type");
	return 0;
}
//=================================================
const char * GetChannelSeq(const CvArr * arr)
{
	SG_Assert(arr, "Array empty");
	if(CV_IS_IMAGE_HDR(arr))
		return ((IplImage*)arr)->channelSeq;
	else if (CV_IS_MAT(arr))
		return "R";//SG_Assert(0, "GetChannelSeq for Matrix not manage yet");
	else
		SG_Assert(0, "GetChannelSeq()Unknown type");

	return NULL;
}
//=================================================
GpuFilter::FilterSize GetSize(const CvArr * arr)
{
	CvSize Size = cvGetSize(arr);
	return GpuFilter::FilterSize (Size.width, Size.height);
}

//=================================================
#if _GPUCV_DEPRECATED
GpuFilter::FilterSize *GetSize(const DataContainer * tex)
{
	return (GpuFilter::FilterSize *) tex;
	//return GpuFilter::FilterSize (size->_GetWidth(), size->_GetHeight());
}
#endif
//=================================================
//=================================================
#if _GPUCV_DEPRECATED
GPUCV_TEXT_TYPE GetTextureId(CvArr *img)
{
	SetThread();
	GPUCV_TEXT_TYPE ret = GPUCV_GET_TEX(img);
	UnsetThread();
	return ret;
}
#endif
//=================================================

#if _GPUCV_DEPRECATED
bool TestDstPointer(const IplImage* dst,const IplImage* src1)
{
	if (dst == src1)// && dst!=NULL)
	{
		GPUCV_WARNING("Warning : TestDstPointer(), both image pointer are equal.");
		return false;
	}
	return true;
}
//=======================================================
void cvgShowFrameBufferImage(const char* name, GLuint width, GLuint height, GLuint Format, GLuint PixelType)
{
	//DebugImgData = new unsigned char[3*width*height];
	int FormatCV = cvgConvertGLTexFormatToCV(Format);
	int TypeCV   = cvgConvertGLPixTypeToCV(PixelType);

	IplImage* test_debug = cvgCreateImage(cvSize(width, height),TypeCV,FormatCV);
	cvgReadPixels(0,0,width, height,Format,PixelType,test_debug->imageData);

	//	memcpy(test_debug->imageData,DebugImgData,3*width*height*sizeof(unsigned char));
	cvNamedWindow(name,1);
	cvShowImage(name,test_debug);
	cvWaitKey();
	cvgReleaseImage(&test_debug);
	cvDestroyWindow(name);
	//delete DebugImgData;
}
#endif


#define WRITE_convert_TO_BUFFER(SRC, VAL, TYPE, COUNTER)\
{\
	TYPE * CastBuffer = (TYPE*) & SRC;\
	CastBuffer[0] = VAL;\
	COUNTER+=sizeof(TYPE);\
}



int ConvertCvHaarFeatureIntoBuffer(CvHaarFeature* _inputHaarFeature, char * _outBuffer)
{
	SG_Assert(_inputHaarFeature, "_inputHaarFeature is NULL");
	SG_Assert(_outBuffer, "_outBuffer is NULL");

	memcpy(_outBuffer, _inputHaarFeature, sizeof(CvHaarFeature));
	return sizeof(CvHaarFeature);
}

int ConvertCvHaarClassifierIntoBuffer(CvHaarClassifier* _inputHaarClassifier, char * _outBuffer)
{
	SG_Assert(_inputHaarClassifier, "_inputHaarClassifier is NULL");
	SG_Assert(_outBuffer, "_outBuffer is NULL");

	int DataSizeWritten=0;

	WRITE_convert_TO_BUFFER(_outBuffer[0], _inputHaarClassifier->count, int, DataSizeWritten);
	//	(int*)(_outBuffer[0]) = _inputHaarClassifier->count;
	//		DataSizeWritten+=sizeof(int);

	for(int i = 0; i <  _inputHaarClassifier->count; i++)
	{
		DataSizeWritten += ConvertCvHaarFeatureIntoBuffer(&_inputHaarClassifier->haar_feature[i], &_outBuffer[DataSizeWritten]);
	}

	WRITE_convert_TO_BUFFER(_outBuffer[DataSizeWritten], *_inputHaarClassifier->threshold, float, DataSizeWritten);
	//	(float)(_outBuffer[DataSizeWritten]) = *_inputHaarClassifier->threshold;
	//		DataSizeWritten+=sizeof(float);

	WRITE_convert_TO_BUFFER(_outBuffer[DataSizeWritten], *_inputHaarClassifier->left, int, DataSizeWritten);
	//	(int)(_outBuffer[DataSizeWritten]) = *_inputHaarClassifier->left;
	//		DataSizeWritten+=sizeof(int);

	WRITE_convert_TO_BUFFER(_outBuffer[DataSizeWritten], *_inputHaarClassifier->right, int, DataSizeWritten);
	//	(int)(_outBuffer[DataSizeWritten]) = *_inputHaarClassifier->right;
	//		DataSizeWritten+=sizeof(int);

	WRITE_convert_TO_BUFFER(_outBuffer[DataSizeWritten], *_inputHaarClassifier->alpha, float, DataSizeWritten);
	/*	(float)(_outBuffer[DataSizeWritten]) = *_inputHaarClassifier->alpha;
	DataSizeWritten+=sizeof(float);
	*/
	return DataSizeWritten;
}

//return size of data written
int ConvertCvHaarStageClassifierIntoBuffer(CvHaarStageClassifier* _inputHaarStage, char * _outBuffer)
{
	SG_Assert(_inputHaarStage, "_inputHaarStage is NULL");
	SG_Assert(_outBuffer, "_outBuffer is NULL");

	int DataSizeWritten=0;

	/*	(int)(_outBuffer[0]) = _inputHaarStage->count;
	DataSizeWritten+=sizeof(int);

	(float)(_outBuffer[DataSizeWritten]) = _inputHaarStage->threshold;
	DataSizeWritten+=sizeof(float);
	*/
	for(int i = 0; i <  _inputHaarStage->count; i++)
	{
		DataSizeWritten += ConvertCvHaarClassifierIntoBuffer(&_inputHaarStage->classifier[i], &_outBuffer[DataSizeWritten]);
	}
	/*
	(int)(_outBuffer[DataSizeWritten]) = _inputHaarStage->next;
	DataSizeWritten+=sizeof(int);

	(int)(_outBuffer[DataSizeWritten]) = _inputHaarStage->child;
	DataSizeWritten+=sizeof(int);

	(int)(_outBuffer[DataSizeWritten]) = _inputHaarStage->parent;
	DataSizeWritten+=sizeof(int);
	*/
	return DataSizeWritten;
}

//haar tools....
/*
IplImage * ConvertHaarClassifierIntoIplImage(CvHaarClassifierCascade* _inputHaar)
{
char *Buffer = new char[0xfffff];

SG_Assert(_inputHaar, "_inputHaarStage is NULL");
//SG_Assert(Buffer, "_outBuffer is NULL");

int DataSizeWritten = 0;

for(int i = 0; i <  _inputHaar->count; i++)
{
DataSizeWritten += ConvertCvHaarStageClassifierIntoBuffer(&_inputHaar->stage_classifier[i], &Buffer[DataSizeWritten]);
}


}
*/

}//namespace GCV
