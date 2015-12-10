
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
#include <GPUCV/cxtypesg.h>
#include <GPUCV/toolscvg.h>

namespace GCV{

int		cvgConvertGLPixTypeToCV(GLuint PixelType)
{
	int PixelFormat;
	switch (PixelType)
	{
	case GL_UNSIGNED_BYTE 	:  	PixelFormat = IPL_DEPTH_8U; break;
	case GL_BYTE			:  	PixelFormat = IPL_DEPTH_8S; break;
	case GL_UNSIGNED_SHORT:	PixelFormat = IPL_DEPTH_16U;break;
	case GL_SHORT:				PixelFormat = IPL_DEPTH_16S;break;
	case GL_UNSIGNED_INT	: 	//PixelFormat = IPL_DEPTH_32U;
		GPUCV_WARNING("IPL_DEPTH_32U does not exist in OpenCV => converting GL_UNSIGNED_INT into IPL_DEPTH_32S instead of IPL_DEPTH_32U");
	case GL_INT			: 	PixelFormat = IPL_DEPTH_32S;break;
	case GL_FLOAT			: 	PixelFormat = IPL_DEPTH_32F;break;
	case GL_RGBA_FLOAT32_ATI : PixelFormat = IPL_DEPTH_32F;break;
	case GL_RGBA_FLOAT16_ATI : PixelFormat = IPL_DEPTH_32F;break;
	case GL_DOUBLE			 :	PixelFormat = IPL_DEPTH_64F;break;
		//		 case : PixelFormat =IPL_DEPTH_64F ;break;
		//		 case : PixelFormat =IPL_DEPTH_32S ;		  break;
	default :  GPUCV_ERROR("Critical : cvgConvertGLPixTypeToCV()=> Unknown pixel type...Using GL_UNSIGNED_INT...");
		PixelFormat = GL_INT;
		break;
	}
	return PixelFormat;
}
//=================================================
GLuint	cvgConvertCVPixTypeToGL(CvArr * image)
{
	return cvgConvertCVPixTypeToGL(GetCVDepth(image));
}
//=================================================
GLuint	cvgConvertCVPixTypeToGL(int format)	
{
	GLuint PixelFormat;
	switch (format)
	{
		//IplImages
	case IPL_DEPTH_8U:  PixelFormat = GL_UNSIGNED_BYTE;break;
	case IPL_DEPTH_8S:  PixelFormat = GL_BYTE;break;
	case IPL_DEPTH_16U: PixelFormat = GL_UNSIGNED_SHORT;break;
	case IPL_DEPTH_16S: PixelFormat = GL_SHORT;break;
		//case IPL_DEPTH_32 U: PixelFormat = GL_UNSIGNED_INT;		  break;
		// GPUCV_WARNING("IPL_DEPTH_32U does not exist in OpenCV => converting GL_UNSIGNED_INT into IPL_DEPTH_32S instead of IPL_DEPTH_32U");
		// break;
	case IPL_DEPTH_32S: PixelFormat = GL_INT;		break;		 
	case IPL_DEPTH_32F: PixelFormat = GL_FLOAT;	break;
	case IPL_DEPTH_64F: PixelFormat = GL_DOUBLE;	break;
	default :  GPUCV_ERROR("Critical : cvgConvertCVPixTypeToGL()=> Unknown pixel type...Using GL_UNSIGNED_INT...");
		PixelFormat = GL_INT;
		break;
	}
	return PixelFormat;
}
//=================================================
std::string GetStrCVPixelType(const IplImage * _ipl)
{
	return GetStrCVPixelType(_ipl->depth);
}
std::string GetStrCVPixelType(const int _depth)
{
	std::string PixelFormat;
	switch (_depth)
	{
	case IPL_DEPTH_8U:  PixelFormat = "IPL_DEPTH_8U";break;
	case IPL_DEPTH_8S:  PixelFormat = "IPL_DEPTH_8S";break;
	case IPL_DEPTH_16U: PixelFormat = "IPL_DEPTH_16U";break;
	case IPL_DEPTH_16S: PixelFormat = "IPL_DEPTH_16S";break;
	case IPL_DEPTH_32S: PixelFormat = "IPL_DEPTH_32S";break;
	case IPL_DEPTH_32F: PixelFormat = "IPL_DEPTH_32F";break;
	case IPL_DEPTH_64F: PixelFormat = "IPL_DEPTH_64F";break;

	default :  GPUCV_ERROR("Critical : cvgConvertCVPixTypeToGL()=> Unknown pixel type...Using GL_UNSIGNED_INT...");
		PixelFormat = "Unknown";
		break;
	}
	return PixelFormat;
}
std::string GetStrCVTextureFormat(const IplImage * _ipl)
{
	return GetStrCVTextureFormat(_ipl->nChannels, _ipl->channelSeq);
}
//=======================================================================
/** \todo Add compatibility with 2 channels images.*/
std::string GetStrCVTextureFormat(const GLuint format, const char * seq/*=NULL*/)
{
	string FormatStr;
	switch (format)
	{
	case 1: FormatStr = "nChannels-1, eq-...";break;//GL_RED
	case 3:  if (seq!=NULL)
				 if ((seq[0] == 'B') && (seq[1] == 'G') && (seq[2] == 'R'))
					 FormatStr ="CV_nChannels-3, seq-BGR";
				 else	FormatStr ="CV_nChannels-3, seq-RGB";
			 else FormatStr="CV_nChannels-3, seq-...";
			 break;
	case 4:  if (seq!=NULL)
				 if ((seq[0] == 'B') && (seq[1] == 'G') && (seq[2] == 'R'))
					 FormatStr ="CV_nChannels-4, seq-BGRA";
				 else FormatStr ="CV_nChannels-4, seq-RGBA";
			 else FormatStr="CV_nChannels-4, seq-...";
			 break;
	default: FormatStr = "Unknown CV texture format"; break;
	}
	return FormatStr;
}
//=================================================
/** \todo Add compatibility with 2 channels images.*/
int		cvgConvertGLTexFormatToCV(GLuint format, char * _seq/*="    "*/)
{
	//Yann.A 18/11/05 : ?? cvgConvertGLTexFormatToCV should be compatible with channelSeq ??TODO
	int FormatCV = 0;
	char seq[5];
	switch(format)
	{
	case GL_GREEN: 	FormatCV = 1; if (seq!=NULL)strcpy(seq,"G");break;
	case GL_BLUE: 		FormatCV = 1; if (seq!=NULL)strcpy(seq,"B");break;
	case GL_RED: 		FormatCV = 1; if (seq!=NULL)strcpy(seq,"R");break;
	case GL_LUMINANCE: FormatCV = 1; if (seq!=NULL)strcpy(seq,"R");break;//???is it correct for luminance?
	case GL_LUMINANCE_ALPHA: FormatCV = 2; if (seq!=NULL)strcpy(seq,"");break;	
	case GL_RGB :		FormatCV = 3; if (seq!=NULL)strcpy(seq,"RGB");break;
	case GL_BGR : 		FormatCV = 3; if (seq!=NULL)strcpy(seq,"BGR");break;
	case GL_RGBA :	 	FormatCV = 4; if (seq!=NULL)strcpy(seq,"RGBA");break;
	case GL_BGRA : 	FormatCV = 4; if (seq!=NULL)strcpy(seq,"BGRA");break;
	default : GPUCV_ERROR("Critical : cvgConvertGLTexFormatToCV()=> Unknown texture format...Using GL_RGBA");
		FormatCV = 4;
		break; 
	}
	return FormatCV;
}
//=================================================
GLuint	cvgConvertCVTexFormatToGL(CvArr * image)
{
	return cvgConvertCVTexFormatToGL(GetnChannels(image), GetChannelSeq(image));
}
//=================================================
GLuint	cvgConvertCVTexFormatToGL(int format, const char * _seq/*="    "*/)
{
	GLuint FormatGL;
	switch (format)
	{
	case 1: FormatGL = GL_LUMINANCE;break;//GL_LUMINANCE;break;//GL_RED
	case 2: FormatGL = GL_LUMINANCE_ALPHA;break;
	case 3: 
#if 1
		if (_seq!=NULL)
			if ((_seq[0] == 'B') && (_seq[1] == 'G') && (_seq[2] == 'R'))
				FormatGL = GL_BGR;
			else
				FormatGL = GL_RGB;
		else 
#endif
			FormatGL = GL_RGB;
		break;
	case 4: 
#if 1
		if (_seq!=NULL)
			if ((_seq[0] == 'B') && (_seq[1] == 'G') && (_seq[2] == 'R'))
				FormatGL = GL_BGRA;
			else
				FormatGL = GL_RGBA;
		else
#endif
			FormatGL = GL_RGBA;
		break;
	default:
		GPUCV_ERROR("Critical : cvgConvertCVTexFormatToGL()=> Unknown texture format...Using GL_RGBA");
		FormatGL = GL_RGBA;
		break;
	}
	return FormatGL;
}
//=================================================
GLuint	cvgConvertCVInternalTexFormatToGL(CvArr * image)
{
	return cvgConvertCVInternalTexFormatToGL(GetCVDepth(image), GetnChannels(image), GetChannelSeq(image));
}
//=================================================
GLuint	cvgConvertCVInternalTexFormatToGL(int pixType, int format, const char * _seq/*="    "*/)
{
	GLuint GLTexFormat = cvgConvertCVTexFormatToGL(format, _seq);
	GLuint GLPixType = cvgConvertCVPixTypeToGL(pixType);

	return MainGPU()->ConvertGLFormatToInternalFormat(GLTexFormat, GLPixType);
}
//=================================================
void cvgConvertCVMatrixFormatToGL(CvMat * _mat, GLuint & _internal_format, GLuint & _format, GLuint &_pixtype)
{
	SG_Assert(_mat, "No input matrix");
	cvgConvertCVMatrixFormatToGL(_mat->type, _internal_format, _format, _pixtype);
}
//=================================================
void cvgConvertCVMatrixFormatToGL(int _type, GLuint & _internal_format, GLuint & _format, GLuint & _pixtype)
{
	//	GLuint Depth;
	GLuint Type;
	GLuint Channel;

	Type	= CV_MAT_TYPE(_type);
	//Depth	= CV_MAT_DEPTH(_type);//not working..???
	Channel = CV_MAT_CN(_type);
	Type = _type & CV_MAT_TYPE_MASK;

	switch(Type)
	{		
	case CV_8UC1:
	case CV_8UC2:
	case CV_8UC3:
	case CV_8UC4	:   _pixtype=GL_UNSIGNED_BYTE;break;
	case CV_8SC1:
	case CV_8SC2:
	case CV_8SC3:
	case CV_8SC4	:   _pixtype=GL_BYTE;break;
		//! \todo Finish type conversion.....
	case CV_16UC1:
	case CV_16UC2:
	case CV_16UC3:
	case CV_16UC4:
		//	case CV_16U	:
		_pixtype=GL_UNSIGNED_SHORT;break;

	case CV_16SC1:
	case CV_16SC2:
	case CV_16SC3:
	case CV_16SC4:
		_pixtype=GL_SHORT;break;
	case CV_32SC1: 
	case CV_32SC2: 
	case CV_32SC3: 

	case CV_32SC4: 
		//	case CV_32S	:
		_pixtype=GL_INT;break;
		//SG_Assert(0, "cvgConvertCVMatrixFormatToGL(), type CV_32S not managed correctly on GPU");


	case CV_32FC1:  // _pixtype=GL_FLOAT;break;
	case CV_32FC2: //  _pixtype=GL_FLOAT;break;
	case CV_32FC3:  // _pixtype=GL_FLOAT;break;
	case CV_32FC4:   _pixtype=GL_FLOAT;break;
	case CV_64F: 
		if(ProcessingGPU()->IsDoubleSupported())
			_pixtype=GL_DOUBLE;
		else
		{
			SG_Assert(0, "cvgConvertCVMatrixFormatToGL(), type CV_64F not managed correctly on GPU");
		}
		break;
	case CV_USRTYPE1	:   //_pixtype=GL_UNSIGNED_BYTE;break;
	default:
		SG_Assert(0, "cvgConvertCVMatrixFormatToGL() type "<< Type <<" is unkown");
	}

	switch (Channel)
	{
	case 1: _format = GL_LUMINANCE; break;
	case 2: _format = GL_LUMINANCE_ALPHA; break;
	case 3: _format = GL_BGR; break;
	case 4: _format = GL_BGRA; break;
	default:
		SG_Assert(0, "cvgConvertCVMatrixFormatToGL() nChannel "<< Channel <<"=> case not done yet.");
	}

	_internal_format = ProcessingGPU()->ConvertGLFormatToInternalFormat(_format, _pixtype);
}
//==================================================================
int cvgConvertGLFormattoCVMatrix(GLuint _pixtype, GLuint _channels)
{
	GLuint Type;
	switch (_pixtype)
	{
	case GL_UNSIGNED_BYTE 	:  	Type = CV_8U; break;
	case GL_BYTE			:  	Type = CV_8S; break;
	case GL_UNSIGNED_SHORT	:	Type = CV_16U;break;
	case GL_SHORT:				Type = CV_16S;break;
	case GL_INT				: 	Type = CV_32S;break;
	case GL_FLOAT			: 	Type = CV_32F;break;
	case GL_RGBA_FLOAT32_ATI:	Type = CV_32F;break;
	case GL_DOUBLE			:	Type = CV_64F;break;

		//		case GL_UNSIGNED_INT	: 	//Type = CV_32U;
		//			GPUCV_WARNING("CV_32U does not exist in OpenCV => converting GL_UNSIGNED_INT into CV_32S instead of CV_32U");
		//		case GL_RGBA_FLOAT16_ATI:	
		//			GPUCV_WARNING("CV_16F does not exist in OpenCV => converting GL_UNSIGNED_INT into CV_32S instead of CV_32U");
		//		 case : Type =CV_64F ;break;
		//		 case : Type =CV_32S ;		  break;
	default :  GPUCV_ERROR("Critical : cvgConvertGLPixTypeToCV()=> Unknown pixel type...Using GL_UNSIGNED_INT...");
		Type = GL_INT;
		break;
	}
	Type = CV_MAKETYPE(Type, _channels);
	return Type;
}

}//namespace GCV

