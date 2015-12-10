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
#ifndef __GPUCV_CXTYPESG_H 
#define __GPUCV_CXTYPESG_H

#include <cxtypes.h>
#include <GPUCV/config.h>

namespace GCV{
/** \file
*	\brief Supply conversion function between GPUCV internal core objects and OpenCv objects.
*/
/** @addtogroup GPUCV_CONVERSION_GRP
*	@{
*/
/** \name IplImage format and type.
@{*/

#ifndef __BEGIN__ //defined in opencv 1.* but not in opencv 2.*
	#define __BEGIN__	__CV_BEGIN__
	#define __END__		__CV_END__
#endif
/** brief Convert OpenGL texture format to OpenCV.*/
_GPUCV_EXPORT int cvgConvertGLTexFormatToCV(GLuint format, char * _seq=NULL);//4 spaces
/** brief Convert OpenGL pixel type to OpenCV.*/
_GPUCV_EXPORT int cvgConvertGLPixTypeToCV(GLuint PixelType);
/** brief Convert OpenCV image format to OpenGL texture format.*/
_GPUCV_EXPORT GLuint cvgConvertCVTexFormatToGL(int format, const char * _seq=NULL);//4 spaces
/** brief Convert OpenCV image format to OpenGL texture format.*/
_GPUCV_EXPORT GLuint cvgConvertCVTexFormatToGL(CvArr * image);
/** brief Convert OpenCV image internal format to OpenGL texture internal format.*/
_GPUCV_EXPORT GLuint cvgConvertCVInternalTexFormatToGL(int pixType, int format, const char * _seq="    ");
/** brief Convert OpenCV image internal format to OpenGL texture internal format.*/
_GPUCV_EXPORT GLuint cvgConvertCVInternalTexFormatToGL(CvArr * image);
/** brief Convert OpenCV pixel type(depth) to OpenGL pixel type.*/
_GPUCV_EXPORT GLuint cvgConvertCVPixTypeToGL(int format);
/** brief Convert OpenCV pixel type(depth) to OpenGL pixel type.*/
_GPUCV_EXPORT GLuint cvgConvertCVPixTypeToGL(CvArr * image);
/**@}*/

/** \name CvMat format and type.
@{*/
/** brief Convert OpenCV matrix format to OpenGL texture format.*/
_GPUCV_EXPORT void cvgConvertCVMatrixFormatToGL(CvMat * _mat, GLuint & _internal_format, GLuint & _format, GLuint &_pixtype);
/** brief Convert OpenCV matrix format to OpenGL texture format.*/
_GPUCV_EXPORT void cvgConvertCVMatrixFormatToGL(int _type, GLuint & _internal_format, GLuint & _format, GLuint &_pixtype);

_GPUCV_EXPORT int cvgConvertGLFormattoCVMatrix(GLuint _pixtype, GLuint _channels);
/**@}*/

/** @}*/ //GPUCV_CONVERSION_GRP
//==============================================================================
//OpenCV format conversion to strings.
/** @addtogroup GPUCV_STRING_CONVERSION_GRP
*	@{
*/
#ifdef __cplusplus
/** brief Return OpenCV pixel type as a string.*/
_GPUCV_EXPORT std::string GetStrCVPixelType(const int _depth);
/** brief Return OpenCV pixel type as a string.*/
_GPUCV_EXPORT std::string GetStrCVPixelType(const IplImage * _ipl);
/** brief Return OpenCV image format as a string.*/
_GPUCV_EXPORT std::string GetStrCVTextureFormat(const IplImage * _ipl);
/** brief Return OpenCV image format as a string.*/
_GPUCV_EXPORT std::string GetStrCVTextureFormat(const GLuint format, const char * seq=NULL);
#endif //__cplusplus
/**@}*/
//==============================================================================
}//namespace GCV
#endif

