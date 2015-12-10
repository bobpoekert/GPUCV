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
\file GLContext.h
\author Yannick Allusse
\author Jean-Philippe Farrugia
*/

#ifndef __GPUCV_HARDWARE_GLCONTEXT_H
#define __GPUCV_HARDWARE_GLCONTEXT_H

#include <GPUCVHardware/ToolsGL.h>
#include <GPUCVHardware/Tools.h>
#include <iostream>


namespace GCV{

/*!
*	\brief Initialize an OpenGL view port to render the shader to.
*	\param x_min        ->	x_min of the image to render
*	\param x_max		->  x_max of the image to render
*	\param y_min		->	y_min of the image to render
*	\param y_max		->	y_max of the image to render
*	\todo Add more parameter to allow ROI in DataContainer.
*	\sa InitGLViewPerspective()
*/
_GPUCV_HARDWARE_EXPORT void InitGLView(int x_min, int x_max, int y_min, int y_max);
_GPUCV_HARDWARE_EXPORT void InitGLView(TextSize<GLsizei> &_size);
_GPUCV_HARDWARE_EXPORT void InitGLView(int x_max, int y_max);

/*!
*	\brief Initialize an OpenGl view port to render the shader to.
*	\param width        -> width of the image to render
*	\param height       -> height of the image to render
*	\param _near		-> Near clipping
*	\param _far			-> far clipping
*	\sa InitGLView().
*	\todo Further test.
*/
_GPUCV_HARDWARE_EXPORT void InitGLViewPerspective(int width, int height,  float _near=0.01, float _far=1000.);
_GPUCV_HARDWARE_EXPORT void InitGLViewPerspective(TextSize<GLsizei> & _size, float _near=0.01, float _far=1000.);

/**
\brief Class used to manipulate current context and encapsulate some OpenGL calls.
\todo Add function to switch between different contexts, and more options.
\todo Add LINUX support.
*/
class _GPUCV_HARDWARE_EXPORT_CLASS GLContext
{
protected:
	static std::vector <GLContext*> m_ContextVector;	//! Stack containing previous context. Use for Pop/Push
	static GLContext*				m_CurrentContext;	//! Actual OpenGl Context.
	static bool						m_MultiContext;		//! Flag to set the application as a multi OpenGL context application.


#ifdef _WINDOWS
	HDC		m_hDC;
	HGLRC	m_hRC;

	//HDC   hDCtmp;
	//HGLRC hRCtmp;
#endif
public:
	static void PushCurrentGLContext();
	static void PopCurrentGLContext();
	static void SetMultiContext(bool _multi);
	static GLContext* GetActiveContext();

public :
	/*! \brief  Default constructor */
	__GPUCV_INLINE
		GLContext();

#ifdef _WINDOWS
	/*! \brief  Constructor
	*	\param hRC => OpenGL device context, given by wglGetCurrentContext().
	*	\param hDC => OpenGL device context, given by wglGetCurrentDC().
	*/
	__GPUCV_INLINE
		GLContext(HDC hDC , HGLRC hRC);
#endif

	/**
	*	\brief  Default destructor
	*/
	__GPUCV_INLINE
		~GLContext();

	/**
	*	\brief  store all environment information before doing filter processing.
	*	\return none
	*/
	__GPUCV_INLINE
		void PushAttribs();

	/**
	*	\brief  Put environment information back to retrieve the same drawing as before filter processing.
	*	\return none
	*/
	__GPUCV_INLINE
		void PopAttribs();

	__GPUCV_INLINE
		void MakeCurrent();

};


/**
*	\brief get one unique occurrence of one GLContext
*	\return GLContext* -> pointer to one unique GLContext
*	\todo multiple GPU could mean multiple GLContext...
*/
_GPUCV_HARDWARE_EXPORT __GPUCV_INLINE GLContext* getGLContext();

/*!
\brief MACRO to test OpenGL errors.
If defined, this macro will test OpenGL error, write a message to the active output giving {FILE/LINE/Error Code/Error message} about the error.
\sa ShowOpenGLError().
*/
#define _GPUCV_GL_ERROR_TEST()\
	if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_CHECK))\
	ShowOpenGLError((char*)__FILE__, __LINE__);

#if _GPUCV_DEPRECATED
//Push/PopExceptionObj() generate some errors in multi-threaded applications...do not use
#define _GPUCV_CLASS_GL_ERROR_TEST()\
{\
	if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_CHECK))\
{\
	GetGpuCVSettings()->PushExceptionObj(this);\
	ShowOpenGLError((char*)__FILE__, __LINE__);\
	GetGpuCVSettings()->PopExceptionObj();\
}\
}
#else

#define _GPUCV_CLASS_GL_ERROR_TEST()\
	if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_GL_ERROR_CHECK))\
	{\
		ShowOpenGLError((char*)__FILE__, __LINE__);\
	}

#endif

#define SetThread()GLContext::PushCurrentGLContext();
#define UnsetThread()GLContext::PopCurrentGLContext();
/**
*	\brief Check if any openGL errors occurred and show them to the console.
*	\return true if error(s) occurred.
*/
_GPUCV_HARDWARE_EXPORT bool ShowOpenGLError(char * File, long Line);

/** @defgroup GPUCV_STRING_CONVERSION_GRP Functions to convert types into strings.
	@ingroup GPUCV_SDK_GRP
@{*/
/**
*	\brief Convert a GLTexture format GLuint into a string. Useful to show debug information.
*	\return The converted string.
*/
_GPUCV_HARDWARE_EXPORT std::string GetStrGLTextureFormat(const GLuint format);

/**
*	\brief Convert a GLTexture internal format GLuint into a string. Useful to show debug information.
*	\return The converted string.
*	\sa Check http://www.opengl.org/resources/code/samples/sig99/advanced99/notes/node51.html for a list of internal format supported.
*/
_GPUCV_HARDWARE_EXPORT std::string GetStrGLInternalTextureFormat(const GLuint format);

/**
*	\brief Convert a GLTexture pixel type GLuint into a string. Useful to show debug information.
*	\return The converted string.
*/
_GPUCV_HARDWARE_EXPORT std::string GetStrGLTexturePixelType(const GLuint format);

/**
*	\brief Convert a GLTexture type into a string. Useful to show debug information.
*	\return The converted string.
*/
_GPUCV_HARDWARE_EXPORT std::string GetStrGLTextureType(const GLuint type);

/**
*	\brief Convert a GLTexture Color Attachment type GLuint into a string. Useful to show debug information.
*	\return The converted string.
*/
_GPUCV_HARDWARE_EXPORT std::string GetStrGLColorAttachment(const GLuint format);

/**
*	\brief Convert a string into an OpenGL geometry type such as GL_LINE....
*	\param _typeStr => geometry type as string.
*	\return The corresponding geometry type.
*/
_GPUCV_HARDWARE_EXPORT GLuint GetGeometryTypeFromStr(const std::string & _typeStr);

/**
*	\brief Convert an Open geometry type such as GL_LINE into a string.
*	\param _type => geometry type as GLuint.
*	\return The corresponding geometry type string.
*/
_GPUCV_HARDWARE_EXPORT const char * GetGeometryStrFromType(const GLuint & _type);

/** @}*/ //GPUCV_STRING_CONVERSION_GRP

/** @defgroup GPUCV_CONVERSION_GRP Type and format conversion.
	@ingroup GPUCV_SDK_GRP
*	@{
*	\brief Return the pixel size corresponding to a pixel type.
*	\return The pixel size.
*/
_GPUCV_HARDWARE_EXPORT GLenum GetGLTypeSize(const GLenum type);

/**
*	\brief Return the number of component of a GLTexture format.
*	\return The component number.
*/
_GPUCV_HARDWARE_EXPORT GLenum GetGLNbrComponent(const GLenum format);

/** @}*/

}//namespace GCV
#endif//#define GLCONTEXT_H
