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

#ifndef __GPUCV_CORE_HANDLE_MANAGER_H
#define __GPUCV_CORE_HANDLE_MANAGER_H


#include <GPUCVCore/config.h>
#include <GPUCVCore/GpuShaderObject.h>

namespace GCV{
/** @addtogroup GPUCV_SHADER_GRP
*  @{
*/
class ShaderObject;

/**
*	\brief Manage a collection of cvgHandle
*	\author Erwan Guehenneux
*	\author Jean-Philippe Farrugia
*
*  This class centralize all cvgHandles and sort them to optimize 
*  shader loading and release.
*/
class GpuHandleManager
{

	/**
	*	\brief store OpenGL shaders handles
	*
	*/
	class CvgHandle
	{
	public : 
		GLhandleARB     shader_handle;
		GLhandleARB     vertex_handle;
		GLhandleARB     fragment_handle;
		ShaderObject  *linked_object;

		/*!
		*	\brief default constructor
		*	\param sh -> GLSL shader program header
		*	\param sv -> GLSL vertex program header
		*	\param sf -> GLSL fragment program header
		*	\param so -> pointer to the corresponding cvgShaderObject wich use those headers
		*/
		CvgHandle(GLhandleARB sh, GLhandleARB sv, GLhandleARB sf, ShaderObject * so);
		/*!
		*	\brief destructor
		*/
		~CvgHandle();
	};

private : 
	CvgHandle ** stack;
	int      current_pos;

public:
	/*! \brief default constructor. */
	GpuHandleManager();
	/*! \brief Destructor. */
	~GpuHandleManager();

	/*!
	*	\brief get the shader handle of one CvgHandle
	*	\param pos -> index array of the CvgHandle
	*	\return GLhandleARB -> OpenGL shader handle
	*/
	GLhandleARB GetHandle(int pos);

	/*!
	*	\brief replace one CvgHandle in top of the list
	*	\param pos -> index array of the cvgHandle to replace
	*	\return GLhandleARB -> shader program handle of the replaced CvgHandle
	*/
	GLhandleARB SetFirst(int pos);

	/*!
	*	\brief add a new CvgHandle ton the array and delete the last used
	*	\param sh -> OpenGL header of shader program
	*	\param vh -> OpenGL header of vertex shader
	*	\param fh -> OpenGL header of fragment shader.
	*	\param so -> pointer to the corresponding cvgShaderObject which use those headers
	*/
	void cvgPush(GLhandleARB sh, GLhandleARB vh, GLhandleARB fh, ShaderObject * so);

	/*!
	*	\brief remove one CvgHandle
	*	\param pos -> index array of the cvgHandle to replace
	*/
	void RemoveHandle(int pos);
};
/** @}*/ //GPUCV_SHADER_GRP
}//namespace GCV	 
#endif
