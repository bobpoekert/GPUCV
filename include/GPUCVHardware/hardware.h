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



#ifndef __GPUCV_HARDWARE_H
#define __GPUCV_HARDWARE_H


#include <GPUCVHardware/config.h>
#include <SugoiTools/config.h>


#ifdef __cplusplus

#include <GPUCVHardware/GLContext.h>
#include <GPUCVHardware/GenericGPU.h>
#include <GPUCVHardware/Profiling.h>


namespace GCV{

/*!
*	\brief get and store some information about hardware
*	\author Erwan Guehenneux
*	\author Jean-Philippe Farrugia
*	\author Yannick Allusse
*
*  Get some information about hardware compatibility with shaders and library in general.
* \sa We suggest you to download the <a href="http://www.realtech-vr.com/glview/" target=blank>OpenGL extension viewer from Realtech-VR</a>. It includes all
* the OpenGL extensions and there compatibilities with hardware constructor.
* \todo Create a GPU manager to manage multiple GPU hardware.
*/
class _GPUCV_HARDWARE_EXPORT_CLASS HardwareProfile
{
private :
	std::vector<GenericGPU*> m_GPUVector;
	GenericGPU	*	m_ProcessingGPU;	//!< Pointer to the processing GPU of the system. Will be used in further release[MORE_HERE].
	GenericGPU	*	m_RenderingGPU;		//!< Pointer to the rendering GPU of the system. Will be used in further release[MORE_HERE].
	GenericGPU	*	m_MainGPU;			//!< Pointer to the main GPU of the system.
	GLuint			m_dfltTextType;		//!< Store the default texture type.

public :
	/*!
	*	\brief default constructor.
	*/
		HardwareProfile();
	/*!
	*	\brief default destructor.
	*/
		__GPUCV_INLINE
			~HardwareProfile();

	
		std::vector<GenericGPU*>::iterator GetFirstGPUIter();
		std::vector<GenericGPU*>::iterator GetLastGPUIter();
		size_t GetGPUNbr()const;

	/*!
	*	\brief get the GLSL compatibility.
	*	\return int -> glsl_profile compatibility
	*/
	__GPUCV_INLINE
		int		GetGLSLProfile();

	/*!
	*	\brief Return the default texture type of the system
	*	\return returns GL_TEXTURE_2D is hardware is compatible, else GL_TEXTURE_RECTANGLE_ARB.
	*/
	__GPUCV_INLINE
		int		GetTextType();

	/*!
	*	\brief Select the best texture type depending on hardware and the _type parameter.
	*	\param	_type => if 0, texture type is automatically selected. Else try to use given type if compatible. Type are GL_TEXTURE_2D and GL_TEXTURE_RECTANGLE.
	*	\sa _GPUCV_TEXTURE_FORCE_TEX_RECT, _GPUCV_TEXTURE_FORCE_TEX_NPO2
	*/
	void SetTextType(GLuint _type=0);

	/*!
	*	\brief returns true if main GPU is compatible with FBO extension, false else.
	*	\return bool -> hardware FBO compatibility
	*/
	__GPUCV_INLINE
		bool IsFBOCompatible();

	/*!
	*	\brief returns true if main GPU is compatible with the corresponding GLSL profile, false else.
	*	\return bool -> hardware FBO compatibility
	*/
	__GPUCV_INLINE
		bool IsCompatible(int profile_needed);



	/*! \return the processing GPU pointer.*/
	__GPUCV_INLINE
		GenericGPU	*	GetProcessingGPU();
	/*! \return the rendering GPU pointer.*/
	__GPUCV_INLINE
		GenericGPU	*	GetRenderingGPU();
	/*! \return the main (active) GPU pointer.*/
	__GPUCV_INLINE
		GenericGPU	*	GetMainGPU();

	/*! \brief Set the processing GPU pointer.*/
	__GPUCV_INLINE
		void	SetProcessingGPU(GenericGPU	*_gpu);
	/*! \brief Set the rendering GPU pointer.*/
	__GPUCV_INLINE
		void	SetRenderingGPU(GenericGPU	*_gpu);
	/*! \brief Set main (active) GPU pointer.*/
	__GPUCV_INLINE
		void	SetMainGPU(GenericGPU	*_gpu);

	__GPUCV_INLINE
		void	AddGPU(GenericGPU	*_gpu);

	/*!
	*	\brief Scan the system hardware to create corresponding GPU(s).
	*	\return The number of GPU(s) found in the system.
	*/
	int	CreateGPU();
};

/*!
*	\brief get one unique occurrence of one HardwareProfile
*	\return HardwareProfile -> pointer to one unique HardwareProfile
*/
_GPUCV_HARDWARE_EXPORT __GPUCV_INLINE HardwareProfile* GetHardProfile();
_GPUCV_HARDWARE_EXPORT __GPUCV_INLINE GenericGPU* ProcessingGPU();
_GPUCV_HARDWARE_EXPORT __GPUCV_INLINE GenericGPU* RenderingGPU();
_GPUCV_HARDWARE_EXPORT __GPUCV_INLINE GenericGPU* MainGPU();

#endif//c++

}//namespace GCV
#endif
