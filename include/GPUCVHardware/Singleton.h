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
#ifndef __GPUCV_HARDWARE_SINGLETON_H
#define __GPUCV_HARDWARE_SINGLETON_H

#include <GPUCVHardware/config.h>
#include <SugoiTools/singleton.h>
namespace GCV{


/** \brief CL_Singleton class is used to have only one main object registered for the given type.
*	CL_Singleton are generally used by manager to have only one main manager registered for a given task.
*	New manager can be registered in place of the existing one, this mechanism can be used by plug ins.
*	\author Yannick Allusse
*/
#define CL_Singleton SGE::CSingleton
#if 0
template<typename TType>
class CL_Singleton
{
protected:
	static TType	* m_registeredSingleton;	//! Pointer to the registered singleton.
public:
	/** \brief Constructor.
	*/
	__GPUCV_INLINE
		CL_Singleton()
	{}

	/** \brief Destructor.
	*/
	__GPUCV_INLINE
		~CL_Singleton()
	{}

	/** \brief Return registered singleton.
	*/
	static __GPUCV_INLINE
		TType * GetSingleton()
	{return m_registeredSingleton;}

	/** \brief Register a new object as the main singleton.
	*/
	static __GPUCV_INLINE
		void RegisterSingleton(TType * _newSingleton)
	{
		m_registeredSingleton = _newSingleton;
	}

	/** \brief Unregister the main singleton.
	*/
	static __GPUCV_INLINE
		void UnRegisterSingleton()
	{
		m_registeredSingleton = NULL;
	}

	/** \brief Delete the main singleton.
	*/
	static __GPUCV_INLINE
		void DeleteSingleton()
	{
		delete m_registeredSingleton;
		UnRegisterSingleton();
	}
};
#endif

}//namespace GCV
#endif
