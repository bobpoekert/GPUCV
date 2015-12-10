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
#ifndef __GPUCV_TEXTURE_DATADSC_CPU_H
#define __GPUCV_TEXTURE_DATADSC_CPU_H

#include <GPUCVTexture/config.h>
#include <GPUCVTexture/DataDsc_base.h>

namespace GCV{

/**	\brief DataDsc_CPU is the CPU implementation of DataDsc_Base class. It is used to store and manipulate data in central memory.
*	\sa DataDsc_Base
*	\author Yannick Allusse
*/
class _GPUCV_TEXTURE_EXPORT DataDsc_CPU
	:public DataDsc_Base
{
protected:
	PIXEL_STORAGE_TYPE	**m_pixels;		//!< Pointer to texture data in RAM.
public:
	/** \brief Default constructor. */
	__GPUCV_INLINE
		DataDsc_CPU(void);
	/** \brief Child Constructor use to set the name of the child class. */
	__GPUCV_INLINE
		DataDsc_CPU(const std::string _ClassName);

	/** \brief Default destructor. */
	__GPUCV_INLINE virtual
		~DataDsc_CPU(void);

	//Redefinition of global parameters manipulation functions
	//...
	//Redefinition of data parameters manipulation
	//...
	//Redefinition of data manipulation
	/**	\brief Allocate data locally in central memory using the predefined data parameters.
	*	\sa DataDsc_Base::Allocate(), IsAllocated(), Free(), HaveData().
	*/
	__GPUCV_INLINE virtual void Allocate(void);
	/**	\brief Check if data has been allocated and return true, else return false.
	*	\sa Allocate(), Free(), HaveData().
	*/
	__GPUCV_INLINE virtual bool IsAllocated(void)const;

	/**	\brief Free allocated data, and also set data flag to false.
	*	\sa Allocate(), IsAllocated(), HaveData().
	*/
	__GPUCV_INLINE virtual void Free();

	//Redefinition of DataDsc_Base interaction with other objects
	virtual bool CopyTo(DataDsc_Base* _destination, bool _datatransfer=true);
	virtual bool CopyFrom(DataDsc_Base* _source, bool _datatransfer=true);
	virtual	DataDsc_CPU * Clone(DataDsc_CPU * _src, bool _datatransfer=true);
	virtual	DataDsc_Base * CloneToNew(bool _datatransfer=true);



	//local functions
	__GPUCV_INLINE virtual PIXEL_STORAGE_TYPE** _GetPixelsData();
	__GPUCV_INLINE virtual void _SetPixelsData(PIXEL_STORAGE_TYPE ** _data);


	virtual std::ostringstream & operator << (std::ostringstream & _stream)const;
};

}//namespace GCV
#endif//__GPUCV_TEXTURE_DATADSC_CPU_H
