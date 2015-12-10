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
#ifndef __GPUCV_DATADSC_CVMAT_H
#define __GPUCV_DATADSC_CVMAT_H
#include <GPUCV/config.h>
#include <GPUCVTexture/DataDsc_CPU.h>

namespace GCV{

/**	\brief DataDsc_CvMat class describes the CvMat data storage into central memory, it is used to manage transfer with other data storages like DataDsc_GLTex, DataDsc_GLBuff, DataDsc_GLCUDA...
*	\sa CL_Profiler, DataDsc_Base
*	\author Yannick Allusse
*/
class _GPUCV_EXPORT DataDsc_CvMat
	:public DataDsc_CPU
{
protected:
	CvMat			* m_CvMat;	//! Pointer to corresponding(linked) CvMat.
	CvArr			** m_CvArr;	//! Pointer to corresponding(linked) CvMat pointer.
public:
	__GPUCV_INLINE
		DataDsc_CvMat(void);
	__GPUCV_INLINE virtual
		~DataDsc_CvMat(void);

	//Redefinition of global parameters manipulation functions

	//Redefinition of data parameters manipulation

	//Redefinition of data manipulation

	//Redefinition of DataDsc_Base interaction with other objects

	//base one from DataDsc_Base
	//access format, parameter are OpenGL enum format descriptor.
	virtual bool CopyTo(DataDsc_Base* _destination, bool _datatransfer=true);
	virtual bool CopyFrom(DataDsc_Base* _source, bool _datatransfer=true);
	__GPUCV_INLINE virtual void Allocate(void);		
	__GPUCV_INLINE virtual bool IsAllocated(void)const;
	__GPUCV_INLINE virtual void Free();
	__GPUCV_INLINE virtual void *GetNewParentID()const;

	//local functions
	__GPUCV_INLINE void		_SetCvMat(CvMat ** _mat);
	__GPUCV_INLINE CvMat*	_GetCvMat()const;
	__GPUCV_INLINE CvMat**	_GetCvMatPtr()const;
	virtual	DataDsc_CvMat * Clone(DataDsc_CvMat * _src, bool _datatransfer=true);
	virtual	DataDsc_Base *	CloneToNew(bool _datatransfer=true);
	DataDsc_CvMat*	CloneCvMat(const CvMat* _source, bool _datatransfer=true);
	__GPUCV_INLINE virtual void		ConvertPixelFormat_GLToLocal(const GLuint _pixelFormat);
	__GPUCV_INLINE virtual GLuint	ConvertPixelFormat_LocalToGL(void);
	__GPUCV_INLINE virtual void		ConvertPixelType_GLToLocal(const GLuint _pixelType);
	__GPUCV_INLINE virtual GLuint	ConvertPixelType_LocalToGL(void);

	virtual std::ostringstream & operator << (std::ostringstream & _stream)const;
};
}//namespace GCV
#endif//__GPUCV_DATADSC_CVMAT_H
