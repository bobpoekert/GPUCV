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



#ifndef __GPUCV_DATADSC_CUDA_BASE_H
#define __GPUCV_DATADSC_CUDA_BASE_H

#include <GPUCVCuda/config.h>
#if _GPUCV_COMPILE_CUDA
#include <GPUCVTexture/DataDsc_base.h>
#include <GPUCVCuda/gcu_runtime_api_wrapper.h>

namespace GCV{
/**	\brief DataDsc_CUDA_Base is the base class to describe CUDA based data descriptors(DataDsc_Base).
*	\author Yannick Allusse
*/
class _GPUCV_CUDA_EXPORT DataDsc_CUDA_Base
	: virtual public DataDsc_Base
{
protected:
	unsigned int			m_cudaPixelType;
	union
	{
		cudaArray				* m_textureArrayPtr;
		unsigned char			* m_deviceDataPtr;
	}m_data;
	//! map CUDA ARRAY/BUFFER to OpenGL Image/Buffer when allocating cuda object, so we do not need to perform transfers between gl and cuda, default is true.
	_DECLARE_MEMBER(bool, AutoMapGLBuff);
public:

	/** \brief Default constructor. */
	_GPUCV_CUDA_INLINE
		DataDsc_CUDA_Base(void);
	/** \brief Default destructor. */
	_GPUCV_CUDA_INLINE virtual
		~DataDsc_CUDA_Base(void);

	/** \brief Redefinition of exception logging function */
	virtual 
		std::string LogException(void)const;

	virtual
		std::string PrintMemoryInformation(std::string text)const;

	//Redefinition of data parameters manipulation
	_GPUCV_CUDA_INLINE  virtual void SetFormat(const GLuint _pixelFormat,const GLuint _pixelType);

	//Redefinition of data manipulation
	_GPUCV_CUDA_INLINE virtual void Allocate(void);		
	_GPUCV_CUDA_INLINE virtual bool IsAllocated(void)const; 
	_GPUCV_CUDA_INLINE virtual void Free();


	//Redefinition of DataDsc_Base interaction with other objects
	virtual bool	CopyTo(DataDsc_Base* _destination, bool _datatransfer=true);
	virtual bool	CopyFrom(DataDsc_Base* _source, bool _datatransfer=true)=0;
	//virtual	DataDsc_Base * Clone(DataDsc_CUDA_Base * _src, bool _datatransfer=true)=0;
	virtual	DataDsc_Base * CloneToNew(bool _datatransfer=true)=0;

	//access format, parameter are OpenGL enum format descriptor.
	virtual void	ConvertPixelFormat_GLToLocal(const GLuint _pixelFormat);
	virtual GLuint	ConvertPixelFormat_LocalToGL(void);
	virtual void	ConvertPixelType_GLToLocal(const GLuint _pixelType);
	virtual GLuint	ConvertPixelType_LocalToGL(void);


	virtual void Flush(void);


	virtual std::ostringstream & operator << (std::ostringstream & _stream)const;

	//CUDA based option
	virtual void _AllocateDevice(unsigned int _datatype)=0;
	_GPUCV_CUDA_INLINE void * _GetDataPtr(void);
};
}//namespace GCV
#endif//_GPUCV_COMPILE_CUDA
#endif//__GPUCV_DATADSC_CUDA_BASE_H
