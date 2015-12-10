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



#ifndef __GPUCV_TEXTURE_DATADSC_CUDA_H
#define __GPUCV_TEXTURE_DATADSC_CUDA_H


#include <GPUCVCuda/DataDsc_CUDA_Array.h>
#include <GPUCVTexture/DataDsc_GLBase.h>
#include <GPUCVTexture/DataDsc_GLBuff.h>
#if _GPUCV_COMPILE_CUDA
namespace GCV{

/**	\brief DataDsc_CUDA_Buffer is the class to describe CUDA device memory objects.
*	\author Yannick Allusse
*/
class _GPUCV_CUDA_EXPORT DataDsc_CUDA_Buffer
	:public DataDsc_CUDA_Base
{
public:
	enum CUDA_DEVICE_DATA_TYPE{
		CUDA_NO_TYPE,
		CUDA_GL_BUFFER,
		CUDA_DX_BUFFER,
		CUDA_CPU_BUFFER,
		CUDA_TEXTURE_ARRAY
	};

protected:
	size_t					m_pitch;		//!< See cudaMallocPitch().
	bool					m_glBufferMapped;	//!< Specify if the object is mapped to a buffer.

#if CUDA_VERSION > 2300
	struct cudaGraphicsResource*	m_pGLResource;
#endif
public:
	/** \brief Default constructor. */
	_GPUCV_CUDA_INLINE
		DataDsc_CUDA_Buffer(void);
	/** \brief Default destructor. */
	_GPUCV_CUDA_INLINE virtual
		~DataDsc_CUDA_Buffer(void);	

	/** \brief Redefinition of exception logging function */
	virtual std::string LogException(void)const;

	//Redefinition of data manipulation
	_GPUCV_CUDA_INLINE virtual void Free();

	//Redefinition of DataDsc_Base interaction with other objects
	//access format, parameter are OpenGL enum format descriptor.
	virtual bool	CopyTo(DataDsc_Base* _destination, bool _datatransfer=true);
	virtual bool	CopyFrom(DataDsc_Base* _source, bool _datatransfer=true);
	virtual	DataDsc_Base * Clone(DataDsc_CUDA_Buffer * _src, bool _datatransfer=true);
	virtual	DataDsc_Base * CloneToNew(bool _datatransfer=true);


	void PostProcessUpdate(void);
	void PreProcessUpdate(void);

	virtual std::ostringstream & operator << (std::ostringstream & _stream)const;

	//local functions:
	/** \brief Allocate a CUDA device buffer.
	\param _datatype => Type of data to allocate using CUDA type descriptors [CU_AD_FORMAT_UNSIGNED_INT8 | CU_AD_FORMAT_UNSIGNED_INT16 | CU_AD_FORMAT_FLOAT ...]
	*/
	void _AllocateDevice(unsigned int _datatype);

	/** \brief Allocate a CUDA device buffer(template function).
	\param TType => Type of data to allocate using "C" data type [char|uchar|int|float...]
	\return True if data allocation was succesful.
	*/
	template <typename TType>
	bool _AllocateDataPtr(void)
	{	
		CLASS_FCT_SET_NAME_TPL(TType,"_AllocateDataPtr");

#ifdef _WINDOWS //we always map an OpenGL buffer with a CUDA buff
		if(GetAutoMapGLBuff()==true)
		{
			CLASS_ASSERT(GetParent(), "No parent");
			DataDsc_GLBuff  * DD_GL = GetParent()->GetDataDsc<DataDsc_GLBuff>();
			//DataDsc_GLTex  * DD_GL = GetParent()->GetDataDsc<DataDsc_GLTex>();

			CLASS_ASSERT(DD_GL, "No OpenGL buffer found");
			if (_MapGLObject(DD_GL, cudaGraphicsMapFlagsWriteDiscard))
			{
				return true;
			}
		}
		//else perform local allocations
#endif//

		unsigned int TotalMem=0;
		unsigned int FreeMem=0;
#if 1
		if(GetParent() && GetParent()->GetOption(DataContainer::DBG_IMG_MEMORY))
		{
			cuMemGetInfo(&FreeMem,&TotalMem);
			CLASS_NOTICE("");
			GPUCV_NOTICE("BEFORE=>Cuda free memory: " << FreeMem << " (" << (double)FreeMem/TotalMem*100. <<"% )");
			if(FreeMem==0 || TotalMem==0)
			{
				GPUCV_NOTICE("We don't have free memory..??");
			}
		}
#endif
#if 0
		gcudaMalloc((void **)&m_data.m_deviceDataPtr, m_memSize);
#else
		gcudaMallocPitch((void **)&m_data.m_deviceDataPtr, &m_pitch, _GetWidth()*_GetNChannels()*sizeof(TType), _GetHeight());
#endif
		CLASS_DEBUG("gcudaMallocPitch(&m_deviceDataPtr, &m_pitch,"<< _GetWidth()*_GetNChannels()*sizeof(unsigned char) <<","<< _GetHeight() <<")");
		GCU_CLASS_ASSERT(m_data.m_deviceDataPtr, "_DeviceAllocate()=> Could not allocate pointer");
#if _DEBUG
		gcudaMemset(m_data.m_deviceDataPtr, 0, m_memSize);				
#endif
		CLASS_DEBUG("cudaMemset(m_deviceDataPtr, 0,"<< m_memSize<<")");
		gcudaCheckError("DataDsc_CUDA_Buffer::_DeviceAllocate() execution failed\n");
#if 0
		//how to get error code from CUDA..?
		if(GetParent() && GetParent()->GetOption(DataContainer::DBG_IMG_MEMORY))
		{
			cuMemGetInfo(&FreeMem,&TotalMem);
			GPUCV_NOTICE("AFTER=>Cuda free memory: " << FreeMem << " (" << (double)FreeMem/TotalMem*100. <<"% )");
		}
#endif
		return true;
	}

	/** \brief Get CUDA memory pitch value.
	\return CUDA memory pitch value.
	*/
	_GPUCV_CUDA_INLINE 
		size_t _GetPitch(void)const
	{
		return m_pitch;
	}

	/** Map an OpenGL object (BUFFER/TEXTURE) to write/read to/from it.
	* \param _pGLObj -> OpenGL object (DataDsc_GLTex or DataDscGL_Buff) to map with the current CUDA BUFFER.
	* \param _flag -> Must be cudaGraphicsMapFlagsReadOnly or cudaGraphicsMapFlagsWriteDiscard (cudaGraphicsMapFlagsNone is not allowed yet).
	* \return true if successful.
	*/
	bool _MapGLObject(DataDsc_GLBase *_pGLObj, enum cudaGraphicsMapFlags _flag);
	void _UnMapGLObject();//DataDsc_GLBase *_pGLObj);

};
}//namespace GCV
#endif//_GPUCV_COMPILE_CUDA
#endif//__GPUCV_TEXTURE_DATADSC_CUDA_H
