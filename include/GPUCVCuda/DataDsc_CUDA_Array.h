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
#ifndef __GPUCV_DataDsc_CUDA_Array_H
#define __GPUCV_DataDsc_CUDA_Array_H

#include <GPUCVCuda/DataDsc_CUDA_Base.h>
#if _GPUCV_COMPILE_CUDA
#include <GPUCVTexture/DataContainer.h>

namespace GCV{
/**	\brief DataDsc_CUDA_Array is the base class to describe CUDA array objects.
*	\author Yannick Allusse
*/
class _GPUCV_CUDA_EXPORT DataDsc_CUDA_Array
	:public DataDsc_CUDA_Base
{
protected:
	size_t						m_pitch;				//!< See cudaMallocPitch().
	cudaChannelFormatDesc	*	m_textureChannelDesc;
public:
	/** \brief Default constructor. */
	_GPUCV_CUDA_INLINE
		DataDsc_CUDA_Array(void);
	/** \brief Default destructor. */
	_GPUCV_CUDA_INLINE virtual
		~DataDsc_CUDA_Array(void);

	/** \brief Redefinition of exception logging function */
	virtual std::string LogException(void)const;


	//Redefinition of data parameters manipulation

	//Redefinition of data manipulation
	_GPUCV_CUDA_INLINE virtual void Free();

	//Redefinition of DataDsc_Base interaction with other objects
	//access format, parameter are OpenGL enum format descriptor.
	virtual bool	CopyTo(DataDsc_Base* _destination, bool _datatransfer=true);
	virtual bool	CopyFrom(DataDsc_Base* _source, bool _datatransfer=true);
	virtual	DataDsc_Base * Clone(DataDsc_CUDA_Array * _src, bool _datatransfer=true);
	virtual	DataDsc_Base * CloneToNew(bool _datatransfer=true);

	virtual std::ostringstream & operator << (std::ostringstream & _stream)const;
	void PostProcessUpdate(void);
	void PreProcessUpdate(void);

	//local functions:
	_GPUCV_CUDA_INLINE 
		void _SetCudaChannelFormatDesc(cudaChannelFormatDesc * _channelDesc);
	_GPUCV_CUDA_INLINE 
		cudaChannelFormatDesc* _GetCudaChannelFormatDesc(void);

	/** \brief Allocate a CUDA device buffer.
	\param _datatype => Type of data to allocate using CUDA type descriptors [CU_AD_FORMAT_UNSIGNED_INT8 | CU_AD_FORMAT_UNSIGNED_INT16 | CU_AD_FORMAT_FLOAT ...]
	*/
	void _AllocateDevice(unsigned int _datatype);

	/** \brief Create a CUDA channel descriptor.
	\param TType => Type of data to allocate using "C" data type [char|uchar|int|float...]
	\return A new channel descriptor.
	*/		

	template <typename TType>
	cudaChannelFormatDesc CreateCudaChannelDesc()
	{
		char x, y, z, w;
		x=y=z=w=0;
		enum cudaChannelFormatKind ChannelFormat; 
		char ChannelSize = sizeof(TType)*8;
		switch(m_nChannels)
		{

		case 3: SG_Assert(m_nChannels!=3, "CUDA array can not use 3 channels textures."); break;
		case 4:	w = ChannelSize;
			z = ChannelSize;
		case 2:	y = ChannelSize;
		case 1:	x = ChannelSize;break;
		default:
			GCU_CLASS_ASSERT(0, "Unknown switch value(channel number).");
		}
		if(typeid(TType) == typeid(char) || typeid(TType) == typeid(short)|| typeid(TType) == typeid(int))
			ChannelFormat = cudaChannelFormatKindSigned;
		else if(typeid(TType) == typeid(unsigned char) || typeid(TType) == typeid(unsigned short) || typeid(TType) == typeid(unsigned int))
			ChannelFormat = cudaChannelFormatKindUnsigned;
		else if(typeid(TType) == typeid(double) || typeid(TType) == typeid(float))
			ChannelFormat = cudaChannelFormatKindFloat;
		else
		{
			GCU_CLASS_ASSERT(0, "Unknown switch value(Channel format).");
		}
		if(!_GetCudaChannelFormatDesc())
			_SetCudaChannelFormatDesc(new cudaChannelFormatDesc);

		//cudaChannelFormatDesc & TmpDesc = gcudaCreateChannelDesc(x, y, z, w, ChannelFormat);
		gcudaCopyChannelDesc(cudaCreateChannelDesc(x, y, z, w, ChannelFormat), _GetCudaChannelFormatDesc()); 
		return *_GetCudaChannelFormatDesc();
	}


	/** \brief Allocate a CUDA device buffer(template function).
	\param TType => Type of data to allocate using "C" data type [char|uchar|int|float...]
	\return True if data allocation was succesful.
	*/
	template <typename TType>
	bool _AllocateDataPtr(void)
	{	
		CLASS_FCT_SET_NAME_TPL(TType,"_AllocateDataPtr");
		unsigned int TotalMem=0;
		unsigned int FreeMem=0;
#if 0//ndef LINUX
		if(GetParent() && GetParent()->GetOption(DataContainer::DBG_IMG_MEMORY))
		{
			cuMemGetInfo(&FreeMem,&TotalMem);
			CLASS_NOTICE("");
			GPUCV_NOTICE("BEFORE=>Cuda free memory:" << FreeMem << "(" << (double)FreeMem/TotalMem*100. <<"%)");
			if(FreeMem==0 || TotalMem==0)
			{
				GPUCV_NOTICE("We don't have free memory..??");
			}
		}
#endif

		//				texture<unsigned char, 1,cudaReadModeElementType> texTest;
		//CreateCudaChannelDesc<TType>();//8,0,0,0,cudaChannelFormatKindUnsigned);//8,8,8,,cudaChannelFormatKindUnsigned);
		//cudaCreateChannelDesc<
		if(!_GetCudaChannelFormatDesc())
		{
#if 0
			_SetCudaChannelFormatDesc(new cudaChannelFormatDesc);
			gcudaCopyChannelDesc(CreateCudaChannelDesc<TType>(), _GetCudaChannelFormatDesc());
#else
			CreateCudaChannelDesc<TType>();
#endif
			//m_textureChannelDesc = &gcudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		}

#if _GPUCV_CUDA_USE_SINGLE_CHANNEL_IMG
		CLASS_DEBUG("cudaMallocArray(m_textureArrayPtr, desc,"<< _GetWidth()*_GetNChannels() <<","<< _GetHeight()<<")");
		gcudaMallocArray(&m_data.m_textureArrayPtr, m_textureChannelDesc, _GetWidth()*_GetNChannels(), _GetHeight());
		//cudaMallocPitch((&m_data.m_textureArrayPtr

#else
		CLASS_DEBUG("cudaMallocArray(m_textureArrayPtr, desc,"<< _GetWidth() <<","<< _GetHeight()<<")");
		gcudaMallocArray(&m_data.m_textureArrayPtr, m_textureChannelDesc, _GetWidth(), _GetHeight());
#endif
		//retrieve real pointer to CUDA dest, the previous one was a local object that will be destroyed when leaving the function.
		//			GCU_CUDA_SAFE_CALL(cudaGetChannelDesc(m_textureChannelDesc, m_data.m_textureArrayPtr));

		gcudaCheckError("Error allocating m_textureArrayPtr.");
		GCU_CLASS_ASSERT(m_data.m_textureArrayPtr, "_AllocateDataPtr()=> Could not allocate pointer");
		CLASS_DEBUG("cudaMemset(m_textureArrayPtr, 0,"<< m_memSize<<")");
#if 0// _DEBUG
		gcudaMemset(m_data.m_textureArrayPtr, 0, m_memSize);
#endif

		//gcudaCheckError("DataDsc_CUDA_Array::_AllocateDataPtr() execution failed\n");
		//how to get error code from CUDA..?
#if 0//LINUX
		if(GetParent() && GetParent()->GetOption(DataContainer::DBG_IMG_MEMORY))
		{
			cuMemGetInfo(&FreeMem,&TotalMem);
			CLASS_NOTICE("AFTER=>Cuda free memory: " << FreeMem << " (" << (double)FreeMem/TotalMem*100. <<"% )");
		}
#endif
		return true;
	}

	/** \brief Get CUDA memory pitch value.
	\return CUDA memory pitch value.
	*/
	_GPUCV_CUDA_INLINE 
		size_t _GetPitch(void)const;
};
}//namespace GCV
#endif//_GPUCV_COMPILE_CUDA
#endif//__GPUCV_DataDsc_CUDA_Array_H
