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
#ifndef __GPUCV_TEXTURE_DATADSC_BASE_H
#define __GPUCV_TEXTURE_DATADSC_BASE_H

#include <GPUCVTexture/TexCoord.h>

namespace GCV{

class DataContainer;


/**	\brief DataDsc_Base is the base class to describe a kind of data associated with its location (central memory, graphics memory...).
*	GpuCV can handle many kind of data (images, matrices...) with different kind of storage (int/float/..., number of channels, format) on different locations
*	(central memory, graphics card memory, ...), theses parameters are stored using DataDsc_Base or one of its child class. DataDsc_Base are designed to interact with each others,
*	they can copy/convert their parameters but also transfer their data from one location to another, see next section for details.
*	DataContainer acts as a container for all DataDsc_Base describing the same set of data that can be stored in different locations.
*	\sa CL_Profiler
*	\author Yannick Allusse
*/
class _GPUCV_TEXTURE_EXPORT DataDsc_Base
	:public CL_Profiler, public TextSize<GLsizei>
{
public:
	typedef GLvoid 				PIXEL_STORAGE_TYPE;	//!< Define the type used to store the Texture data in ram.
protected:
	std::string		m_dscType;			//!< should be GL_CPU, GL_GPU, CV_CPU, CUDA_GPU
	GLenum			m_glPixelFormat;	//!< Texture format.
	GLenum			m_glPixelType;		//!< Texture pixel type.
	GLenum			m_nChannels;		//!< Number of channels.
	GLenum			m_nChannelsBackup;	//!< backup of channels number, used we we reshape temporarily the DataDsc.
	DataContainer *	m_parent;			//!< Points to the texture container
	bool			m_haveData;			//!< Specify if this descriptor possess data.
	DataDsc_Base  *	m_lockedBy;			//!< Point to another DataDsc_Base they have locked this resources. It is used to make sure we are not accessing an object while there is a memory transfer with another object.
	DataDsc_Base  * m_lockedObj;		//!< Pointer to a DataDsc_Base that is being locked by current object, this is used for memory mapping between CUDA and OpenGL.
	unsigned int	m_dataSize;			//!< Contains the number of elements: width * height * channels
	unsigned int	m_memSize;			//!< Contains the amount of memory required to store the data: width * height * channels * sizeof(pixeltype)
	long			m_localMemoryAllocated;		//!< Total amount of memory allocated in current DataDsc* object.
public:
	static long		ms_totalMemoryAllocated;	//!< Total amount of memory allocated in all DataDsc* objects.

#if _GPUCV_PROFILE
	static SG_TRC::CL_CLASS_TRACER<CL_Profiler::TracerType>	* m_TransferClassTracer;
#endif
public:

	//-------------------------------------
	//
	//! \name Constructors/Destructors
	//@{
	//-------------------------------------
	/** \brief Default constructor. */
	__GPUCV_INLINE
		DataDsc_Base(void);

	/** \brief Child Constructor use to set the name of the child class. */
	__GPUCV_INLINE
		DataDsc_Base(const std::string _className);

	/** \brief Default destructor. */
	__GPUCV_INLINE virtual
		~DataDsc_Base(void);
	//@}
	//-------------------------------------
	//
	//! \name Integration with framework
	//@{
	//-------------------------------------
	/** \brief Write debugging informations to a stream buffer.
	*/
	virtual
		std::ostringstream & operator << (std::ostringstream & _stream)const;
	//@}
	//-------------------------------------
	//
	//! \name General parameters manipulation
	//@{
	//-------------------------------------

	/** \brief Set the parent container.
	*	\param _parent => DataContainer* that store and manipulate current DataDsc_Base.
	*/
	__GPUCV_INLINE virtual
		void	SetParent(DataContainer* _parent);

	/** \brief Get the parent container.
	*	\return DataContainer* that store and manipulate current DataDsc_Base.
	*/
	__GPUCV_INLINE
		DataContainer *	GetParent()const;
	__GPUCV_INLINE
		CL_Options::OPTION_TYPE GetParentOption(CL_Options::OPTION_TYPE _opt)const;

	/** \brief Return the ID to re-affect to parent data container.
	*   \note This is used by DataDsc such as DataDsc_IplImage or DataDsc_CvMat cause we are using their CvArr pointer as ID for the DataContainer.
	*	\return void * the ID to affect to the parent.
	*/
	__GPUCV_INLINE  virtual
		void *	GetNewParentID()const;

	/** \brief Return a string describing the class identity.
	*	\return A string with the ID of the parent DataContainer.
	*	\sa DataContainer::GetValStr().
	*/
	__GPUCV_INLINE virtual
		const std::string GetValStr()const;

	/** \brief Make a lock request to this object.
	*	\param _lockerObj => Pointer to the object that make the request.
	*	\return _lockerObj if lock is successful, else return NULL when object is already locked.
	*	\sa UnLock(), IsLocked(), m_lockedBy.
	*/
	__GPUCV_INLINE
		DataDsc_Base* Lock(DataDsc_Base* _lockerObj);

	/** \brief Return true if object is locked.
	*	\sa Lock(), UnLock(), m_lockedBy.
	*/
	__GPUCV_INLINE
		bool IsLocked()const;

	/** \brief Return locker object.
	*	\sa Lock(), UnLock(), m_lockedBy.
	*/
	__GPUCV_INLINE
	const DataDsc_Base* LockedBy()const;

	/** \brief Return locked object.
	*	\sa Lock(), UnLock(), m_lockedObj.
	*/
	__GPUCV_INLINE
	DataDsc_Base* GetLockedObj();
	const DataDsc_Base* GetLockedObj()const;

	/** \brief Set locked object.
	*	\sa Lock(), UnLock(), m_lockedObj.
	*/
	__GPUCV_INLINE
	const DataDsc_Base* SetLockedObj(DataDsc_Base* _obj);


	/** \brief Make an unlock request to this object.
	*	\note Only the original locker object can unlock it.
	*	\param _lockerObj => Pointer to the object that make the request.
	*	\return _lockerObj if lock is successful, else return NULL when object is locked by another object.
	*	\sa Lock(), IsLocked(), m_lockedBy.
	*/
	__GPUCV_INLINE
		DataDsc_Base* UnLock(DataDsc_Base*_lockerObj);

	/** \brief Return a string describing the class type.
	*	\return A string with the name of the class(or child class).
	*/
	__GPUCV_INLINE
		const std::string & GetDscType(void)const;

	static long GetTotalMemoryAllocated();
	long GetLocalMemoryAllocated()const;

	virtual
		std::string PrintMemoryInformation(std::string text)const;


	//@}
	//-------------------------------------
	//
	//! \name Data properties manipulation
	//@{
	//-------------------------------------
	/**
	*	\brief Set pixel format and type using OpenGL reference enum.
	*	Set the DataDsc_Base data format and type using the OpenGL enum as reference.
	*	It store values in m_glPixelFormat and m_glPixelType, retrieve the number of channels by calling GetGLNbrComponent().
	*	Then it calls functions ConvertPixelFormat_GLToLocal()/ConvertPixelType_GLToLocal() to perform local conversion.
	*	\param _pixelFormat => OpenGL format like GL_RGB, GL_RGBA, GL_LUMINANCE...
	*	\param _pixelType	=> OpenGL pixel type like GL_FLOAT, GL_INT, GL_BYTE...
	*	\sa GetPixelFormat(), GetPixelType(), GetGLNbrComponent(), ConvertPixelFormat_GLToLocal(), ConvertPixelType_GLToLocal().
	*/
	__GPUCV_INLINE virtual
		void SetFormat(const GLuint _pixelFormat,const GLuint _pixelType);

	/**
	*	\brief Get pixel format using OpenGL reference enum.
	*	\return Pixel format using OpenGL reference enum.
	*	\sa SetFormat() GetPixelType().
	*/
	__GPUCV_INLINE
		GLuint GetPixelFormat()const;

	/**
	*	\brief Get pixel type using OpenGL reference enum.
	*	\return Pixel type using OpenGL reference enum.
	*	\sa SetFormat() GetPixelFormat().
	*/
	__GPUCV_INLINE
		GLuint GetPixelType()const;

	__GPUCV_INLINE
		GLuint GetnChannels()const;


	/**
	*	\brief Convert the OpenGL reference enum format to local pixel format.
	*	Convert OpenGL enum to local DataDsc_Base class.
	*	\note This function must be redefined into each children class.
	*	\sa SetFormat(), GetPixelFormat(), ConvertPixelFormat_LocalToGL().
	*/
	virtual
		void	ConvertPixelFormat_GLToLocal(const GLuint _pixelFormat);
	/**
	*	\brief Convert the local pixel format to OpenGL reference enum.
	*	\note This function must be redefined into each children class.
	*	\sa SetFormat(), GetPixelFormat(), ConvertPixelFormat_GLToLocal().
	*/
	virtual
		GLuint	ConvertPixelFormat_LocalToGL(void);
	/**
	*	\brief Convert the OpenGL reference enum to local pixel type.
	*	\note This function must be redefined into each children class.
	*	\sa SetFormat(), GetPixelType(), ConvertPixelType_LocalToGL().
	*/
	virtual
		void	ConvertPixelType_GLToLocal(const GLuint _pixelType);
	/**
	*	\brief Convert the local pixel format to OpenGL reference enum.
	*	\note This function must be redefined into each children class.
	*	\sa SetFormat(), GetPixelType(), ConvertPixelType_GLToLocal().
	*/
	virtual
		GLuint	ConvertPixelType_LocalToGL(void);
#if _GPUCV_DEPRECATED
	/**	\brief Get the width of the data(from the parent DataContainer).
	*/
	__GPUCV_INLINE
		GLuint			_GetWidth();
	/**	\brief Get the height of the data(from the parent DataContainer).
	*/
	__GPUCV_INLINE
		GLuint			_GetHeight();
#endif
	/**	\brief Get the number of channels.
	*/
	__GPUCV_INLINE
		GLuint			_GetNChannels()const;
	/**	\brief Set the number of channels.
	*/
	__GPUCV_INLINE
		void			_SetNChannels(GLuint _channels);

	/**	\brief Redefine the size of the texture, flag m_needReload will be set to true.
	*	\param width -> new width.
	*	\param height	-> new height.
	*	\note A test is done to check that new size is compatible with current hardware maximum resolution(_GPUCV_TEXTURE_MAX_SIZE_X/_GPUCV_TEXTURE_MAX_SIZE_Y).
	*/
	void _SetSize(const unsigned int width, const unsigned int height);


	//@}
	//-------------------------------------
	//
	//! \name Data manipulation
	//@{
	//-------------------------------------
	/**	\brief Allocate data locally using the predefined data parameters.
	*	\sa IsAllocated(), Free(), HaveData().
	*/
	__GPUCV_INLINE virtual
		void	Allocate(void);

	/**	\brief Check if data has been allocated and return true, else return false.
	*	\sa Allocate(), Free(), HaveData().
	*/
	__GPUCV_INLINE virtual
		bool	IsAllocated(void)const {return false;};
	/**	\brief Free allocated data, and also set data flag to false.
	*	\sa Allocate(), IsAllocated(), HaveData().
	*/
	__GPUCV_INLINE virtual
		void	Free();

	/**	\brief Return the data flag.
	*	\sa SetDataFlag().
	*/
	__GPUCV_INLINE virtual
		bool	HaveData()const;

	/**	\brief Set the data flag value.
	*	The data flag specifies if data are present in the allocated memory.
	*	It is used by the copy mechanisms to find available data between all available DataDsc_Base objects.
	*	Special case: when current object is locked, it will also transmit the data flag to the locker object.
	*	Special case: when current object lock another object, it will also transmit the data flag to the locked object.
	*   \sa HaveData().
	*/
	__GPUCV_INLINE virtual
		void  SetDataFlag(bool _val);

	/**	\brief Calculate data size as width*height*channels.
	*/
	__GPUCV_INLINE
		GLuint			GetDataSize(void)const;
	/**	\brief Calculate memory size as width*height*channels*sizeof(elem).
	*/
	__GPUCV_INLINE
		GLuint			GetMemSize(void)const;

	//@}
	//-------------------------------------
	//
	//! \name Interaction with other objects
	//@{
	//-------------------------------------
	/**	\brief Copy and convert the data format from the source to the current object.
	*	\sa SetFormat().
	*/
	__GPUCV_INLINE virtual
		void TransferFormatFrom(const DataDsc_Base * _src);

	/** \sa cvReshape()
	*/
	__GPUCV_INLINE virtual
		void SetReshape(int new_cn, int new_rows=0);

	__GPUCV_INLINE virtual
		void UnsetReshape();

	/**	\brief Copy the local data to the destination object(only if the transfer methode is known).
	*	CopyTo() check if the destination object type is know, if not it returns false.
	*	If known, it allocates data on destination and process the copy if _datatransfer is true.
	*	\note If the destination object type is identical to current object, the Clone() function is called.
	*	\note If CopyTo() is not working(return false), the CopyFrom() may succeed.
	*	\sa CopyFrom(), TransferFormatFrom(), Clone().
	*/
	virtual
		bool CopyTo(DataDsc_Base* _destination, bool _datatransfer=true){return false;}

	/**	\brief Copy the destination data to local data(only if the transfer method is known).
	*	CopyFrom() check if the source object type is know, if not it returns false.
	*	If known, it allocates data locally and process the copy if _datatransfer is true.
	*	\note If the source object type is identical to current object, the Clone() function is called.
	*	\note If CopyFrom() is not working(return false), the CopyTo() may succeed.
	*	\sa CopyTo(), TransferFormatFrom(), Clone().
	*/
	virtual
		bool CopyFrom(DataDsc_Base* _source, bool _datatransfer=true){return false;}

#if _GPUCV_PROFILE
	/**	\brief Estimate transfer time to copy from current location to destination location, DataContainer::SMART_TRANSFER option must be enable.
	*	\return -1 if no known path is found between source and destination locations, or if transfer estimation is disabled.
	*	\return Time estimated to perform the copy between source and destination locations.
	*	\sa CopyTo(), CopyFromEstimation(), DataContainer::SMART_TRANSFER.
	*/
	virtual
		long int CopyToEstimation(DataDsc_Base* _destination, bool _datatransfer=true, SG_TRC::CL_TRACE_BASE_PARAMS*_params=NULL);

	/**	\brief Estimate transfer time to copy from current location to destination location, DataContainer::SMART_TRANSFER option must be enable.
	*	\return -1 if no known path is found between source and destination locations, or if transfer estimation is disabled.
	*	\return Time estimated to perform the copy between source and destination locations.
	*	\sa CopyFrom(), CopyToEstimation(), DataContainer::SMART_TRANSFER.
	*/
	virtual
		long int CopyFromEstimation(DataDsc_Base* _source, bool _datatransfer=true, SG_TRC::CL_TRACE_BASE_PARAMS*_params=NULL);
#endif
	/**	\brief Clone the source parameters and data to local object.
	*	Allocates data locally and process the copy if _datatransfer is true.
	*	\sa CopyTo(),  CopyFrom(), TransferFormatFrom(), CloneToNew().
	*/
	virtual
		DataDsc_Base * Clone(DataDsc_Base * _src, bool _datatransfer=true);

	/**	\brief Clone the local parameters and data to a new object.
	*	Create a new object of the same type and call Clone().
	*	\sa CopyTo(),  CopyFrom(), TransferFormatFrom(), Clone().
	*/
	virtual
		DataDsc_Base * CloneToNew(bool _datatransfer=true);

	/**	\brief Flush the processing pipeline corresponding to the image location( OpenGL/CUDA).
	*	Create a new object of the same type and call Clone().
	*	\sa CopyTo(),  CopyFrom(), TransferFormatFrom(), Clone().
	*/
	virtual void Flush(void);

	protected:
		void Log_DataAlloc(unsigned int _uiSize);
		void Log_DataFree(unsigned int _uiSize);
	//@}
	//-------------------------------------
#if _GPUCV_DEPRECATED
	/**	\brief Convert the current pointer into a DataContainer pointer(similar to GetParent()).
	*	\sa GetParent().
	*/
	__GPUCV_INLINE
		operator DataContainer*();

	/**	\brief Convert the current reference into a DataContainer reference(similar to GetParent()).
	*	\sa GetParent().
	*/
	__GPUCV_INLINE
		operator DataContainer&();
#endif
};

_GPUCV_TEXTURE_EXPORT std::ostringstream & operator << (std::ostringstream & _stream, const DataDsc_Base & TexDsc);



#if _GPUCV_DEBUG_MODE
#define TEXTUREDESC_COPY_START(IMG, DATA)\
	GPUCV_GET_FCT_NAME()+= "(datatransfer:";\
	GPUCV_GET_FCT_NAME()+= (DATA)?"1)":"0)";\
	CLASS_ASSERT(IMG, "TextureDsc_*::Copy*()=> no input object");\
	if((DataDsc_Base*)this==IMG)\
	return false;\
	CLASS_FCT_PROF_CREATE_START();
#else
#define TEXTUREDESC_COPY_START(IMG, DATA)\
	CLASS_ASSERT(IMG, "TextureDsc_*::Copy*()=> no input object");\
	if(this==IMG)\
	return false;
//CLASS_FCT_PROF_CREATE_START();
#endif

#if _GPUCV_PROFILE
#define GPUCV_DD_TRANSFER_BENCH(NAME, SRC_TYPE, DEST_TYPE, DD_SRC, DD_DEST, DATA_TRANSFER)\
	SG_TRC::CL_TRACE_BASE_PARAMS *_PROFILE_PARAMS = NULL;\
	std::string LocalFctName;\
	if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_PROFILING_TRANSFER)\
	&& (m_TransferClassTracer!=NULL))\
	{\
		LocalFctName = DD_SRC->GetClassName();\
		LocalFctName+= "-to-";\
		LocalFctName+= DD_DEST->GetClassName();\
		_PROFILE_PARAMS = new SG_TRC::CL_TRACE_BASE_PARAMS();\
		_PROFILE_PARAMS->AddChar("data", (DATA_TRANSFER)?"1":"0");\
		if(DD_SRC)\
		{\
			_PROFILE_PARAMS->AddParam("width", DD_SRC->_GetWidth());\
			_PROFILE_PARAMS->AddParam("height", DD_SRC->_GetHeight());\
			_PROFILE_PARAMS->AddParam("channels", DD_SRC->GetnChannels());\
			_PROFILE_PARAMS->AddParam("format", GetStrGLInternalTextureFormat(DD_SRC->GetPixelFormat()));\
		}\
		if(true)glFinish();\
	}\
	SG_TRC::CL_TEMP_TRACER<SG_TRC::SG_TRC_Default_Trc_Type> Tracer (LocalFctName, _PROFILE_PARAMS);\
	Tracer.SetOpenGL(true);
#else// _GPUCV_PROFILE
#define GPUCV_DD_TRANSFER_BENCH(NAME, SRC_TYPE, DEST_TYPE, DD_SRC, DD_DEST, DATA_TRANSFER)
#endif

}//namespace GCV

#endif//__GPUCV_TEXTURE_DATADSC_BASE_H

