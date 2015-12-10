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
#ifndef __GPUCV_TEXTURE_H
#define __GPUCV_TEXTURE_H

#include <GPUCVTexture/DataDsc_base.h>
#include <GPUCVHardware/moduleInfo.h>
namespace GCV{
class TextureGrp;
class DataContainer;


/** \brief Return the image depth in openGL Format.
*	\note The last used Data Descriptor is used to get the value.
*	\sa GetWidth(), GetHeight(), GetnChannels(), GetCVDepth(CvArr*).
*/
_GPUCV_TEXTURE_EXPORT GLuint GetGLDepth(DataContainer * tex);
_GPUCV_TEXTURE_EXPORT GLuint GetGLDepth(DataDsc_Base * tex);
/** \brief Return the image width.
*	\note The last used Data Descriptor is used to get the value.
*	\sa GetGLDepth(), GetHeight(), GetnChannels().
*/
_GPUCV_TEXTURE_EXPORT GLuint GetWidth(DataContainer * tex);
_GPUCV_TEXTURE_EXPORT GLuint GetWidth(DataDsc_Base * tex);
/** \brief Return the image height.
*	\note The last used Data Descriptor is used to get the value.
*	\sa GetWidth(), GetGLDepth(), GetnChannels().
*/
_GPUCV_TEXTURE_EXPORT GLuint GetHeight(DataContainer * tex);
_GPUCV_TEXTURE_EXPORT GLuint GetHeight(DataDsc_Base * tex);

/** \brief Return the image number of channels.
*	\note The last used Data Descriptor is used to get the value.
*	\sa GetWidth(), GetHeight(), GetGLDepth().
*/
_GPUCV_TEXTURE_EXPORT GLuint GetnChannels(DataContainer * tex);
_GPUCV_TEXTURE_EXPORT GLuint GetnChannels(DataDsc_Base * tex);


//! \sa http://www.mathematik.uni-dortmund.de/~goeddeke/gpgpu/tutorial3.html#conventionalreadback
/**
*	\brief Most important class, it is used to manipulate texture properties and data.
*	DataContainer is one of the most important class from GpuCV, it is used to manipulate data of any kind and manage automatique transfer
between all possible data locations(Central memory, Graphics memory...).
*	\todo Add support for region of interest ROI
*	\todo Add support for read only, write only, Read/Write texture type???
*/

class _GPUCV_TEXTURE_EXPORT  DataContainer
	:	public CL_Profiler,
	public SGE::CL_BASE_OBJ<void*>
{
	friend class TextureGrp;
protected :
#if _GPUCV_DEPRECATED
	GLuint				m_nChannel;		//!< Number of channels used by the image.
	GLuint				m_nChannel_force; //!< Flag to force number of channel manually, this can happen when image internal format does not match real data format.
#endif
	GLuint				m_location;								//!< Give the actual location of the data, CPU or GPU?
	bool				m_needReload;							//!< Set to true when some parameters of the texture have been changed and need the texture to be reloaded.
	std::string			m_label;								//!< Image label, useful to add debugging name to textures.
	DataDsc_Base		*m_textureLocationsArr[_GPUCV_MAX_TEXTURE_DSC];	//!< List of data descriptors. Only one object of each type is allowed.
	unsigned char		m_textureLocationsArrLastID;				//!< Index of the last data descriptors of the list.
	DataDsc_Base		*m_textureLastLocations;					//!< Pointer to an object that is supposed to have the "freshest data".
	//! When image is processed by the switch mecanisme, this flag will say if we allow switching on this image or if we force one implementation.
	_DECLARE_MEMBER(BaseImplementation, SwitchLockImplementationID);
public :
	typedef void *				TplIDType;			//!< Type of texture name used in manager.

	//! Options that can be used with the CL_Options class.
	enum TextureOptions
	{
		UBIQUITY		=	0X0001,		//!< When true, image can be at several location at the same time, usually when it is an input image. Else only one location is permitted
		CPU_RETURN		=	0X0002,		//!< When true, image is transfered back to CPU after a filtered has been applied. Default behavior.
		DEST_IMG		=	0x0008,		//!< Set the texture as a destination texture, else it is considered as an input texture.
		//debugging
		DBG_IMG_TRANSFER	= 0x0010,	//!< When true, show debug information on image transfer.
		DBG_IMG_FORMAT		= 0x0020,	//!< When true, show debug information on image format and properties.
		DBG_IMG_LOCATION	= 0x0040,	//!< When true, show debug information on image location mechanism.
		DBG_IMG_MEMORY		= 0x0080,	//!< When true, show debug information on image momery allocation.
		//reserved from 0x100 to 0x800 by LCL_OPT_DEBUG
		PRESERVE_MEMORY		= 0x1000,	//!< When true, GpuCV free texture locations when not used(preserve memory but reduce speed).
		SMART_TRANSFER		= 0x2000,	//!< When true, GpuCV will estimate all transfer time when calling SetLocation() and choose the best one.
		VISUAL_DBG			= 0x8000	//!< When true, GpuCV will show all the data container details on image.
	};

	//-------------------------------------
	//
	//! \name Constructors/Destructors
	//@{
	//-------------------------------------
	/**
	*\brief Constructor using an ID for the manager, create a DataContainer object without any location(LOC_NO_LOCATION).
	*/
	__GPUCV_INLINE
		DataContainer(const TplIDType _ID=0);

	/**
	*	\brief Copy Constructor
	*	\param &_Copy => Copy only the DataDsc* that have the m_dataFlag set to true;
	*	\param _dataOnly => Copy only the Data;
	*/
	__GPUCV_INLINE
		DataContainer(DataContainer &_Copy, bool _dataOnly=false);

	/**
	*	\brief Destructor
	*/
	__GPUCV_INLINE
		~DataContainer();

	//-------------------------------------
	//@}
	//! \name Integration with framework
	//@{
	//-------------------------------------

	/**	\brief Define a label for current texture, label are printed by Print() when debugging.
	*	\sa Print(), GetLabel().
	*/
	__GPUCV_INLINE
		void	SetLabel(const std::string _lab);
	/**	\brief Get current texture label, label are printed by Print() when debugging.
	*	\sa Print(), SetLabel().
	*/
	__GPUCV_INLINE
		const std::string & GetLabel()const;

	/** \brief Return a string describing the texture with following format : "ID:%OpenGL Texture id% | '%Texture Label%'".
	*	\return std::string : texture short description.
	*/
	__GPUCV_INLINE
		virtual
		const	std::string	GetValStr()const;

	virtual
		std::string PrintMemoryInformation(std::string text)const;

#if _GPUCV_GL_USE_MIPMAPING
	//virtual bool GenerateMipMaps(GLuint _textureMinFilter = GL_NEAREST_MIPMAP_NEAREST);
#endif
	//=================================
	//Access member group
	/** \brief Get a pointer to CPU data, if image is not stored into CPU, it returns an empty pointer.
	*	\return pointer contained into m_pixels, corresponding to CPU image buffer.
	*/
	__GPUCV_INLINE
		PIXEL_STORAGE_TYPE **	_GetPixelsData();

#if _GPUCV_DEPRECATED
	/**
	*	\brief Set the pointer to CPU pixels data. Do not change image location.
	*/
	//			__GPUCV_INLINE
	//		void	_SetPixelsData(PIXEL_STORAGE_TYPE ** _pix);
#endif

	/**	\brief Print debugging informations about the texture.
	*/
	virtual void	Print()const;
	//Render to texture
	/**	\brief Force the "render to texture" mechanism to start for current texture.
	*	Start the mechanism to render to current texture. Should be use when we render to only one texture.
	*	If texture is not on GPU, a texture is allocated.
	*	If the texture option AUTO_MIPMAP is enable, MipMaps will be generated when calling UnForceRenderToTexture().
	*	\sa UnForceRenderToTexture(), SetRenderToTexture(), UnsetRenderToTexture().
	*/
	__GPUCV_INLINE
		void		ForceRenderToTexture();

	/**	\brief Force the "render to texture" mechanism to stop for current texture
	*	Stop the mechanism to render to current texture. Should be use when we render to only one texture.
	*	If the texture option AUTO_MIPMAP is enable, MipMaps will be generated when calling UnForceRenderToTexture().
	*	\sa ForceRenderToTexture(), SetRenderToTexture(), UnsetRenderToTexture().
	*/
	__GPUCV_INLINE
		void		UnForceRenderToTexture();

	/**	\brief Set the "render to texture" mechanism to start for current texture
	*	Start the mechanism to render to current texture. Should be use when we render to only one texture.
	*	If the texture option AUTO_MIPMAP is enable, MipMaps will be generated when calling UnsetRenderToTexture().
	*	\sa ForceRenderToTexture(), UnForceRenderToTexture(), UnsetRenderToTexture().
	*/
	__GPUCV_INLINE
		void		SetRenderToTexture();

	/**	\brief Force the "render to texture" mechanism to stop for current texture
	*	Stop the mechanism to render to current texture. Should be use when we render to only one texture.
	*	If texture is not on GPU, a texture is allocated.
	*	If the texture option AUTO_MIPMAP is enable, MipMaps will be generated when calling UnForceRenderToTexture().
	*	\sa ForceRenderToTexture(), UnForceRenderToTexture(), SetRenderToTexture().
	*/
	__GPUCV_INLINE
		void		UnsetRenderToTexture();
	//=========

protected:

public:
	/**	\brief Set given option '_opt' to given value 'val'.
	*	\sa CL_Options.
	*/
	void SetOption(CL_Options::OPTION_TYPE _opt, bool val);

	virtual void _CopyProperties(DataContainer &_src);
	virtual void _CopyAllDataDsc(DataContainer &_src, bool _dataOnly);

	virtual void _CopyActiveDataDsc(DataContainer &_src, bool _dataOnly);


	//=========================================================
	/** \brief Force the number of channels really used in the texture.
	*	During conversion between CPU format to GPU texture format, it may happen that the best conversion is not compatible with current GPU. So we may need to use internal format with higher number of channel.
	*	In that case we force the number of channels that are really used.
	*	\sa _GetNChannels(), _SetNChannels(), _UnForceNChannels().
	\deprecated
	*/
#if _GPUCV_DEPRECATED
	__GPUCV_INLINE
		void _ForceNChannels(GLuint _nchannel)
	{
		m_nChannel=_nchannel;
		m_nChannel_force = true;
	}
	/** \brief Unforce the channels number.
	*	\sa _GetNChannels(), _SetNChannels(), _ForceNChannels().
	\deprecated
	*/
	__GPUCV_INLINE
		void _UnForceNChannels()
	{	m_nChannel_force = false;}
#endif

	//-------------------------------------
	//@}
	//! \name Data descriptors manipulation
	//@{
	//-------------------------------------
	/** \brief Check if this DataContainer contains the given DataDsc.
	*	\return true if given DataDsc exists.
	*/
	template <typename TPLType_DataDsc>
	bool _IsLocation()
	{
		return (FindDataDscID<TPLType_DataDsc>()!=-1)? true:false;
	}

	/** \brief Return a DataDsc object corresponding to the TPLType_DataDsc. If not exists, it create it.
	*	\return Corresponding data descriptor.
	*/
	template <typename TPLType_DataDsc>
	TPLType_DataDsc * GetDataDsc(void)
	{
		//CLASS_FCT_SET_NAME_TPL(TPLType_DataDsc,"GetDataDsc");
		//CLASS_FCT_PROF_CREATE_START();//GPUCV_PROFILE_CURRENT_FCT(FctName, (DataContainer*) this, true);
		for (int i = 0; i < m_textureLocationsArrLastID; i++)
		{
			if(m_textureLocationsArr[i]==NULL)
				continue;
			if(dynamic_cast<TPLType_DataDsc*>(m_textureLocationsArr[i]))
				return dynamic_cast<TPLType_DataDsc*>(m_textureLocationsArr[i]);
		}
		//not found...
		//we create it and add it
		TPLType_DataDsc * newObj = new TPLType_DataDsc();
		m_textureLocationsArr[m_textureLocationsArrLastID++] = newObj;
		newObj->SetParent(this);
		if(m_textureLastLocations==NULL)
			m_textureLastLocations = newObj;
		return newObj;
	}

	/** \brief Return the index of data descriptor object in the data descriptor list.
	*	\return Given data descriptor type index or -1 if not found.
	*/
	template <typename TPLType_DataDsc>
	char FindDataDscID(void)
	{
		//CLASS_FCT_SET_NAME_TPL(TPLType_DataDsc,"FindDataDscID");
		//CLASS_FCT_PROF_CREATE_START();//GPUCV_PROFILE_CURRENT_FCT(FctName, (DataContainer*)this, true);
		for (int i = 0; i < m_textureLocationsArrLastID; i++)
		{
			if(m_textureLocationsArr[i]==NULL)
				continue;
			if(dynamic_cast<TPLType_DataDsc*>(m_textureLocationsArr[i]))
				return i;
		}
		return -1;
	}

	/** \brief Remove a given data descriptor object from the data descriptor list.
	*/
	template <typename TPLType_DataDsc>
	void RemoveDataDsc()
	{
		CLASS_FCT_SET_NAME_TPL(TPLType_DataDsc,"RemoveDataDsc");
		CLASS_FCT_PROF_CREATE_START();//GPUCV_PROFILE_CURRENT_FCT(FctName, (DataContainer*)this, true);
		int i=0;
		bool Found=false;
		//find and remove the given descriptor
		for (i = 0; i < m_textureLocationsArrLastID; i++)
		{
			if(m_textureLocationsArr[i]==NULL)
				continue;
			
			if(m_textureLocationsArr[i]->IsLocked())
				continue;//can not disable data flag of a locked descriptor

			if(dynamic_cast<TPLType_DataDsc*>(m_textureLocationsArr[i]))
			{
				if(m_textureLastLocations==m_textureLocationsArr[i])
					m_textureLastLocations = NULL;

				delete m_textureLocationsArr[i];
				m_textureLocationsArrLastID--;
				Found = true;
				break;
			}
		}
		//reformat descriptor table...fill the hole..
		if(Found)
		{
			for (i; i < m_textureLocationsArrLastID; i++)
			{
				m_textureLocationsArr[i] = m_textureLocationsArr[i+1];
			}
			m_textureLocationsArr[i+1]=NULL;
			if(m_textureLastLocations==NULL)
				m_textureLastLocations = m_textureLocationsArr[0];
		}

	}

	/** \brief Remove and delete all data descriptor objects from the data descriptor list.
	*	\return Number of objects removed and deleted.
	*/
	virtual char RemoveAllTextureDesc();

	/** \brief Remove and delete all data descriptor objects from the data descriptor list except one type given by TPLType_DataDsc.
	*	\return Number of objects removed and deleted.
	*/
	template <typename TPLType_DataDsc>
	char RemoveAllDataDscExcept()
	{

		CLASS_FCT_SET_NAME_TPL(TPLType_DataDsc,"RemoveAllDataDscExcept");
		CLASS_FCT_PROF_CREATE_START();//GPUCV_PROFILE_CURRENT_FCT(FctName, this, true);

		int pos = FindDataDscID<TPLType_DataDsc>();
		DataDsc_Base * ObjToKeep = NULL;
		int i=0;
		int NbrDeleted=0;
		for (int i = 0; i < m_textureLocationsArrLastID; i++)
		{
			if(m_textureLocationsArr[i]==NULL)
				continue;

			if(m_textureLocationsArr[i]->IsLocked())
				continue;//can not disable data flag of a locked descriptor

			if(pos==i)
			{
				ObjToKeep = m_textureLocationsArr[pos];
				continue;
			}
			CLASS_DEBUG("Delete obj:" << *m_textureLocationsArr[i]);
			delete m_textureLocationsArr[i];
			m_textureLocationsArr[i] = NULL;
			NbrDeleted++;
		}
		m_textureLocationsArrLastID = 1;
		m_textureLocationsArr[0]=ObjToKeep;
		m_textureLastLocations = ObjToKeep;
		return NbrDeleted;
	}

	template <typename TPLType_DataDsc>
	TPLType_DataDsc* SetLocation(bool _dataTransfer=true)
	{
		CLASS_FCT_SET_NAME_TPL(TPLType_DataDsc,"SetLocation");
		CLASS_FCT_PROF_CREATE_START();//GPUCV_PROFILE_CURRENT_FCT(FctName, (DataContainer*)this, true);

		bool CopyDone = false;
		TPLType_DataDsc * DestinationObj = GetDataDsc<TPLType_DataDsc>();
		bool DestImg = (GetOption(DEST_IMG))?true:false;
		bool Ubi	 = (GetOption(UBIQUITY))?true:false;
		bool PreserveMemory	 = (GetOption(PRESERVE_MEMORY))?true:false;

		//we do not transfer data...so no need to look for data locations...
		//we take the last known data location to perform parameter transfer
		if (!_dataTransfer
			&& m_textureLastLocations!= DestinationObj
			&& m_textureLastLocations!= NULL)
		{//copy not done due to datatransfer, but we use last location obj to transfer parameters...
			m_textureLastLocations->UnsetReshape();
			DestinationObj->TransferFormatFrom(m_textureLastLocations);
			DestinationObj->Allocate();
			CopyDone = true;
		}


		//try to find best source considering the transfer time estimation
		char BestSrcID=-1;
		if(!CopyDone)
		{
			if(GetOption(SMART_TRANSFER))
			{

				long int TimeEstimation;
				TimeEstimation = SetLocationEstimation<TPLType_DataDsc>(BestSrcID, _dataTransfer);
				if(TimeEstimation==0)
				{
					CLASS_DEBUG("Data already on location.");
					CopyDone = true;
				}
				else if(BestSrcID>-1)
				{
					unsigned char uchar_BestSrcID = BestSrcID;
					GPUCV_WARNING("Transfer from "<< m_textureLocationsArr[uchar_BestSrcID]->GetClassName()
						<< " to "<< std::string(typeid(TPLType_DataDsc).name())
						<< " will take about " << TimeEstimation << "us");
					if(m_textureLocationsArr[uchar_BestSrcID]->CopyTo(DestinationObj, _dataTransfer))
						CopyDone = true;
					else if (DestinationObj->CopyFrom(m_textureLocationsArr[uchar_BestSrcID], _dataTransfer))
						CopyDone = true;
					else
						CopyDone = false;
				}
				else
				{
					GPUCV_WARNING("Transfer to "<< std::string(typeid(TPLType_DataDsc).name()) << " as not known path...");
				}
			}
		}


		//can't get it from the last location...try others...
		if(CopyDone==false)
		if(BestSrcID==-1)
		{
			for (int i = 0; i < m_textureLocationsArrLastID; i++)
			{
				if(m_textureLocationsArr[i]==NULL)
					continue;

				if(DestinationObj->HaveData() && !CopyDone)
				{
					CopyDone = true;
					CLASS_DEBUG("Data already on location.");
					//continue;
				}

				if(DestinationObj == m_textureLocationsArr[i])
				{
					continue;
				}
				//check locker, locked desciptor can be used to get data
				/*
				if(m_textureLocationsArr[i]->IsLocked())
				{
					CLASS_DEBUG(m_textureLocationsArr[i]->GetValStr() << " is locked.");
					continue;
				}*/
				if(m_textureLocationsArr[i]->HaveData() && !CopyDone)
				{
					if(m_textureLocationsArr[i]->CopyTo(DestinationObj, _dataTransfer))
					{
						CopyDone = true;
 					}
					else if (DestinationObj->CopyFrom(m_textureLocationsArr[i], _dataTransfer))
					{
						CopyDone = true;
					}
				}
			}
		}




		if(CopyDone)
		{//copy has been done
			m_textureLastLocations = DestinationObj;
			SetDataFlag<TPLType_DataDsc>(true);
			return DestinationObj;
		}
		else if (DestImg)
		{//no need to make a copy..we allocate it.
			DestinationObj->Allocate();
			DestinationObj->SetDataFlag(true);//data are here now
			m_textureLastLocations = DestinationObj;
			return DestinationObj;
		}
		else
		{
			//could not find a way to copy data...
#if _GPUCV_DEBUG_MODE
			GPUCV_DEBUG("List of DataDsc_Base:");
			for (int i = 0; i < m_textureLocationsArrLastID; i++)
			{
				GPUCV_ERROR(*m_textureLocationsArr[i]);
			}
#endif
			CLASS_ASSERT(0, "template<TPLType_DataDsc>SetLocation()=> could not find a way to copy data...");
			return NULL;
		}
	}

	template <typename TPLType_DataDsc>
	long int SetLocationEstimation(char & _bestSrcId, bool _dataTransfer=true)
	{
		CLASS_FCT_SET_NAME_TPL(TPLType_DataDsc,"SetLocationEstimation");
		CLASS_FCT_PROF_CREATE_START();//GPUCV_PROFILE_CURRENT_FCT(FctName, (DataContainer*)this, true);

		//

		long int MinTime = 999999999;
		long int CurTime = 0;
		_bestSrcId = -1;

		TPLType_DataDsc * DestinationObj = GetDataDsc<TPLType_DataDsc>();


#if _GPUCV_PROFILE
		//create a base param set
		SG_TRC::CL_TRACE_BASE_PARAMS* _PROFILE_PARAMS = NULL;

		_PROFILE_PARAMS = new SG_TRC::CL_TRACE_BASE_PARAMS();
			//_PROFILE_PARAMS->AddChar("src_type", "");
			//_PROFILE_PARAMS->AddChar("dst_type", "");
			_PROFILE_PARAMS->AddChar("dataTr", (_dataTransfer)?"1":"0");
			if(m_textureLocationsArr[0])
			{
			_PROFILE_PARAMS->AddChar("size", SGE::ToCharStr(m_textureLocationsArr[0]->GetDataSize()).data());
			_PROFILE_PARAMS->AddChar("format", GetStrGLTextureFormat(m_textureLocationsArr[0]->GetPixelFormat()).data());
			_PROFILE_PARAMS->AddChar("type", GetStrGLTexturePixelType(m_textureLocationsArr[0]->GetPixelType()).data());
			}

			if(DestinationObj->HaveData())
			{
				CLASS_DEBUG("Data already on location.");
				return 0;
			}

			for (int i = 0; i < m_textureLocationsArrLastID; i++)
			{
				if(m_textureLocationsArr[i]==NULL)
					continue;

				if(DestinationObj == m_textureLocationsArr[i])
					continue;

				//check locker
				if(m_textureLocationsArr[i]->IsLocked())
				{
					CLASS_DEBUG(m_textureLocationsArr[i]->GetValStr() << " is locked.");
					continue;
				}
				if(m_textureLocationsArr[i]->HaveData())
				{
					CurTime = m_textureLocationsArr[i]->CopyToEstimation(DestinationObj, _dataTransfer, _PROFILE_PARAMS);
					if(CurTime!=-1 && CurTime < MinTime)
					{
						MinTime = CurTime;
						_bestSrcId = i;
					}
				}
			}
#endif
			if(_bestSrcId!=-1)
			{//we found a path to copy
				return MinTime;
			}
			else
			{//no path
				return -1;
			}
	}

	template <typename TPLType_DataDsc>
	TPLType_DataDsc* SetDataFlag(bool _dataFlag, bool _forceUniqueData=false)
	{
		CLASS_FCT_SET_NAME_TPL(TPLType_DataDsc,"SetDataFlag");
		CLASS_FCT_PROF_CREATE_START();//GPUCV_PROFILE_CURRENT_FCT(FctName, (DataContainer*)this, true);

		TPLType_DataDsc * DestinationObj = GetDataDsc<TPLType_DataDsc>();

		//if set to false, no need to update other DD
		//we return directly
		if(_dataFlag == false)
		{
			DestinationObj->SetDataFlag(false);
		}
		else
		{//check all other flags
			bool DestImg = (GetOption(DEST_IMG))?true:false;
			bool Ubi	 = (GetOption(UBIQUITY))?true:false;
			bool PreserveMemory	 = (GetOption(PRESERVE_MEMORY))?true:false;

			for (int i = 0; i < m_textureLocationsArrLastID; i++)
			{
				if(m_textureLocationsArr[i]==NULL)
					continue;

				//if the destination obj is locked or lock another object, they must not be updated here
				//they will be updated by the locker/locked obj destinationObj
				if(m_textureLocationsArr[i]->LockedBy() == DestinationObj)
					continue;//can not update data flag of a locked descriptor
				if(m_textureLocationsArr[i]->GetLockedObj() == DestinationObj)
					continue;//can not update data flag of a locked descriptor
				

				if(DestinationObj ==m_textureLocationsArr[i])
				{//DD to set.
					DestinationObj->SetDataFlag(_dataFlag);
				}
				else if (_forceUniqueData && _dataFlag)
				{//we remove data from all other objects
					if(PreserveMemory)//release memory
					{
						m_textureLocationsArr[i]->Free();
					}
					else//preserve memory but reset dataflag
						m_textureLocationsArr[i]->SetDataFlag(false);
				}
				else
				{//other images
					if(_dataFlag==true)
					{
						//do they keep their data flag?
						if(		DestImg //dest image can have only one location active
							|| !Ubi)	//tolerate only one DD active
						{
							if(PreserveMemory)//release memory
							{
								m_textureLocationsArr[i]->Free();
							}
							else//preserve memory but reset dataflag
								m_textureLocationsArr[i]->SetDataFlag(false);
						}
					}
				}
			}
		}
		return DestinationObj;
	}

	template <typename TPLType_DataDsc>
	bool DataDscHaveData()
	{
		CLASS_FCT_SET_NAME_TPL(TPLType_DataDsc,"DataDscHaveData");
		CLASS_FCT_PROF_CREATE_START();//GPUCV_PROFILE_CURRENT_FCT(FctName, (DataContainer*)this, true);

		if(!_IsLocation<TPLType_DataDsc>())
			return false;
		else
		{
			TPLType_DataDsc * DD_Obj = GetDataDsc<TPLType_DataDsc>();
			return DD_Obj->HaveData();
		}
	}

	/** Exchange one DataDsc* type between two DataContainer, it is used mainly when using temp container to avoid copies.
	*\note If one of the source/target do not own the corresponding DataDsc*, a one way transfer is done.
	*/
	template <typename TPLType_DataDsc>
	bool SwitchDataDsc(DataContainer * _src)
	{
		CLASS_FCT_SET_NAME_TPL(TPLType_DataDsc,"SwitchDataDsc");
		CLASS_FCT_PROF_CREATE_START();//GPUCV_PROFILE_CURRENT_FCT(FctName, (DataContainer*)this, true);

		CLASS_ASSERT(_src, "No texture to switch DataDsc");
		int posA, posB;
		posA = FindDataDscID<TPLType_DataDsc>();
		posB = _src->FindDataDscID<TPLType_DataDsc>();
		

		if(posA!=-1 && posB != -1)
		{
			CLASS_ASSERT((*dynamic_cast<TextSize<GLsizei>*>(_src->m_textureLocationsArr[posB]))==(*dynamic_cast<TextSize<GLsizei>*>(m_textureLocationsArr[posA])), "Texture have different size, they can't switch data");
			SWITCH_VAL(DataDsc_Base*, m_textureLocationsArr[posA], _src->m_textureLocationsArr[posB]);
			m_textureLocationsArr[posA]->SetParent(this);
			_src->m_textureLocationsArr[posB]->SetParent(_src);
			m_textureLastLocations = m_textureLocationsArr[posA];
			_src->m_textureLastLocations = _src->m_textureLocationsArr[posB];
		}
		//we have only one side...we just perform one way transfer
		//no test is done yet on image properties with the destination DataContainer, we assume it is all right...
		//CLASS_ASSERT((*dynamic_cast<TextSize<GLsizei>*>(_src->m_textureLocationsArr[posB]))==(*dynamic_cast<TextSize<GLsizei>*>(m_textureLocationsArr[posA])), "Texture have different size, they can't switch data");
		else if(posA!=-1)
		{
			//add to destination container and remove paternity
			_src->AddNewDataDsc(RemoveDataDsc(posA));
		}
		else
		{
			//add to destination container and remove paternity
			AddNewDataDsc(_src->RemoveDataDsc(posB));
		}
		return true;
	}

	template <typename TPLType_DataDsc>
	bool CopyDataDsc(DataContainer * _src, bool _dataTransfer)
	{
		CLASS_FCT_SET_NAME_TPL(TPLType_DataDsc,"CopyDataDsc");
		CLASS_FCT_PROF_CREATE_START();//GPUCV_PROFILE_CURRENT_FCT(FctName, (DataContainer*)this, true);

		CLASS_ASSERT(_src, "No texture to switch DataDsc");
		//CLASS_ASSERT((*dynamic_cast<TextSize<GLsizei>*>(_src))==(*dynamic_cast<TextSize<GLsizei>*>(this)), "Texture have different size, they can't switch data");
		int posRemote;
		//create local one
		TPLType_DataDsc * localDD = GetDataDsc<TPLType_DataDsc>();
		//find remote one
		posRemote = _src->FindDataDscID<TPLType_DataDsc>();
		CLASS_ASSERT(localDD, "No local DataDsc");
		CLASS_ASSERT(posRemote, "No remote DataDsc");
		if(localDD && posRemote!=-1)
		{//clone DataDsc
			localDD->Clone(dynamic_cast<TPLType_DataDsc*>(_src->m_textureLocationsArr[posRemote]),_dataTransfer);
			m_textureLastLocations = localDD;
			return true;
		}
		return false;
	}

protected:
	DataDsc_Base* AddNewDataDsc(DataDsc_Base*_newDD)
	{
		CLASS_FCT_SET_NAME("AddNewDataDsc");
		//CLASS_FCT_PROF_CREATE_START();//GPUCV_PROFILE_CURRENT_FCT(FctName, (DataContainer*)this, true);


		CLASS_ASSERT(_newDD, "No DataDsc_Base to Add");
		//CLASS_ASSERT((*dynamic_cast<TextSize<GLsizei>*>(_src))==(*dynamic_cast<TextSize<GLsizei>*>(this)), "Texture have different size, they can't switch data");
		CLASS_ASSERT(_GPUCV_MAX_TEXTURE_DSC > m_textureLocationsArrLastID, "Maximum DataDsc_Base reached:" << _GPUCV_MAX_TEXTURE_DSC << "please consider incresing value _GPUCV_MAX_TEXTURE_DSC");
		m_textureLocationsArr[m_textureLocationsArrLastID++] = _newDD;
		m_textureLastLocations = _newDD;
		m_textureLastLocations->SetParent(this);
		//change parent ID cause it might have changed...
		if(m_textureLastLocations->GetNewParentID())
			SGE::CL_BASE_OBJ<void*>::SetID(m_textureLastLocations->GetNewParentID());
		return _newDD;
	}


	/** Remove a DataDsc* from current container based on its ID. The DataDsc is not desctroyed and m_textureLocationsArr is rearranged.
	*/
	DataDsc_Base* RemoveDataDsc(unsigned int _ID)
	{
		CLASS_FCT_SET_NAME("RemoveDataDsc");
		
		CLASS_ASSERT(_ID <m_textureLocationsArrLastID, "Index out of range");
		DataDsc_Base* pCurDataDsc = m_textureLocationsArr[_ID];
		m_textureLocationsArr[_ID] = NULL;//set it to NULL, cause it might be the last one
		m_textureLocationsArr[_ID] = m_textureLocationsArr[m_textureLocationsArrLastID];//fill hole
		m_textureLocationsArrLastID--;
		pCurDataDsc->SetParent(NULL);

		m_textureLastLocations = NULL;

		return pCurDataDsc;
	}
public:
	DataDsc_Base* GetLastDataDsc()
	{
		CLASS_FCT_SET_NAME("GetLastDataDsc");
		//GPUCV_PROFILE_CURRENT_FCT(FctName, (DataContainer*)this, true);
		return m_textureLastLocations;
	}

	/** Call the flush function of all Data Descriptors.
	*	\sa DataDsc_Base::Flush()
	*/
	void Flush()
	{
		for (int i = 0; i < m_textureLocationsArrLastID; i++)
		{
			if(m_textureLocationsArr[i])
				if(m_textureLocationsArr[i]->HaveData())
					m_textureLocationsArr[i]->Flush();
		}
	}

	/** \brief Return the total amount of memory allocated for all sub DataDsc*(CPU+GPU memory).
	*/
	long int GetMemoryAllocated() const
	{
		long int Sum = 0;
		for (int i = 0; i < m_textureLocationsArrLastID; i++)
		{
			if(m_textureLocationsArr[i]==NULL)
				continue;
			Sum += m_textureLocationsArr[i]->GetLocalMemoryAllocated();
		}
		return Sum;
	}


	//========================================================
	//========================================================
	//========================================================
};




#if _GPUCV_DEBUG_MODE
#if _GPUCV_PROFILE
#define _GPUCVTEXTURE_GL_ERROR_TEST(TEXT)
#else
#define _GPUCVTEXTURE_GL_ERROR_TEST(TEXT) \
	if(ShowOpenGLError(__FILE__, __LINE__))\
	if(TEXT)\
	TEXT->Print();
//TEXT->PushSetOptions(CL_Options::LCL_OPT_DEBUG, 1);
//TEXT->PopOptions();
#endif
#else
#define _GPUCVTEXTURE_GL_ERROR_TEST(TEXT)
#endif

}//namespace GCV
#endif //GPUCV_HARDWARE_TEXTURE_H
