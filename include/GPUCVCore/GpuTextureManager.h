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
#ifndef __GPUCV_CORE_TEXTURE_MANAGER_H
#define __GPUCV_CORE_TEXTURE_MANAGER_H
#include <GPUCVCore/config.h>
#include <SugoiTools/cl_tpl_manager_array.h>


namespace GCV{
/*! 
*	\brief image data management
*	 \author Yannick Allusse
*	\author Erwan Guehenneux
*	\author Jean-Philippe Farrugia
*
*  Manage all DataContainer objects and connect them with OpenCV by using CvArr pointer value.
*  adding and removing corresponding objects.
*/
class _GPUCV_CORE_EXPORT TextureManager
	: public SGE::CL_TEMPLATE_OBJECT_MANAGER/*CL_TplObjManager*/<DataContainer, void*>
	, public CL_Singleton<TextureManager>
{
public:
	typedef		SGE::CL_TEMPLATE_OBJECT_MANAGER<DataContainer, void*> TplManager;
	typedef		void		TypeID;	//! < Type of the identifier.
	typedef		void*		TypeIDPtr;	//! < Type of the identifier.
	typedef		DataContainer*	TplObjPtr;

public :  
	/*! 
	*	\brief default constructor
	*/     
	TextureManager();

	/*!
	*	\brief destructor
	*/   
	~TextureManager();

	/**	\brief Look for given _ID into the image manager and return the associated CvgArr.
	\note If no DataContainer is found for given _ID, we create a new one.
	\sa Find().
	*/
	template<typename NewObjType>
	DataContainer* Get(TypeIDPtr _ID)//, DataContainer::TextureLocation _Location=DataContainer::LOC_NO_LOCATION, bool _data_transfer=true)
	{
		SG_Assert(_ID, "TextureManager ::Get()=> _ID obj is null");
#if 0//test
		if(dynamic_cast<DataContainer*>((void*)_ID)!=NULL)
		{
			if(dynamic_cast<NewObjType*>((void*)_ID)!=NULL)
			{
				//test
				GPUCV_NOTICE("TextureManager::Get() - >object is alreay of the good type");
				TplManager::AddObj((DataContainer*)_ID);
				return (DataContainer*)_ID;
			}
		}
#endif
		DataContainer* temp= Find(_ID);
		if(!temp)
		{//create new CvgArr
			//should be put outside of the manager
			temp =  new NewObjType(&_ID);//CvgArr * newCvg = new CvgArr(&_ID);
			AddObj(temp);
		}
		//if(_Location!=DataContainer::LOC_NO_LOCATION)
		//	temp->SetLocation(_Location, _data_transfer);
		return temp;
	}
	//=======================================================


	/**	\brief Look for given _ID into the image manager and return the associated CvgArr.
	\note If no DataContainer is found for given _ID, we don't create it and return NULL.
	\sa Get(TypeIDPtr, _Location, _data_transfer).
	*/
	DataContainer* Find(TypeIDPtr _ID);

	/** \brief Print some informations about textures that the manager contains and some memory informations.
	*/ 
	virtual 
		void
		PrintAllObjects();
			

protected:
	//_SG_TLS_INLINE
	TplManager::TplHdle CreateObj(IDRefCst _ID, std::string _ObjType="")
	{
		DataContainer * temp = new DataContainer(_ID);
		//SetObjId(temp, _ID);
		return temp;//new T(_ID);
	}
public:
};


/*!
*	\brief get the main instance of a TextureManager
*	\return TextureManager * -> pointer to the main instance of TextureManager
*/
#define GetTextureManager() TextureManager::GetSingleton()

}//namespace GCV

#endif
