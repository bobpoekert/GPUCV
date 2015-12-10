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



#ifndef __GPUCV_CVGARR_H
#define __GPUCV_CVGARR_H

#include <GPUCV/config.h>
#include <GPUCV/DataDsc_IplImage.h>
#include <GPUCV/DataDsc_CvMat.h>

namespace GCV{
/*! 
*	\brief Class to manage CvArr(IplImage and CvMat) on GPU
*	\author Yannick Allusse
*	\author Erwan Guehenneux
*	\author Jean-Philippe Farrugia
*  A CvgArr is an object inherited from DataContainer, its goal is to encapsulate and manage IplImage into GPUCV library.
*  It manages automatic image data transfer between RAM (IplImage) and GPU (OpenGL texture using DataContainer).
*/
class _GPUCV_EXPORT  CvgArr
	: public DataContainer
{
public:
	typedef		CvArr	TInput;		//!< Input type definition.
private :
	DataDsc_IplImage * m_texIplImage;	//!< pointer to corresponding DataDsc_IplImage class.
#if _GPUCV_SUPPORT_CVMAT
	DataDsc_CvMat	* m_texCvMat;		//!< pointer to corresponding DataDsc_CvMat class.
#endif
public : 
	/*! \brief Main Constructor
	*  It creates a link to the corresponding CvArr but does not allocate texture yet.
	*	\param  _origin -> link to the related IplImage(CPU image).
	*	\note NULL pointer is accepted.
	*/
	explicit __GPUCV_INLINE
		CvgArr(TInput	** _origin);

	__GPUCV_INLINE
		CvgArr(void);

#if _GPUCV_SUPPORT_CVMAT
	/*! \brief Constructor for CvMat
	*	\param  _origin -> link to the related CvMat(CPU matrix).
	*	\sa CvgArr(TInput), CvgArr(IplImage).
	*	\note NULL pointer is accepted.
	*/
	explicit __GPUCV_INLINE
		CvgArr(CvMat	** _origin);
#endif
	/*!  \brief Constructor for IplImage
	*	\param  _origin -> link to the related IplImage(CPU image).
	*	\sa CvgArr(TInput), CvgArr(CvMat).
	*	\note NULL pointer is accepted.
	*/
	explicit __GPUCV_INLINE
		CvgArr(IplImage ** _origin);

	/*! \brief Copy Constructor
	*/
	explicit __GPUCV_INLINE
		CvgArr(CvgArr & _origin);

	/*! \brief Destructor, free allocated texture.
	*/    
	__GPUCV_INLINE
		~CvgArr();

	/*! \brief Return current CvArr** linked with the object.
	*	\return Current linked CvArr**.
	*	\sa GetCvMat(), GetIplImage().
	*/  
	CvgArr::TInput** GetCvArr()const;


	/*! \brief Affect a new IplImage* to the CvgArr object.
	*	Call the function SetCvArr() with a dynamic cast.
	*	\param  _img => pointer to the new linked IplImage*
	*	\return FALSE if any error occurred, else TRUE.
	*	\sa GetCvArr(), SetCvArr(), SetCvMat()/GetCvMat(), GetIplImage().
	*/   
	__GPUCV_INLINE
		void SetIplImage(IplImage **_img);

	/*! \brief Return current IplImage* linked with the object.
	*	\return Current linked IplImage*.
	*	\sa GetCvMat(), GetCvArr().
	*/  
	__GPUCV_INLINE
		IplImage	* GetIplImage()const;

	/*! \brief Affect a new CvArr* to the CvgArr object.
	*	Affect a new CvArr* to the CvgArr object. The IplImage data pointer will be send to DataContainer to allow future data transfer but no texture are allocated yet.
	*	If a texture was previously allocated, we free it.
	*	\param  _arr => pointer to the new linked CvArr*
	*	\note NULL ponter is supported.
	*	\return FALSE if any error occurred, else TRUE.
	*	\sa GetCvArr(), SetCvMat()/GetCvMat(), SetIplImage()/GetIplImage().
	*/    
	bool SetCvArr(TInput **_arr);

	/*! \brief Get the CvArr of CvgArr (do memory copy between GPU & CPU if necessary)
	*	\return IplImage* : pointer to the related IplImage structure.
	*	\note If the CvgArr is not Linked with any IplImage, NULL is returned and a warning message is raised.
	*	\sa GetCvArr(), SetCvArr(), SetCvMat()/GetCvMat(), SetIplImage()/GetIplImage().
	*/    
	__GPUCV_INLINE
		TInput* 
		GetCvArrObject();

#if _GPUCV_SUPPORT_CVMAT
	/*! \brief Affect a new CvMat* to the CvgArr object.
	*	Call the function SetCvArr() with a dynamic cast.
	*	\param  _mat => pointer to the new linked CvMatt*
	*	\return FALSE if any error occurred, else TRUE.
	*	\sa GetCvArr(), SetCvArr(), GetCvMat(), SetIplImage()/GetIplImage().
	*/   
	__GPUCV_INLINE
		void SetCvMat(CvMat  **_mat);

	/*! \brief Return current CvMat* linked with the object.
	*	\return Current linked CvMat*.
	*	\sa GetIplImage(), GetCvArr().
	*/
	__GPUCV_INLINE
		CvMat		* GetCvMat()const;
#endif

	/*! \brief Get the DataContainer parent object of the CvgArr (do the memory copy between CPU & GPUif needed)
	*	\return DataContainer * :  parent object of the current CvgArr
	*/    
	__GPUCV_INLINE
		GPUCV_TEXT_TYPE		
		GetGpuImage();

	/*! \brief Return a string describing the CvgArr object with following format : "ID:%OpenGL Texture id% | '%Texture Label%' | CvArr:%CvArr object pointer%".
	*	\note 'CvArr' is replace by 'IPL' or 'CvMat' depending on object type.
	*	\return std::string : texture short description.
	*	\sa DataContainer::GetValStr().
	*/
	virtual	__GPUCV_INLINE
		const std::string	GetValStr()const;

	/** \brief Copy texture properties from _src to current object.
	*	Copy texture properties from _src to current object. _src object has any linked CvArr we clone it
	*	and affect the clone handler to current CvgArr. Function does not delete previous linked CvArr object.
	*	\warning If current CvgArr object is already linked to any CvArr object, this one will not be replace but not deleted.
	*/
	virtual
		void _CopyProperties(DataContainer &_src);
	void _CopyProperties(const IplImage *_srcIpl);

#if _GPUCV_SUPPORT_CVMAT
	void _CopyProperties(const CvMat *_srcMat);
#endif

};
}//namespace GCV
#endif
