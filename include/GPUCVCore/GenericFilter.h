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



#ifndef __GPUCV_CORE_GENERIC_FILTER_H
#define __GPUCV_CORE_GENERIC_FILTER_H

#include <GPUCVCore/GpuTextureManager.h>

namespace GCV{
#define FCT_PTR_DRAW_TEXGRP(NAME)void(*NAME)(TextureGrp*,TextureGrp*, GLuint, GLuint)
#define FCT_PTR_DRAW_TEX(NAME)void(*NAME)(DataContainer**, GLuint, DataContainer**, GLuint, GLuint, GLuint)


#if 0//not tested yet
/**
*	\brief Generic filter that must be used as a base class for specific implementation(OpenGL with GLSL, CUDA,...).
*	 \author Yannick Allusse
*
*  This class is a Generic filter that must be used as a base class for specific implementation(OpenGL with GLSL, CUDA,...).
*  It is also the base class used in the filter manager. It have member functions to load, configure, process a given filter and
*  manipulate corresponding data.
*/
class _GPUCV_CORE_EXPORT GenericFilter
	: public SGE::CL_BASE_OBJ<std::string>
{
public:
	typedef	TextSize<GLsizei>	FilterSize;
	typedef	std::string			TpName;
	typedef	float				TpParam;
	typedef DataContainer			TpSingleElem;
	typedef TextureGrp			TpGrpElem;

private: 
	//TpName		m_shader_name;   //!< filter Id.
	TpParam*		m_params;        //!< Parameters for the filter.
	int			m_nbrParams;     //!< Number of parameters.
	bool			m_loaded;
public : 
	/*!
	*	\brief default constructor
	*/
	GenericFilter();

	/*!
	*	\brief Constructor
	*/
	GenericFilter(const TpName & _name);

	/*!
	*	\brief Destructor
	*/
	~GenericFilter();

	/*!
	*	\brief load the given filter.
	*	\return True for sucess, else false.
	*/
	virtual bool Load()=0;

	/*!
	*	\brief Unload the given filter.
	*	\return True for sucess, else false.
	*/
	virtual bool Unload()=0;

	//=====================
	// parameter management
	//=====================
	/*!
	*	\brief initialize the parameter array for the cvgFilter
	*/
	virtual __GPUCV_INLINE
		void   ClearParams();

	/*!
	*	\brief get the parameter array
	*	\return TpParam * -> pointer to the parameter array
	*/
	virtual __GPUCV_INLINE
		TpParam*      GetParams()const;

	/*!
	*	\brief get the number of parameters set in the parameter array
	*	\return int -> number of parameters
	*/
	virtual __GPUCV_INLINE
		int     GetNbParams()const;

	/*!
	*	\brief update a float array of parameters 
	*	For each parameters of the array, test if it already exists and updates it, else add a new parameter.
	*	\param _params -> float array of parameters to update
	*	\param _paramNbr -> number of parameters
	*/
	virtual __GPUCV_INLINE
		void	SetParams(const TpParam * _params , const int _paramNbr);

	//======================================
	// DATA management
	//======================================
protected:
	/*!
	*   \brief Prepare the input or output data group before a filter process.
	*	For each element from the TextureGrp, options will checked and data will be moved to the right location to be processed. 
	*	This function is called automatically by Apply() at the beginning of the process.
	*   \note For each element of TextureGrp, the function PushOptions() is called.
	*   \param _Grp -> Group of element to prepare for processing.
	*	\sa PostProcessDataTransfer(), Apply().
	*/	
	virtual 
		void PreProcessDataTransfer(TpGrpElem * _grp)=0;

	/*!
	*   \brief Update the input or output data group after before a filter process.
	*	For each element from the TextureGrp, options will checked and data will be moved to the right location. 
	*	This function is called automatically by Apply() at the end of the process.
	*   \note For each element of TextureGrp, the function PopOptions() is called.
	*   \param _Grp -> Group of element to update after processing.
	*	\sa PreProcessDataTransfer(), Apply().
	*/	
	virtual 
		void PostProcessDataTransfer(TpGrpElem * _grp)=0;
public:  
	/*! 
	*	\brief Apply a filter on a TextureGrp and store result in another TextureGrp.
	*	\param InputGrp -> source TextureGroup.
	*	\param OutputGrp -> destination TextureGroup.
	*	\return int -> status of execution
	*	\note The processing sized is defined by the output texture group parameters.
	*	\sa PostProcessDataTransfer(), PreProcessDataTransfer().
	*/
	virtual 
		int Apply(TpGrpElem * _inputGrp, TpGrpElem * _outputGrp)=0;


	/*! 
	*	\brief Applies a filter on one input Texture and stores result in another texture.
	*	\param _inputElem -> source element.
	*	\param _outputElem -> destination element.
	*	\note The processing sized is defined by the output texture group parameters.
	*	\sa Apply, PostProcessDataTransfer(), PreProcessDataTransfer().
	*	\return int -> status of execution
	*/
	virtual 
		int Apply(TpSingleElem*  _inputElem, TpSingleElem* _outputElem)=0;
};
#endif
}//namespace GCV
#endif
