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
#ifndef __GPUCV_TEXTURE_COORD_H
#define __GPUCV_TEXTURE_COORD_H

#include <GPUCVTexture/config.h>

namespace GCV{

#define GPUCV_TEXT_COORD_USE_ARRAY 1

class TextureGrp;
/**
*\brief A class to store and manipulate Texture Coordinates.
*/
template<typename TplType=double>
class TextCoord
{
	friend class TextureGrp;
	friend class DataContainer;
public:
	typedef typename SGE::CL_Vector2D<TplType> TplVector2D;
	/**
	*\brief Default Constructor
	*/
	__GPUCV_INLINE
		TextCoord()
	{
		Clear();
	}

	__GPUCV_INLINE
		TextCoord(TplType _vxMin, TplType _vxMax, TplType _vyMin, TplType _vyMax)
	{
		SetCoord(_vxMin,_vxMax,_vyMin,_vyMax);
	}

	/**
	*	\brief Constructor with array of Type value.
	\param _valArray => Array of 8 values to store as texture coordinates.
	*/
	__GPUCV_INLINE
		TextCoord(const TplType * _valArray)
	{
#if GPUCV_TEXT_COORD_USE_ARRAY
		for(int i = 0; i < 8; i++)
			m_Coord[i] = _valArray[i];
#else
		Clear();
		m_Coord[0] = &_valArray[0];
		m_Coord[1] = &_valArray[2];
		m_Coord[2] = &_valArray[4];
		m_Coord[3] = &_valArray[6];
#endif
	}

	__GPUCV_INLINE
		~TextCoord()
	{}

	/**
	*	\brief Reset textures coordinates to default { {0,0}, {1,0}, {1,1}, {0,1} }
	*/
	__GPUCV_INLINE
		void Clear()
	{
#if GPUCV_TEXT_COORD_USE_ARRAY
		for(int i = 0; i < 4*2; i++)
			m_Coord[i] = GPUCVDftTextCoord[i];
#else
		m_Coord[0].Set(0,0);
		m_Coord[1].Set(1,0.);
		m_Coord[2].Set(1.,1.);
		m_Coord[3].Set(0,1);
#endif
	}


	__GPUCV_INLINE
		void Set(const TplType * _val);//???

	/**
	*	\brief Scale texture coordinates using a vector 2D.
	*	We scale the texture coordinates using the center of the four points as a reference point for the scale.
	*	\param _scale => Template 2D vector used as scaling factor.
	*/
	void Scale(TplVector2D &_scale)
	{
		SG_Assert(_scale.IsNotNull(), "TextCoord::Scale() => Scale factor is NULL");

		double delta[2]={0,0};
		int signe=1;

#if GPUCV_TEXT_COORD_USE_ARRAY
		TplType TmpCoord[8];
		for (int a = 0; a< 8; a++)
		{
			TmpCoord[a] = m_Coord[a];
		}
#else
		SGE::CL_Vector2Dd TmpCoord[4];

		for (int a = 0; a< 4; a++)
		{
			TmpCoord[a] = m_Coord[a];
		}
#endif
		_scale.x = (_scale.x - 1) /2;
		_scale.y = (_scale.y - 1) /2;

		for (int j =0; j< 2; j++)
			for (int k =0; k< 2; k++)
			{
#if GPUCV_TEXT_COORD_USE_ARRAY
				delta[k] = fabs(TmpCoord[j*2+4+k] - TmpCoord[j*2+k]);
				if (TmpCoord[j*2+k] < TmpCoord[j*2+4+k]){
					m_Coord[j*2+k]	= TmpCoord[j*2+k]		- (signe* delta[k] * _scale[k]);
					m_Coord[j*2+4+k]	= TmpCoord[j*2+4+k]	+ (signe* delta[k] *  _scale[k]);
				}else{
					m_Coord[j*2+k]	= TmpCoord[j*2+k]		 + (signe* delta[k] *  _scale[k]);
					m_Coord[j*2+4+k]	= TmpCoord[j*2+4+k]	 - (signe* delta[k] *  _scale[k]);
				}
#else
				delta[k] = fabs(TmpCoord[j+2][k] - TmpCoord[j][k]);
				if (TmpCoord[j][k] < TmpCoord[j+2][k]){
					m_Coord[j][k]	= TmpCoord[j][k]		- (signe* delta[k] * _scale[k]);
					m_Coord[j+2][k]	= TmpCoord[j+2][k]	+ (signe* delta[k] *  _scale[k]);
				}else{
					m_Coord[j][k]	= TmpCoord[j][k]		 + (signe* delta[k] *  _scale[k]);
					m_Coord[j+2][k]	= TmpCoord[j+2][k]	 - (signe* delta[k] *  _scale[k]);
				}
#endif
			}
	}

	/**
	*	\brief Set texture coo donates by giving two opposite corners coordinates.
	*	Textures coordinates are :
	\code
	m_Coord[0].Set(_vxMin,_vyMin);
	m_Coord[1].Set(_vxMax,_vyMin);
	m_Coord[2].Set(_vxMax,_vyMax);
	m_Coord[3].Set(_vxMin,_vyMax);
	\endcode
	*/
	__GPUCV_INLINE
		void SetCoord(TplType _vxMin, TplType _vxMax, TplType _vyMin, TplType _vyMax)
	{
#if GPUCV_TEXT_COORD_USE_ARRAY
		m_Coord[0] = m_Coord[6] = _vxMin;
		m_Coord[2] = m_Coord[4] = _vxMax;
		m_Coord[1] = m_Coord[3] = _vyMin;
		m_Coord[5] = m_Coord[7] = _vyMax;
#else
		m_Coord[0].Set(_vxMin,_vyMin);
		m_Coord[1].Set(_vxMax,_vyMin);
		m_Coord[2].Set(_vxMax,_vyMax);
		m_Coord[3].Set(_vxMin,_vyMax);
#endif
	}

	/**
	*	\brief Apply a horizontal flipping the coordinates.
	*/
	__GPUCV_INLINE
		void FlipH()
	{
#if GPUCV_TEXT_COORD_USE_ARRAY
		SGE::Swap(m_Coord[0], m_Coord[6]);
		SGE::Swap(m_Coord[1], m_Coord[7]);
		SGE::Swap(m_Coord[2], m_Coord[4]);
		SGE::Swap(m_Coord[3], m_Coord[5]);
#else
		SGE::Swap(m_Coord[0],m_Coord[3]);
		SGE::Swap(m_Coord[1],m_Coord[2]);
#endif
	}

	/**
	*	\brief Apply a vertical flipping the coordinates.
	*/
	__GPUCV_INLINE
		void FlipV()
	{
#if GPUCV_TEXT_COORD_USE_ARRAY
		SGE::Swap(m_Coord[0], m_Coord[2]);
		SGE::Swap(m_Coord[1], m_Coord[3]);
		SGE::Swap(m_Coord[4], m_Coord[6]);
		SGE::Swap(m_Coord[5], m_Coord[7]);
#else
		SGE::Swap(m_Coord[0],m_Coord[1]);
		SGE::Swap(m_Coord[2],m_Coord[3]);
#endif
	}

	/**
	*	\brief Apply a 90°*_time rotation clockwise.
	*	\param _time => number of time the 90° rotation is applied.
	*/
#if GPUCV_TEXT_COORD_USE_ARRAY
	__GPUCV_INLINE
		void Rotate90(const int _time)
	{
		TplType Temp;
		for (int i = 0; i< _time; i++)
		{
			Temp = m_Coord[0];
			m_Coord[0] = m_Coord[2];
			m_Coord[2] = m_Coord[4];
			m_Coord[4] = m_Coord[6];
			m_Coord[6] = Temp;
			Temp = m_Coord[1];
			m_Coord[1] = m_Coord[3];
			m_Coord[3] = m_Coord[5];
			m_Coord[5] = m_Coord[7];
			m_Coord[7] = Temp;
		}
	}
#else
	__GPUCV_INLINE
		void Rotate90(const int _time)
	{
		SGE::CL_Vector2Dd Temp = m_Coord[0];
		for (int i = 0; i< _time; i++)
		{
			Temp = m_Coord[0];
			m_Coord[0] = m_Coord[1];
			m_Coord[1] = m_Coord[2];
			m_Coord[2] = m_Coord[3];
			m_Coord[3] = Temp;
		}
	}
#endif
	/**
	*	\brief Return the corresponding Corner.
	*	\param _id => corner ID (0 => lower-left, 1 => lower-right, 2 => higher-right, 3 => Higher-left).
	*	\return The texture coordinates corresponding to the index _id.
	*/
#if GPUCV_TEXT_COORD_USE_ARRAY
	__GPUCV_INLINE
		TplType * operator [] (const int _id)
	{
		SG_Assert(_id<8, "TextCoord::operator [](i) => i must be < 4");
		SG_Assert(_id>=0, "TextCoord::operator [](i) => i must be >=0");
		return &m_Coord[_id];
	}
#else
	__GPUCV_INLINE
		TplVector2D & operator [] (const int _id)
	{
		SG_Assert(_id<4, "TextCoord::operator [](i) => i must be < 4");
		SG_Assert(_id>=0, "TextCoord::operator [](i) => i must be >=0");
		return m_Coord[_id];
	}
#endif
	/**
	*	\brief Apply selected corner simple texturing coordinates to next vertex drawn.
	*	\param _id => corner ID (0 => lower-left, 1 => lower-right, 2 => higher-right, 3 => Higher-left).
	*/
	__GPUCV_INLINE
		void	glTexCoord(GLuint _id)const
	{
#if GPUCV_TEXT_COORD_USE_ARRAY
		glTexCoord2dv(&m_Coord[_id*2]);
#else
		glTexCoord2dv(m_Coord[_id].vec);
#endif
	}

	/**
	*	\brief Apply selected corner multi-texturing coordinates to next vertex drawn.
	*	\param _textureARB_ID => corner ID (0 => lower-left, 1 => lower-right, 2 => higher-right, 3 => Higher-left).
	*	\param _coordID => corner ID (0 => lower-left, 1 => lower-right, 2 => higher-right, 3 => Higher-left).
	*/
	__GPUCV_INLINE
		void glMultiTexCoordARB(GLuint _textureARB_ID, GLuint _coordID)const
	{
#if GPUCV_TEXT_COORD_USE_ARRAY
		ProcessingGPU()->m_glExtension.__glewMultiTexCoord2dARB(_textureARB_ID, m_Coord[_coordID*2], m_Coord[_coordID*2+1]);
#else
		ProcessingGPU()->m_glExtension.__glewMultiTexCoord2dARB(_textureARB_ID, m_Coord[_coordID].x, m_Coord[_coordID].y);
#endif
	}
protected:
#if GPUCV_TEXT_COORD_USE_ARRAY
	TplType m_Coord[4*2];
#else
	TplVector2D m_Coord[4];		//!< Array of four points storing texture coordinates.
#endif
};

}//namespace GCV
#endif
