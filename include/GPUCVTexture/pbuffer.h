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



#ifndef __GPUCV_TEXTURE_PBUFFERS_H
#define __GPUCV_TEXTURE_PBUFFERS_H

#include "GPUCVTexture/config.h"
namespace GCV{

/** @addtogroup RENDERTOTEXT_GRP 
*  @{
*/


/*!
*	\brief PBuffer is a mechanism to render OpenGL scene directly into texture.
*	\author NVIDIA
*/      
class _GPUCV_TEXTURE_EXPORT PBuffer
{
public:

	/*! 
	*	\brief PBuffer constructor.
	*	\param strMode -> char * containing the following parameters :
	*	The pixel format for the pbuffer is controlled by the mode string passed
	*	into the PBuffer constructor. This string can have the following attributes:
	*
	<ul>
	* <li>r					-> r pixel format (for float buffer).
	* </li><li>rg			-> rg pixel format (for float buffer).
	* </li><li>rgb          -> rgb pixel format. 8 bit or 16/32 bit in float buffer mode
	* </li><li>rgba         -> same as "rgb alpha" string
	* </li><li>alpha        -> must have alpha channel
	* </li><li>depth        -> must have a depth buffer
	* </li><li>depth=n      -> must have n-bit depth buffer
	* </li><li>stencil      -> must have a stencil buffer
	* </li><li>double       -> must support double buffered rendering
	* </li><li>samples=n    -> must support n-sample antialiasing (n can be 2 or 4)
	* </li><li>float=n      -> must support n-bit per channel floating point
	* 
	* </li><li>texture2D
	* </li><li>textureRECT
	* </li><li>textureCUBE<ul>
	*	<li>- must support binding pbuffer as texture to specified target
	* </li><li>- binding the depth buffer is also supporting by specifying
	*                '=depth' like so: texture2D=depth or textureRECT=depth
	* </li><li>- the internal format of the texture will be rgba by default or
	*                float if pbuffer is floating point
	* </li>
	* </li></ul>
	</ul>
	* \param managed -> set managed to true if you want the class to cleanup OpenGL objects in destructor.
	*/
	PBuffer(const char *strMode, bool managed = false);

	~PBuffer();

	/*! 
	* \brief This function actually does the creation of the p-buffer. It can only be called once a window has already been created.
	* \param iWidth -> width of the PBuffer
	* \param iHeight -> height of the PBuffer
	* \param bShareContexts -> [MORE_HERE]
	* \param bShareObjects -> [MORE_HERE]
	* \return TRUE is success, else FALSE.
	*/
	bool Initialize(int iWidth, int iHeight, bool bShareContexts, bool bShareObjects);


	void Destroy();

	void Activate(PBuffer *current = NULL); // to switch between pbuffers, pass active pbuffer as argument
	void Deactivate();

	//#if defined(_WINDOWS)
	int Bind(int iBuffer);
	int Release(int iBuffer);
	/*! 
	* \brief Check to see if the pbuffer was lost. If it was lost, destroy it and then recreate it.
	*/
	void HandleModeSwitch();
	//#endif

	/*!
	*	\brief Get total size in bytes of the PBuffer.
	*	\return total size in bytes of the PBuffer.
	*/
	unsigned int GetSizeInBytes();

	/*!
	*	\brief Make a copy the entire PBuffer in the memory. 
	*	Make a copy the entire PBuffer in the memory.
	*	If ever you want to read a smaller size : specify it through w,h. otherwise w=h=-1
	*	\param ptr -> Pointer to an allocated buffer
	*	\param w -> width of the buffer.
	*	\param h -> width of the buffer.
	*	\return Number of pixels copied.
	*/
	unsigned int CopyToBuffer(void *ptr, int w=-1, int h=-1);

	__GPUCV_INLINE int GetNumComponents()
	{ return m_iNComponents; }

	__GPUCV_INLINE int GetBitsPerComponent()
	{ return m_iBitsPerComponent; }

	__GPUCV_INLINE int GetWidth()
	{ return m_iWidth; }

	__GPUCV_INLINE int GetHeight()
	{ return m_iHeight; }

	__GPUCV_INLINE bool IsSharedContext()
	{ return m_bSharedContext; }

#if defined(_WINDOWS)
	__GPUCV_INLINE bool IsTexture()
	{ return m_bIsTexture; }
#endif

protected:
#if defined(_WINDOWS)
	HDC         m_hDC;		//!< Handle to a device context.
	HGLRC       m_hGLRC;	//!< Handle to a GL context.
	HPBUFFERARB m_hPBuffer;	//!< Handle to a pbuffer.

	HGLRC       m_hOldGLRC;
	HDC         m_hOldDC;

	std::vector<int> m_pfAttribList;
	std::vector<int> m_pbAttribList;

	bool m_bIsTexture;
#elif defined(_LINUX)
	Display    *m_pDisplay;
	GLXPbuffer  m_glxPbuffer;
	GLXContext  m_glxContext;

	Display    *m_pOldDisplay;
	GLXPbuffer  m_glxOldDrawable;
	GLXContext  m_glxOldContext;

	std::vector<int> m_pfAttribList;
	std::vector<int> m_pbAttribList;
#elif defined(MACOS)
	AGLContext  m_context;
	WindowPtr   m_window;
	std::vector<int> m_pfAttribList;
#endif

	int m_iWidth;
	int m_iHeight;
	int m_iNComponents;
	int m_iBitsPerComponent;

	const char *m_strMode;
	bool m_bSharedContext;
	bool m_bShareObjects;

private:
	std::string getStringValue(std::string token);
	int getIntegerValue(std::string token);
#if defined(_LINUX) || defined(_WINDOWS)
	void parseModeString(const char *modeString, std::vector<int> *pfAttribList, std::vector<int> *pbAttribList);

	bool m_bIsBound;
	bool m_bIsActive;
	bool m_bManaged;
#endif
};

class _GPUCV_TEXTURE_EXPORT PBufferManager
{
public : 
	PBuffer *princ;
	PBuffer *sec;
	int  active;
	bool unique_context;

public : 
	PBufferManager(const char *strMode, bool managed = false);

	void ActivateFirst(GLuint tex_bind = 0);
	void ForceFirst(GLuint tex_bind = 0);
	void ActivateSecond(GLuint tex_bind = 0);
	void ForceSecond(GLuint tex_bind = 0);
	void Desactivate();

	int  IsActive();
	bool IsUnique();
	void SetUniqueContext();
	void UnsetUniqueContext();
};

PBufferManager * PBuffersManager();
/** @} */ // end of RENDERTOTEXT_GRP
}//namespace GCV
#endif
