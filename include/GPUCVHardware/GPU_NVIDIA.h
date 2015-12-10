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


#ifndef __GPU_NVIDIA__
#define __GPU_NVIDIA__

#include <GPUCVHardware/GenericGPU.h>
#include <nvidia/NvCpl.h>
namespace GCV{


/*!
\brief NVIDIA GPU class definition.
\sa GenericGPU
*/
class _GPUCV_HARDWARE_EXPORT_CLASS GPU_NVIDIA
	: public GenericGPU
{
public:
	/*!
	*	\note GeForce GO are mobile.
	*/
	typedef enum{
		NV_GEFORCE_FAM		= HRD_FAM_NVIDIA+0x0100,
		NV_QUADRO_FAM		= HRD_FAM_NVIDIA+0x0300,
		NV_QUADRO_FX_FAM	= HRD_FAM_NVIDIA+0x0400,
		NV_QUADRO_NVS_FAM	= HRD_FAM_NVIDIA+0x0410,
	}NVHardFamily;

	/*!
	\brief NVIDIA MotherBoard bus mode, used internally to make conversion with global bus mode GenericGPU::BusMode.
	*/
	typedef enum {
		NV_PCI          = 1,
		NV_AGP          = 4,
		NV_PCI_EXPRESS  = 8,
	} NVBusMode;
public:
	GPU_NVIDIA();
	~GPU_NVIDIA();

	virtual bool ReadBrandSettings();

	virtual bool ReadGPUSettings();
	virtual bool SetMultiGPUMode(MultiGPUMode _mode);
	virtual	GLuint	ConvertGLFormatToInternalFormat(GLuint _format, GLuint _type)const;

protected:

#ifdef _WINDOWS
	NvCplSetDataIntType m_FctSetDataInt;	//!< Pointer to write int function.
	NvCplGetDataIntType m_FctGetDataInt;	//!< Pointer to read int function.
#endif
};

_GPUCV_HARDWARE_EXPORT bool createGPU(GenericGPU * _GpuTable, int * _gpuNbr);
}//namespace GCV
#endif
