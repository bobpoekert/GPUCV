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
#ifndef __GPU_ATI__
#define __GPU_ATI__

#include <GPUCVHardware/GenericGPU.h>
namespace GCV{

/**
\brief ATI/AMD GPU class definition.
\todo Add support for CrossFire
\sa GenericGPU
\todo Check for ATI hardware compatibilities
*/
class _GPUCV_HARDWARE_EXPORT_CLASS GPU_ATI
	: public GenericGPU
{
	//enum
public:
	/*!
	\brief ATI/AMD MotherBoard bus mode, used internally to make conversion with global bus mode GenericGPU::BusMode.
	*/

	typedef enum {
		ATI_PCI          = 1,
		ATI_AGP          = 4,
		ATI_PCI_EXPRESS  = 8,
	} ATIBusMode;

	/*!
	\note Radeon MOB are mobile.
	*/
	typedef enum{
		ATI_RADEON_FAM			= HRD_FAM_ATI+0x0100,
		ATI_RADEON_X_FAM		= HRD_FAM_ATI+0x0200,
		ATI_RADEON_HD_FAM		= HRD_FAM_ATI+0x0400,
		ATI_FIREGL_FAM			= HRD_FAM_ATI+0x0800,
	}ATIHardFamily;

public:
	GPU_ATI();
	~GPU_ATI();

	virtual bool ReadBrandSettings();
	virtual bool ReadGPUSettings();
	virtual bool SetMultiGPUMode(MultiGPUMode _mode);

protected:
};

#if _GPUCV_SUPPORT_GPU_PLUGIN
__declspec(dllexport) bool createGPU(GenericGPU * _GpuTable, int * _gpuNbr);
#endif
}//namespace GCV
#endif
