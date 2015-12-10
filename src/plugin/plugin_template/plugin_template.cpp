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
#include "StdAfx.h"
#include <GPUCV/misc.h>
#include <stdio.h>
#include <string>
#include <plugin_template/config.h>
#include <plugin_template/plugin_template.h>
//some GpuCV base files
#include <cxcoreg/cxcoreg.h>
#include <cvgcu/cvgcu.h>
//----------------------

using namespace GCV;

//=============================================
_GPUCV_GCVNPP_EXPORT_C 
LibraryDescriptor *modGetLibraryDescriptor(void)
{
	static LibraryDescriptor * pLibraryDescriptor=NULL;
	if(pLibraryDescriptor ==NULL)//init
	{
		pLibraryDescriptor = new LibraryDescriptor();
		pLibraryDescriptor->SetVersionMajor("1");	//???your Plugin Version, MAJOR
		pLibraryDescriptor->SetVersionMinor("0");	//???your Plugin Version, MINOR
		pLibraryDescriptor->SetSvnRev("570");		//???your Plugin CSN/CVS revision
		pLibraryDescriptor->SetSvnDate("");			//???
		pLibraryDescriptor->SetWebUrl(DLLINFO_DFT_URL);		//???your homepage URL
		pLibraryDescriptor->SetAuthor(DLLINFO_DFT_AUTHOR);	//???contact names
		pLibraryDescriptor->SetDllName("cvg");				//???library name
		pLibraryDescriptor->SetImplementationName("GLSL");	//???Implementation name
		pLibraryDescriptor->SetBaseImplementationID(GPUCV_IMPL_GLSL);	//???implementation base descriptor
		pLibraryDescriptor->SetUseGpu(true);							//???use GPU
	}
	return pLibraryDescriptor;
}
//=============================================
void cvgXXXAdd( CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask /*CV_DEFAULT(NULL)*/)
{
	GPUCV_START_OP(cvAdd(src1, src2, dst, mask),
		"cvgNppAdd",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	
	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");
	GCV_OPER_COMPAT_ASSERT(mask==NULL, "Operator does not support mask");

	//??? your operator BODY ???
	
	GPUCV_STOP_OP(
		cvAdd(src1, src2, dst, mask),//in case of error this operator is called
		src1, src2, mask, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
	);
}
//============================================================
void cvgXXXDilate(CvArr* src, CvArr* dst, IplConvKernel* element, int iterations )
{
	GPUCV_START_OP(cvDilate(src, dst, element, iterations),
		"cvgNppDilate",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src, 	"No input images src!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	//??? your operator BODY ???
	
	GPUCV_STOP_OP(
		cvDilate(src,dst,element,iterations),
		src, dst, NULL, NULL
		);
}
//============================================================