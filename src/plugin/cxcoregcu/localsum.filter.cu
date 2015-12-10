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
#include "config.h"
#include <typeinfo> 
#include <GPUCVCuda/gcu_runtime_api_wrapper.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include "localsum.filter.h"


_GPUCV_CXCOREGCU_EXPORT_CU              
void gcuLocalSum(CvArr* src1,CvArr* dst,int h,int w)
{
	void* d_dst = gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE,NULL);
	void* d_src1 = gcuPreProcess(src1, GCU_INPUT, CU_MEMORYTYPE_DEVICE,NULL);

	int s_width = gcuGetWidth(dst);
	int s_height= gcuGetWidth(dst);

	size_t pitch = (uint)gcuGetPitch(dst)/(sizeof(int1));
	dim3 threads(16,16,1);
	dim3 blocks = dim3(iDivUp(s_width, threads.x), 
		iDivUp(s_height,threads.y), 1);

	LocalSumKernel<<<blocks, threads>>>((int1 *)d_src1,(int1 *)d_dst,h,w,s_width,s_height,pitch);

	
	gcudaThreadSynchronize();

	gcuPostProcess(dst);
	gcuPostProcess(src1);
}
