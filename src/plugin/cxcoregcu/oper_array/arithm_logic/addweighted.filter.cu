//CVG_LicenseBegin========================================== ====================
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
#include <cxcoregcu/oper_array/arithm_logic/arithm_logic.h>

#if _GPUCV_COMPILE_CUDA
#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <cxcoregcu/oper_array/arithm_logic/addweighted.filter.h>



//=====================================================
_GPUCV_CXCOREGCU_EXPORT_CU
void gcuAddWeighted_f4(CvArr* src1,float4 fS1,CvArr* src2,float4 fS2,float4 fGamma,CvArr* dst)
{
	//params
	unsigned int width		= gcuGetWidth(src1);
	unsigned int height		= gcuGetHeight(src1);
	unsigned int channels	= gcuGetnChannels(src1);
	unsigned int depth		= gcuGetGLDepth(src1);
	//------------------

  	void* d_dst = gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE,NULL);
	void* d_src1 = gcuPreProcess(src1, GCU_INPUT, CU_MEMORYTYPE_DEVICE,NULL);
	void* d_src2 = gcuPreProcess(src2, GCU_INPUT, CU_MEMORYTYPE_DEVICE,NULL);
  
	unsigned int NewChannels = channels;
#if 0//Reshape 
	if(IS_MULTIPLE_OF(width *channels, 4))
		NewChannels = 4;
	else if(IS_MULTIPLE_OF(width *channels, 2))
		NewChannels = 2;

	//try reshaping	
	if (NewChannels != channels)
	{
		//refresh size
		width = (width * channels)/NewChannels;
	}
#endif

	dim3 threads(16,16,1);
	dim3 blocks = dim3(iDivUp(width,threads.x), iDivUp(height,threads.y), 1);

	CUT_CHECK_ERROR("Kernel execution could not start");

#define GCUADDW_SWITCH_FCT(CHANNELS, DST_TYPE, SRC_TYPE)\
	{uint pitch = gcuGetPitch(dst)/(sizeof(DST_TYPE)*CHANNELS);\
	gcudaKernel_AddWeighted<CHANNELS,SRC_TYPE,DST_TYPE, SRC_TYPE##CHANNELS,DST_TYPE##CHANNELS> <<<blocks, threads>>> \
	((SRC_TYPE##CHANNELS*) d_src1,(float4)fS1,(SRC_TYPE##CHANNELS*) d_src2,(float4)fS2,(float4)fGamma,(DST_TYPE##CHANNELS*)d_dst,width,height,pitch);}

	GCU_MULTIPLEX_ALLCHANNELS_ALLFORMAT(GCUADDW_SWITCH_FCT, NewChannels, depth);

	//kernel executed... 
	CUT_CHECK_ERROR("Kernel execution failed");

	gcuPostProcess(dst);
	gcuPostProcess(src1);
	gcuPostProcess(src2);
}
//=====================================================
_GPUCV_CXCOREGCU_EXPORT_CU
void gcuAddWeighted_d(CvArr* src1,double s1,CvArr* src2,double s2,double gamma,CvArr* dst)
{
	//set float values to float4
	float4 fS1, fS2, fGamma;
	fS1.x = fS1.y = fS1.z = fS1.w = s1;
	fS2.x = fS2.y = fS2.z = fS2.w = s2;
	fGamma.x = fGamma.y = fGamma.z = fGamma.w = gamma;

	gcuAddWeighted_f4(src1,fS1,src2,fS2,fGamma,dst);
}
//=====================================================
_GPUCV_CXCOREGCU_EXPORT_CU
void gcuAddWeighted_scalar(CvArr* src1,gcuScalar s1,CvArr* src2,gcuScalar s2,gcuScalar gamma,CvArr* dst)
{
	//set float values to float4
	float4 fS1, fS2, fGamma;
	fS1.x = s1.val[0];
	fS1.y = s1.val[1];
	fS1.z = s1.val[2];
	fS1.w = s1.val[3];
	fS2.x = s2.val[0];
	fS2.y = s2.val[1];
	fS2.z = s2.val[2];
	fS2.w = s2.val[3];
	fGamma.x = gamma.val[0];
	fGamma.y = gamma.val[1];
	fGamma.z = gamma.val[2];
	fGamma.w = gamma.val[3];

	gcuAddWeighted_f4(src1,fS1,src2,fS2,fGamma,dst);
}
//=====================================================
#endif
