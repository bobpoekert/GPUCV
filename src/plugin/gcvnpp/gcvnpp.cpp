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
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <npp.h>
#include <gcvnpp/config.h>
#include <gcvnpp/gcvnpp.h>
#include <cxcoreg/cxcoreg.h>
#include <cvgcu/cvgcu.h>
#include <GPUCVHardware/moduleInfo.h>
using namespace GCV;

const char * GetStrNPPError(NppStatus result);

//typedef TYPE_GCV_NPP_ARITHM__SCALE_ptr 
//typedef TYPE_GCV_NPP_ARITHM__ptr
typedef void (*NPPVoidFctType)(void); 
typedef NppStatus (*TYPE_GCV_NPP_ARITHM__ptr_8U)(const Npp8u * , int , const Npp8u * , int , Npp8u * , int , NppiSize , int );
typedef NppStatus (*TYPE_GCV_NPP_ARITHM__ptr_32F)(const Npp32f * , int , const Npp32f * , int , Npp32f * , int , NppiSize);

enum GCV_NPP_TYPE
{
GCV_NPP_TYPE_8U1=0
,GCV_NPP_TYPE_8U2
,GCV_NPP_TYPE_8U3
,GCV_NPP_TYPE_8U4
,GCV_NPP_TYPE_8U=GCV_NPP_TYPE_8U1
,GCV_NPP_TYPE_16U1=GCV_NPP_TYPE_8U+4
,GCV_NPP_TYPE_16U2
,GCV_NPP_TYPE_16U3
,GCV_NPP_TYPE_16U4
,GCV_NPP_TYPE_16U=GCV_NPP_TYPE_16U1
,GCV_NPP_TYPE_32F1=GCV_NPP_TYPE_16U+4
,GCV_NPP_TYPE_32F2
,GCV_NPP_TYPE_32F3
,GCV_NPP_TYPE_32F4
,GCV_NPP_TYPE_32F=GCV_NPP_TYPE_32F1
,GCV_NPP_TYPE_64F1=GCV_NPP_TYPE_32F+4
,GCV_NPP_TYPE_64F2
,GCV_NPP_TYPE_64F3
,GCV_NPP_TYPE_64F4
,GCV_NPP_TYPE_64F=GCV_NPP_TYPE_64F1
};

//====================================
GCV_NPP_TYPE GetGCVNPPType(unsigned int depth, unsigned int channels)
{
	GCV_NPP_TYPE NppType;
	switch	(depth)
	{
		case GL_UNSIGNED_BYTE:	NppType=GCV_NPP_TYPE_8U;break;
		case GL_UNSIGNED_INT:	NppType=GCV_NPP_TYPE_16U;break;
		case GL_FLOAT:			NppType=GCV_NPP_TYPE_32F;break;
		case GL_DOUBLE:			NppType=GCV_NPP_TYPE_64F;break;
	}
	NppType = (GCV_NPP_TYPE) (NppType + (GCV_NPP_TYPE)channels-1);
	return NppType;
}
//====================================
GCV_NPP_TYPE GetGCVNPPType(CvArr* img)
{
	return GetGCVNPPType(GetGLDepth(img), GetnChannels(img));
}
//====================================
_GPUCV_GCVNPP_EXPORT_C LibraryDescriptor *modGetLibraryDescriptor(void)
{
	static LibraryDescriptor * pLibraryDescriptor=NULL;
	if(pLibraryDescriptor ==NULL)//init
	{
		pLibraryDescriptor = new LibraryDescriptor();
		pLibraryDescriptor->SetVersionMajor(SGE::ToCharStr(NPP_VERSION_MAJOR));
		pLibraryDescriptor->SetVersionMinor(SGE::ToCharStr(NPP_VERSION_MINOR));
		pLibraryDescriptor->SetSvnRev("");
		pLibraryDescriptor->SetSvnDate("");
		pLibraryDescriptor->SetWebUrl("http://www.nvidia.com/object/nvpp.htm");
		pLibraryDescriptor->SetAuthor("GpuCV Team");//author of this plugin..
		pLibraryDescriptor->SetDllName("gcvnpp");
		pLibraryDescriptor->SetImplementationName("CUDANPP");
		pLibraryDescriptor->SetBaseImplementationID(GPUCV_IMPL_CUDA);
		pLibraryDescriptor->SetUseGpu(true);
		pLibraryDescriptor->SetStartColor(CreateModColor(100,140,100,255));
		pLibraryDescriptor->SetStopColor(CreateModColor(100,255,100,255));
	}
	return pLibraryDescriptor;
}
//=============================================
//reformat image
bool ReformatImageShape(NppiSize * pSize, unsigned int *pChannels, unsigned int destChannel)
{
	if(IS_MULTIPLE_OF((*pChannels), destChannel))
	{//nop..
		return true;
	}
	else if(IS_MULTIPLE_OF(pSize->width *pSize->height * (*pChannels), destChannel))
	{
		if(IS_MULTIPLE_OF(pSize->height, destChannel))
		{
			pSize->height=(pSize->height*(*pChannels)) /destChannel;
			*pChannels = destChannel;
			return true;
		}
		else if(IS_MULTIPLE_OF(pSize->width, destChannel))
		{
			pSize->width=(pSize->width*(*pChannels)) /destChannel;
			*pChannels = destChannel;
			return true;
		}
	}
	return false;
}

//============================================================
NPPVoidFctType GetNPPFctToCall(NPPVoidFctType *pNPPSubFctList, unsigned int uiFctNbr, CvArr* arr, NppiSize *nppSize, unsigned int* nppStep)
{
	GCV_OPER_ASSERT(pNPPSubFctList,	"No NPP function list!");
	GCV_OPER_ASSERT(uiFctNbr>0,		"Empty NPP function list!");
	GCV_OPER_ASSERT(arr, 	"No input images!");
	GCV_OPER_ASSERT(nppSize,"Empty ROI size!");
	GCV_OPER_ASSERT(nppStep,"Empty image step!");


	//check that we have a function compatible
	GCV_NPP_TYPE dstNPPType = GetGCVNPPType(arr);
	SG_Assert(uiFctNbr > dstNPPType, "GCV_NPP_TYPE type out of range");
	
	
	//prepare parameters
	unsigned int depth = GetGLDepth(arr);
	unsigned int channels = GetnChannels(arr);
	nppSize->width	=	GetWidth(arr);
	nppSize->height	=	GetHeight(arr);

	

	if(pNPPSubFctList[dstNPPType]==NULL)
	{//try to find a better fct by reshaping the image and changing the channels number		
		if(!ReformatImageShape(nppSize, &channels, 4))
		{
			if(!ReformatImageShape(nppSize, &channels, 2))
			{
				nppSize->width *= channels;
				channels = 1;//channels
			}
		}
		//update GCVNPP type:
		dstNPPType = GetGCVNPPType(depth, channels);
	}
	*nppStep = nppSize->width*channels;

	//still NULL...we do not know what to do, exit with compatibility error
	GCV_OPER_COMPAT_ASSERT(pNPPSubFctList[dstNPPType], "Image destination format is not compatible with current operator.");

	return pNPPSubFctList[dstNPPType];
}
//+===============================================================
#if 0
void cvgNppAdd( CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask /*CV_DEFAULT(NULL)*/)
{

	GPUCV_START_OP(cvAdd(src1, src2, dst, mask),
		"cvgNppAdd",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	
	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");
	GCV_OPER_COMPAT_ASSERT(mask==NULL, "Operator does not support mask");
	
	void * d_src1 = NULL;
	void * d_src2 = NULL;
	void * d_mask = NULL;
	void * d_dst = NULL;

	unsigned int depth = GetCVDepth(src1);
	unsigned int channels = GetnChannels(src1);
	NppiSize nppSize;
	nppSize.width	=	GetWidth(dst);
	nppSize.height	=	GetHeight(dst);
	int step=0;
	NppStatus result;
	
	if(dst)//output is always in CUDA_BUFFER
		d_dst = gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
	if(src1)
		d_src1 = gcuPreProcess(src1, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	if(src2)
		d_src2 = gcuPreProcess(src2, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	//no mask here 	if(mask)		d_src2 = gcuPreProcess(mask, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	//===============
	

	if(depth==IPL_DEPTH_8U)
	{
		if(!ReformatImageShape(nppSize, channels, 4))
		{
			if(!ReformatImageShape(nppSize, channels, 2))
			{
				nppSize.width *= channels;
				channels = 1;//channels
			}
		}
	
		//unsigned int Pitch = gcuGetPitch(src);
		//(Pitch > 0)?Pitch:
		step = nppSize.width*channels;

		//call operator
		if(channels==1)
			result = nppiAdd_8u_C1RSfs((unsigned char*)d_src1, step, (unsigned char*)d_src2, step, (unsigned char*)d_dst, step, nppSize, 0); 
		else if(channels==4)
			result = nppiAdd_8u_C4RSfs((unsigned char*)d_src1, step, (unsigned char*)d_src2, step, (unsigned char*)d_dst, step, nppSize, 0); 
	}
	else if (depth==IPL_DEPTH_32F)
	{
		unsigned int Pitch = gcuGetPitch(src1);
		step = (Pitch > 0)?Pitch:nppSize.width*channels;

		result = nppiAdd_32f_C1R((float*)d_src1, step, (float*)d_src2, step, (float*)d_dst, step, nppSize); 
		//GCV_OPER_COMPAT_ASSERT(channels==1, "The source image must be 1 or 4 channels");	
	}
	else
	{
		GCV_OPER_COMPAT_ASSERT(0, "The source image must be 8U or 32F");
	}

	if(result!=NPP_NO_ERROR)
	{
		GPUCV_ERROR("NPP error: "<< GetStrNPPError(result));
	}
	GCV_OPER_COMPAT_ASSERT(result==NPP_NO_ERROR, "NPP error");
	

	GPUCV_STOP_OP(
		cvAdd(src1, src2, dst, mask),//in case of error this operator is called
		src1, src2, mask, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
	);

}
#endif
//============================================================
//============================================================
NPPVoidFctType NPPFctList_Add[] = {
									(NPPVoidFctType)nppiAdd_8u_C1RSfs	//8U1
									,NULL				//8U2
									,NULL				//8U3
									,(NPPVoidFctType)nppiAdd_8u_C4RSfs	//8U4
									//NOT USED,nppiAdd_8u_AC4RSfs
									,NULL	//16U1
									,NULL	//16U2
									,NULL	//16U3
									,NULL	//16U4
									,(NPPVoidFctType)nppiAdd_32f_C1R	//32F1
									,NULL	//32F2
									,NULL	//32F3
									,NULL	//32F4
									,NULL	//64F1
									,NULL	//64F2
									,NULL	//64F3
									,NULL	//64F4
									};

NPPVoidFctType NPPFctList_Sub[] = {
									(NPPVoidFctType)nppiSub_8u_C1RSfs	//8U1
									,NULL				//8U2
									,NULL				//8U3
									,(NPPVoidFctType)nppiSub_8u_C4RSfs	//8U4
									//NOT USED,nppiSub_8u_AC4RSfs
									,NULL	//16U1
									,NULL	//16U2
									,NULL	//16U3
									,NULL	//16U4
									,(NPPVoidFctType)nppiSub_32f_C1R	//32F1
									,NULL	//32F2
									,NULL	//32F3
									,NULL	//32F4
									,NULL	//64F1
									,NULL	//64F2
									,NULL	//64F3
									,NULL	//64F4
									};

void GcvNPP_TemplateArithmetic(NPPVoidFctType *pNPPSubFctList, unsigned int uiFctNbr, CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask, int scale = 0)
{
	GPUCV_START_OP(,
		"GcvNPP_TemplateArithmetic",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_COMPAT_ASSERT(mask==NULL, "Operator does not support mask yet");
	GCV_OPER_ASSERT(pNPPSubFctList,	"No NPP function list!");
	GCV_OPER_ASSERT(uiFctNbr>0,		"Empty NPP function list!");
	GCV_OPER_ASSERT(src1, 	"No input images src1!");
	GCV_OPER_ASSERT(src2, 	"No input images src2!");
	GCV_OPER_ASSERT(dst,	"No destination image!");

	//prepare data
	void * d_src1 = NULL;
	void * d_src2 = NULL;
//	void * d_mask = NULL;
	void * d_dst = NULL;

	//output is always in CUDA_BUFFER
	if(dst) d_dst	= gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
	if(src1)d_src1	= gcuPreProcess(src1, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	if(src2)d_src2	= gcuPreProcess(src2, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	//===============
	NppiSize nppSize;
	unsigned int nppStep=0;
	NppStatus nppResult;
	
	//get pointer to function to call
	NPPVoidFctType pFct = GetNPPFctToCall(pNPPSubFctList, uiFctNbr, dst, &nppSize, &nppStep);
	
	SG_Assert(pFct,"Empty function to call...");
	//call it
	switch(GetGLDepth(dst))
	{
		case GL_UNSIGNED_BYTE:	nppResult = ((TYPE_GCV_NPP_ARITHM__ptr_8U) (pFct))((unsigned char*)d_src1,	nppStep, (unsigned char*)d_src2,	nppStep, (unsigned char*)d_dst,	nppStep, nppSize, scale);break;
		case GL_FLOAT:			nppResult = ((TYPE_GCV_NPP_ARITHM__ptr_32F)(pFct))((float*)d_src1,			nppStep, (float*)d_src2,			nppStep, (float*)d_dst,			nppStep, nppSize);break;
		default:
			GCV_OPER_COMPAT_ASSERT(0, "Image destination format is not compatible with current operator.");
	}
	
	if(nppResult!=NPP_NO_ERROR)
	{
		GPUCV_ERROR("NPP error: "<< GetStrNPPError(nppResult));
	}

	if(dst)//output is always in CUDA_BUFFER
		gcuPostProcess(dst);
	if(src1)
		gcuPostProcess(src1);
	if(src2)
		gcuPostProcess(src2);
	//no mask here if(mask)		gcuPostProcess(src2);

	GCV_OPER_COMPAT_ASSERT(nppResult==NPP_NO_ERROR, "NPP error");

	GPUCV_STOP_OP(
		,//in case of error this operator is called
		src1, src2, mask, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
	);
}
//============================================================
void cvgNppAdd( CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask /*CV_DEFAULT(NULL)*/)
{
	GPUCV_START_OP(cvAdd(src1, src2, dst, mask),
		"cvgNppAdd",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GcvNPP_TemplateArithmetic(NPPFctList_Add, sizeof(NPPFctList_Add)/sizeof(NPPVoidFctType), src1, src2, dst, mask);

	GPUCV_STOP_OP(
		cvAdd(src1, src2, dst, mask),//in case of error this operator is called
		src1, src2, mask, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
	);
}
//================================================================================
void cvgNppSub( CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask /*CV_DEFAULT(NULL)*/)
{
	GPUCV_START_OP(cvSub(src1, src2, dst, mask),
		"cvgNppSub",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GcvNPP_TemplateArithmetic(NPPFctList_Sub, sizeof(NPPFctList_Sub)/sizeof(NPPVoidFctType), src1, src2, dst, mask);

	GPUCV_STOP_OP(
		cvAdd(src1, src2, dst, mask),//in case of error this operator is called
		src1, src2, mask, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
	);
}
//================================================================================
void cvgNppDiv( CvArr* src1, CvArr* src2, CvArr* dst, double scale CV_DEFAULT(1))
{
	GPUCV_START_OP(cvDiv(src1, src2, dst, mask),
		"cvgNppSub",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_COMPAT_ASSERT(scale==1, "Operator does not support scale factor");

	GcvNPP_TemplateArithmetic(NPPFctList_Sub, sizeof(NPPFctList_Sub)/sizeof(NPPVoidFctType), src1, src2, dst, NULL);

	GPUCV_STOP_OP(
		cvAdd(src1, src2, dst, mask),//in case of error this operator is called
		src1, src2, mask, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
	);
}
//================================================================================
void cvgNppMul( CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask /*CV_DEFAULT(NULL)*/)
{
	GPUCV_START_OP(cvSub(src1, src2, dst, mask),
		"cvgNppSub",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GcvNPP_TemplateArithmetic(NPPFctList_Sub, sizeof(NPPFctList_Sub)/sizeof(NPPVoidFctType), src1, src2, dst, mask);

	GPUCV_STOP_OP(
		cvAdd(src1, src2, dst, mask),//in case of error this operator is called
		src1, src2, mask, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
	);
}
//================================================================================
void cvgNppDilate(CvArr* src, CvArr* dst, IplConvKernel* element, int iterations )
{
#if 0
	GPUCV_START_OP(cvDilate(src, dst, element, iterations),
		"cvgNppDilate",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(src, 	"No input images src!");
	GCV_OPER_ASSERT(dst,	"No destination image!");
	void * d_src = NULL;
	void * d_dst = NULL;

	unsigned int depth		= GetCVDepth(src);
	unsigned int channels	= GetnChannels(src);
	GCV_OPER_COMPAT_ASSERT(depth==IPL_DEPTH_8U, "The source image must be 8U");
	GCV_OPER_COMPAT_ASSERT(channels ==1 || channels==4, "Image must be single or 4 channels");

	NppiSize nppSize, maskSize;
	nppSize.width	=	GetWidth(dst);
	nppSize.height	=	GetHeight(dst);
	int step=0;
	NppStatus result;
	
	if(dst)//output is always in CUDA_BUFFER
		d_dst = gcuPreProcess(dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
	if(src)
		d_src = gcuPreProcess(src, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	//===============
	
	//set parameters
	int MaskfullSize = element->nCols * element->nRows;
	maskSize.width = element->nCols;
	maskSize.height = element->nRows;
	unsigned char *mask = new unsigned char [MaskfullSize];
	for(int i=0;i< MaskfullSize;i++)
		mask[i] = (unsigned char)(element->values[i]);
	 NppiPoint anchor;
	anchor.x = 0.5 * (element->nCols -1);
	anchor.y = 0.5 * (element->nRows -1);
	//================
	
	
	
	//can not reformat here cause we use neighborhood
	unsigned int Pitch = gcuGetPitch(src);
	step = (Pitch > 0)?Pitch:nppSize.width*channels;

	//call operator
	if(channels==1)
		result = nppiDilate_8u_C1R((unsigned char*)d_src, step, (unsigned char*)d_dst, step, nppSize, mask, maskSize, anchor); 
	else if(channels==4)
		result = nppiDilate_8u_C4R((unsigned char*)d_src, step, (unsigned char*)d_dst, step, nppSize, mask, maskSize, anchor); 
	
	if(result!=NPP_NO_ERROR)
	{
		GPUCV_ERROR("NPP error: "<< GetStrNPPError(result));
	}
	GCV_OPER_COMPAT_ASSERT(result==NPP_NO_ERROR, "NPP error");
	
	GPUCV_STOP_OP(
		cvDilate(src,dst,element,iterations),
		src, dst, NULL, NULL
		);
#endif
}
//============================================================
void cvgNppCanny(CvArr* image, CvArr* edges, double threshold1, double threshold2, int aperture_size/*=3*/ )
{
#if 0
	GPUCV_START_OP(cvCanny(image, edges, threshold1, threshold2),
		"cvgNppCanny",
		dst,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_ASSERT(image, 	"No input images src!");
	GCV_OPER_ASSERT(edges,	"No destination image!");
	void * d_src = NULL;
	void * d_dst = NULL;
	void * d_x = NULL;
	void * d_y = NULL;

	unsigned int depth		= GetCVDepth(image);
	unsigned int channels	= GetnChannels(image);
	//GCV_OPER_COMPAT_ASSERT(depth==IPL_DEPTH_32F, "The source image must be 32F");
	GCV_OPER_COMPAT_ASSERT(channels==1, "Image must be single channels");

	NppiSize nppSize, maskSize;
	nppSize.width	=	GetWidth(edges);
	nppSize.height	=	GetHeight(edges);
	int step=0;
	NppStatus result;
	

	cvgSynchronize(image);
	IplImage *dx = cvgCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);
	IplImage *dy = cvgCreateImage(cvGetSize(image), IPL_DEPTH_32F, 1);

	//set option so cuda bufer are not mapped to OpenGL buffer.
	//dx
	CvgArr * pImg = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(dx));	
	pImg->GetDataDsc<DataDsc_CUDA_Buffer>()->SetAutoMapGLBuff(false);
	//dy
	pImg = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(dy));	
	pImg->GetDataDsc<DataDsc_CUDA_Buffer>()->SetAutoMapGLBuff(false);

	//perform sobel...
	cvgCudaSobel( image, dx, 1, 0, aperture_size );
    cvgCudaSobel( image, dy, 0, 1, aperture_size );



	if(dx)//output is always in CUDA_BUFFER
		d_x = gcuPreProcess(dx, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	if(dy)//output is always in CUDA_BUFFER
		d_y = gcuPreProcess(dy, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	if(edges)//output is always in CUDA_BUFFER
		d_dst = gcuPreProcess(edges, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
//	if(image)
//		d_src = gcuPreProcess(image, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	//===============
	//alloc temp buffer

	int tmpBufferStep=0;
	nppiCannyGetSize(nppSize, &tmpBufferStep);
	Npp8u* tmpBuffer = NULL;
	cudaMalloc((void**)&tmpBuffer, tmpBufferStep);


	//can not reformat here cause we use neighborhood
	unsigned int PitchX = gcuGetPitch(dx);
	unsigned int PitchY = gcuGetPitch(dy);
	unsigned int PitchDst = gcuGetPitch(edges);
	
	unsigned int stepX = (PitchX > 0)?PitchX:	nppSize.width*channels*sizeof(float);
	unsigned int stepY = (PitchY > 0)?PitchY:	nppSize.width*channels*sizeof(float);
	unsigned int stepDst = (PitchDst > 0)?PitchDst:	nppSize.width*channels;

	//call operator
	result = nppiCanny_32f8u_C1R((float*)d_x, stepX, 
								(float*)d_y, stepY,
								(unsigned char*)d_dst, stepDst,// nppSize.width, 
								nppSize, 
								threshold1, threshold2,
								tmpBuffer); 
	
	if(result!=NPP_NO_ERROR)
	{
		GPUCV_ERROR("NPP error: "<< GetStrNPPError(result));
	}
	GCV_OPER_COMPAT_ASSERT(result==NPP_NO_ERROR, "NPP error");
	
	GPUCV_STOP_OP(
		cvCanny(image, edges, threshold1, threshold2),
		src, dst, NULL, NULL
		);
#endif
}


#define ADD_NPP_ERROR_CASE_MSG(VAL, MSG) case VAL: return #VAL##MSG;break;

#define ADD_NPP_ERROR_CASE(VAL)case VAL: return #VAL;break;

const char * GetStrNPPError(NppStatus result)
{
	switch(result)
	{
		ADD_NPP_ERROR_CASE(NPP_NOT_SUPPORTED_MODE_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_ROUND_MODE_NOT_SUPPORTED_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_RESIZE_NO_OPERATION_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_BAD_ARG_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_LUT_NUMBER_OF_LEVELS_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_TEXTURE_BIND_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_COEFF_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_RECT_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_QUAD_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_WRONG_INTERSECTION_ROI_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_NOT_EVEN_STEP_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_INTERPOLATION_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_RESIZE_FACTOR_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_MEMFREE_ERR);
		ADD_NPP_ERROR_CASE(NPP_MEMSET_ERR); 	
		ADD_NPP_ERROR_CASE(NPP_MEMCPY_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_MEM_ALLOC_ERR); 	
		ADD_NPP_ERROR_CASE(NPP_HISTO_NUMBER_OF_LEVELS_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_MIRROR_FLIP_ERR); 	
		ADD_NPP_ERROR_CASE(NPP_INVALID_INPUT); 	
		ADD_NPP_ERROR_CASE(NPP_ALIGNMENT_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_STEP_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_SIZE_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_POINTER_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_NULL_POINTER_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_CUDA_KERNEL_EXECUTION_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_NOT_IMPLEMENTED_ERROR);	
		ADD_NPP_ERROR_CASE(NPP_ERROR); 	
	//	ADD_NPP_ERROR_CASE_MSG(NPP_NO_ERROR, "Error free operation.");
		ADD_NPP_ERROR_CASE_MSG(NPP_SUCCESS, "Successful operation (same as NPP_NO_ERROR).");
		ADD_NPP_ERROR_CASE(NPP_WARNING); 	
		ADD_NPP_ERROR_CASE(NPP_WRONG_INTERSECTION_QUAD_WARNING);
		ADD_NPP_ERROR_CASE_MSG(NPP_MISALIGNED_DST_ROI_WARNING, "Speed reduction due to uncoalesced memory accesses warning.");
		ADD_NPP_ERROR_CASE_MSG(NPP_AFFINE_QUAD_INCORRECT_WARNING, " 	Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties. First 3 vertices are used, the fourth vertex discarded.");
//		ADD_NPP_ERROR_CASE_MSG(NPP_AFFINE_QUAD_CHANGED_WARNING, " 	Alias for NPP_AFFINE_QUAD_INCORRECT_WARNING resembling the IPP warning more closely.");
		ADD_NPP_ERROR_CASE_MSG(NPP_ADJUSTED_ROI_SIZE_WARNING, " 	Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing.");
//		ADD_NPP_ERROR_CASE_MSG(NPP_DOUBLE_SIZE_WARNING, " 	Alias for the ADJUSTED_ROI_SIZE_WARNING, this name is closer to IPP's original warning enum.");
		ADD_NPP_ERROR_CASE_MSG(NPP_ODD_ROI_WARNING, " 	Indicates that for 422/411/420 sampling the ROI width/height was forced to even value. ");
	default: return "Unknown NPP error";
	}

}