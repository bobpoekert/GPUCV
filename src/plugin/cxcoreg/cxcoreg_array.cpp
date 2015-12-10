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
#include <GPUCVTexture/TextureTemp.h>
#include <GPUCV/misc.h>
#include <cxcoreg/cxcoreg.h>
#include <highguig/highguig.h>
#include <GPUCVHardware/moduleInfo.h>

/** \file cxcoreg_array.cpp
\author Yannick Allusse
\brief Contain the GPUCV function correspondance of cxarray.cpp

Basic data management, object creation, data allocation
correspond in Cvcore documentation to
"Operations on Arrays"
*/

using namespace GCV;

_GPUCV_CXCOREG_EXPORT_C LibraryDescriptor *modGetLibraryDescriptor(void)
{
	static LibraryDescriptor * pLibraryDescriptor=NULL;
	if(pLibraryDescriptor ==NULL)//init
	{
		pLibraryDescriptor = new LibraryDescriptor();
		pLibraryDescriptor->SetVersionMajor("1");
		pLibraryDescriptor->SetVersionMinor("0");
		pLibraryDescriptor->SetSvnRev("570");
		pLibraryDescriptor->SetSvnDate("");
		pLibraryDescriptor->SetWebUrl(DLLINFO_DFT_URL);
		pLibraryDescriptor->SetAuthor(DLLINFO_DFT_AUTHOR);
		pLibraryDescriptor->SetDllName("cxcoreg");
		pLibraryDescriptor->SetImplementationName("GLSL");
		pLibraryDescriptor->SetBaseImplementationID(GPUCV_IMPL_GLSL);
		pLibraryDescriptor->SetUseGpu(true);
		pLibraryDescriptor->SetStartColor(GPUCV_IMPL_GLSL_COLOR_START);
		pLibraryDescriptor->SetStopColor(GPUCV_IMPL_GLSL_COLOR_STOP);
	}
	return pLibraryDescriptor;
}


_GPUCV_CXCOREG_EXPORT_C int cvgDLLInit(bool InitGLContext, bool isMultiThread)
{
	return GpuCVInit(InitGLContext, isMultiThread);
}


/*
//http://msdn.microsoft.com/en-us/library/ms682583.aspx
BOOL WINAPI DllMain(
					__in  HINSTANCE hinstDLL,
					__in  DWORD fdwReason,
					__in  LPVOID lpvReserved
					)
{
	if(fdwReason==DLL_PROCESS_ATTACH)//DLL is loading
	{
		if(lpvReserved==NULL)//dynamic Load
	}
	else if (fdwReason==DLL_PROCESS_DETACH)//DLL is being freed
	{

	}

}*/


//___________________________________________________
//___________________________________________________
//___________________________________________________
//===================================================
//=>start CVGXCORE_OPER_ARRAY_INIT_GRP
//===================================================
IplImage* cvgCreateImage( CvSize size, int depth, int channels )
{
	IplImage * tmp=NULL;
	GPUCV_START_OP(return cvCreateImage(size, depth, channels),
		"cvgCreateImage",
		(IplImage*)NULL,
		GenericGPU::HRD_PRF_0);

	//profile image creation
#if 0//_GPUCV_PROFILE
	if(GET_GPUCV_OPTION(GpuCVSettings::GPUCV_SETTINGS_PROFILING_OPER))
	{
		std::string Size="=";\
			Size+= SGE::ToCharStr(size.width) + "*";
		Size+= SGE::ToCharStr(size.height);
		_PROFILE_PARAMS->AddChar("size", Size.data());
		Size= SGE::ToCharStr(depth);
		_PROFILE_PARAMS->AddChar("depth", Size.data());
		Size= SGE::ToCharStr(channels);
		_PROFILE_PARAMS->AddChar("nChannels", Size.data());
	}
#endif

	tmp = cvCreateImage( size, depth, channels );
	CvgArr * newCvgImage= new CvgArr(&tmp);

	//look if image is already in the manager?
	//this can happen if a previous image at the same memory location has not been released!!
	CvgArr * tmp2=(CvgArr *)GetTextureManager()->Find(tmp);
	if(tmp2)
	{
		tmp2 = tmp2;
		GetTextureManager()->PrintAllObjects();
	}
	SG_Assert(!tmp2, "Application is creating a new IplImage, but there is already one IplImage with the same pointer in the GpuCV Image Manager. You may have forgotten to release a previous IplImage");
	GetTextureManager()->AddObj(newCvgImage);

	GPUCV_DEBUG(newCvgImage->GetValStr() << ">" << FctName);

	GPUCV_STOP_OP(
		return cvCreateImage(size, depth, channels),
		NULL, NULL, NULL, NULL
		);
	return tmp;
}
//===================================================
void cvgReleaseImage(IplImage **img)
{
	GPUCV_START_OP(cvReleaseImage(img),
		"cvgReleaseImage",
		*img,
		GenericGPU::HRD_PRF_0);

	if(img==NULL || *img==NULL)
		GPUCV_WARNING("Trying to release empty pointer");

	GPUCV_DEBUG("ID:IPL:"<<*img<<" - "<< cvgGetLabel(*img)<< ">" << FctName);
	if (*img!=NULL)
	{
		if(!GetTextureManager()->Delete(*img))
		{
			//cvReleaseImage(img);// it is not working when image has no data
			IplImage* tempimg = *img;
			if(tempimg->imageData)
				cvReleaseData( tempimg );
			cvReleaseImageHeader( &tempimg );
		}
		*img = NULL;
		/*if(img)
		cvReleaseImage((IplImage**)img);
		else
		GPUCV_WARNING("cvgReleaseImage() :: IplImage has be corrupted(freed) in GPUCV");
		*/
	}
	else
		GPUCV_WARNING("cvgReleaseImage() *img==NULL");

	GPUCV_STOP_OP(
		cvReleaseImage(img),
		NULL, NULL, NULL, NULL
		);
}
//===================================================
IplImage* cvgCloneImage(IplImage *img)
{
	IplImage * Cloned = NULL;
	GPUCV_START_OP(return cvCloneImage(img),
		"cvgCloneImage",
		img,
		GenericGPU::HRD_PRF_0);

	CvgArr * src = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(img));
	SG_Assert(src, "Could not retrieve source image");
	CvgArr * dst = new CvgArr(*src);
	GetTextureManager()->AddObj(dst);
	Cloned =  dst->GetIplImage();

	GPUCV_STOP_OP(
		return cvCloneImage(img),
		img, NULL, NULL, NULL
		);
	return Cloned;
}
//===================================================
#if _GPUCV_SUPPORT_CVMAT
CvMat* cvgCreateMat( int rows, int cols, int type )
{
	CvMat * tmp = NULL;
	GPUCV_START_OP(return cvCreateMat(rows, cols, type),
		"cvgCreateMat",
		(CvMat*)NULL,
		GenericGPU::HRD_PRF_0);


	tmp = cvCreateMat( rows, cols, type);
	CvgArr * newCvgMat= new CvgArr(&tmp);
	GetTextureManager()->AddObj(newCvgMat);


	GPUCV_STOP_OP(
		return cvCreateMat(rows, cols, type),
		NULL, NULL, NULL, NULL
		);
	return tmp;
}
//===================================================
void cvgReleaseMat(CvMat **mat)
{
	GPUCV_START_OP(cvReleaseMat(mat),
		"cvgReleaseMat",
		(CvMat*)NULL,
		GenericGPU::HRD_PRF_0);

	if (*mat!=NULL)
	{
		if(!GetTextureManager()->Delete(*mat))
			cvReleaseMat(mat);
		*mat=NULL;
	}
	else
		GPUCV_WARNING("cvgReleaseMat() *mat==NULL");

	GPUCV_STOP_OP(
		cvReleaseMat(mat),
		NULL, NULL, NULL, NULL
		);
}
//===================================================
CvMat* cvgCloneMat(CvMat* mat )
{
	CvMat * Cloned=NULL;
	GPUCV_START_OP(return cvgCloneMat(mat),
		"cvgCloneMat",
		(CvMat*)NULL,
		GenericGPU::HRD_PRF_0);

	CvgArr * src = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(mat));
	SG_Assert(src, "Could not retrieve source matrix");
	CvgArr * dst = new CvgArr(*src);//copy constructor
	GetTextureManager()->AddObj(dst);
	Cloned= dst->GetCvMat();

	GPUCV_STOP_OP(
		return cvgCloneMat(mat),
		mat, NULL, NULL, NULL
		);
	return Cloned;
}
#endif
//===================================================
void cvgSetData( CvArr* arr, void* data, int step )
{
	GPUCV_START_OP(cvSetData(arr, data, step),
		"cvgSetData",
		arr,
		GenericGPU::HRD_PRF_0);

	//get back DataDsc_IplImage
	CvgArr * gpuArr = dynamic_cast <CvgArr*>(GPUCV_GET_TEX(arr));
	gpuArr->SetLocation<DataDsc_CPU>(false);
	//affect data
	cvSetData(arr, data, step);
	//set Data flag
	gpuArr->SetDataFlag<DataDsc_CPU>(true, true);

	GPUCV_STOP_OP(
		cvSetData(arr, data, step),
		arr, NULL, NULL, NULL
		);
}
//===================================================
void cvgGetRawData(CvArr* arr, uchar** data, int* step/*=NULL*/, CvSize* roi_size/*=NULL*/ )
{
	GPUCV_START_OP(cvGetRawData(arr, data, step, roi_size),
		"cvgGetRawData",
		arr,
		GenericGPU::HRD_PRF_0);

	cvgSynchronize(arr);
	cvGetRawData(arr, data, step, roi_size);

	GPUCV_STOP_OP(
		cvGetRawData(arr, data, step, roi_size),
		arr, NULL, NULL, NULL
		);
}

//#endif//test
//#if 0//test


//===================================================
//===================================================
//=>stop CVGXCORE_OPER_ARRAY_INIT_GRP
//===================================================
//___________________________________________________
//___________________________________________________
//___________________________________________________
//===================================================
//=>start CVGXCORE_OPER_ARRAY_ACCESS_ELEM_GRP
//===================================================


//===================================================
//=>stop CVGXCORE_OPER_ARRAY_ACCESS_ELEM_GRP
//===================================================
//___________________________________________________
//___________________________________________________
//___________________________________________________
//===================================================
//=>start CVGXCORE_OPER_COPY_FILL_GRP
//===================================================
void cvgCopy(CvArr* src, CvArr* dst , CvArr* mask/* = NULL*/)
{
	GPUCV_START_OP(cvCopy(src, dst, mask),
		"cvgCopy",
		dst,
		GenericGPU::HRD_PRF_0);

	TemplateOperator("cvgCopy", "FShaders/copy", "",
		src, NULL, mask,
		dst, NULL, 0,
		TextureGrp::TEXTGRP_SAME_ALL, (mask)?"$DEF_MASK=1//":"");

	GPUCV_STOP_OP(
		cvCopy(src, dst, mask),
		src, dst, mask, NULL
		);
}
//===================================================
void cvgSetZero( CvArr* arr )
{
	GPUCV_START_OP(cvSetZero(arr),
		"cvgSetZero",
		arr,
		GenericGPU::HRD_PRF_0);

	CvgArr * gpuArr = dynamic_cast <CvgArr*>(GPUCV_GET_TEX(arr));
	if (gpuArr->_IsLocation<DataDsc_CPU>())
	{//do processing on CPU, no need to transfer for this.
		cvSetZero(arr);
	}

	if(gpuArr->_IsLocation<DataDsc_GLTex>())
	{//set empty data to texture...
		gpuArr->GetDataDsc<DataDsc_GLTex>()->Free();
	}

	GPUCV_STOP_OP(
		cvSetZero(arr),
		arr, NULL, NULL, NULL
		);
}
//===================================================
//=>stop CVGXCORE_OPER_COPY_FILL_GRP
//===================================================
//___________________________________________________
//___________________________________________________
//___________________________________________________
//===================================================
//=>start CVGXCORE_OPER_TRANSFORM_PERMUT_GRP
//===================================================
void cvgFlip(  CvArr* src, CvArr* dst, int flip_mode/*=0*/)
{
	GPUCV_START_OP(cvFlip( src, dst, flip_mode),
			"cvgFlip",
			dst,
			GenericGPU::HRD_PRF_2);

	SG_Assert(src, "cvgFlip() => No src image");
	CvgArr * cvgSrc = (CvgArr * )GetTextureManager()->Get<CvgArr>(src);
	SG_Assert(cvgSrc, "Could not get DataConainer pointer");
	DataDsc_GLTex * ddGLTexSrc1 = cvgSrc->SetLocation<DataDsc_GLTex>(true);
	SG_Assert(ddGLTexSrc1, "Could not get DataDsc_GLTex pointer");


	//we flip src image
		TextCoord<double> * tCoord = ddGLTexSrc1->_GenerateTextCoord();
		ddGLTexSrc1->_UpdateTextCoord();
		TextCoord<double> tBackup(*tCoord);

		if(flip_mode==0)
		{//arround x axes
			tCoord->FlipH();
		}
		else if (flip_mode > 0)
		{//arround y axes
			tCoord->FlipV();
		}
		else
		{
			tCoord->FlipH();
			tCoord->FlipV();
		}
		cvgCopy(src, dst);
		//restore back old coordinates
		*tCoord = tBackup;

	GPUCV_STOP_OP(
			cvFlip( src, dst, flip_mode),
			src, dst, NULL, NULL
			);
}
/*! \todo Check that the N first images are not NULL, see OpenCV Doc.
*/
void cvgMerge(CvArr* src0, CvArr* src1,
			  CvArr* src2, CvArr* src3,
			  CvArr* dst )
{
	GPUCV_START_OP(cvMerge( src0, src1, src2, src3, dst),
		"cvgMerge",
		dst,
		GenericGPU::HRD_PRF_2);

		SG_Assert(dst, "Destination image is NULL");
		SG_Assert(src0, "Src0 image is NULL");

		std::string TemplateParams ="";
		//Src
		TextureGrp SrcGrp;
		//src 1
		CvgArr * pCurImage = (CvgArr*)GetTextureManager()->Get<CvgArr>(src0);
		pCurImage->GetDataDsc<DataDsc_GLTex>()->_SetTextureName("inputR");
		SrcGrp.AddTexture(pCurImage);
		TemplateParams+="$INPUT_R=1//";
		if(src1)
		{
			pCurImage = (CvgArr*)GetTextureManager()->Get<CvgArr>(src1);
			pCurImage->GetDataDsc<DataDsc_GLTex>()->_SetTextureName("inputG");
			SrcGrp.AddTexture(pCurImage);
			TemplateParams+="$INPUT_G=1//";
		}
		if(src2)
		{
			pCurImage = (CvgArr*)GetTextureManager()->Get<CvgArr>(src2);
			pCurImage->GetDataDsc<DataDsc_GLTex>()->_SetTextureName("inputB");
			SrcGrp.AddTexture(pCurImage);
			TemplateParams+="$INPUT_B=1//";
		}
		if(src3)
		{
			pCurImage = (CvgArr*)GetTextureManager()->Get<CvgArr>(src3);
			pCurImage->GetDataDsc<DataDsc_GLTex>()->_SetTextureName("inputA");
			SrcGrp.AddTexture(pCurImage);
			TemplateParams+="$INPUT_A=1//";
		}
		//dst
		TextureGrp DstGrp;
		DstGrp.AddTexture(GetTextureManager()->Get<CvgArr>(dst));

		TemplateOperator(FctName, "FShaders/merge_color", "",
			&SrcGrp, &DstGrp, NULL, 0,TextureGrp::TEXTGRP_SAME_SIZE, TemplateParams);

	GPUCV_STOP_OP(
		cvgSynchronize(dst);//WE DO MANUAL SYNCHRONIZE HERE CAUSE MACRO ACCEPT ONLY 4 images...
		cvMerge( src0, src1, src2, src3, dst),
		src0, src1, src2, src3
		);
}
//========================================
//===============================================================
void cvgSplit( CvArr* src, CvArr* dst0, CvArr* dst1,
			  CvArr* dst2, CvArr* dst3 )
{
	GPUCV_START_OP(cvCvtPixToPlane( src, dst0, dst1, dst2, dst3 ),
		"cvgSplit",
		src,
		GenericGPU::HRD_PRF_2);

	string chosen_filter;

#if 0//test with multiple render target
	std::string metatag;

	//set groups
	GCV::TextureGrp Igrp;
	GCV::TextureGrp Ogrp;
	Igrp.SetGrpType(GCV::TextureGrp::TEXTGRP_INPUT);
	Ogrp.SetGrpType(GCV::TextureGrp::TEXTGRP_OUTPUT);
	Igrp.AddTexture(GPUCV_GET_TEX(src));


	int ChannelsToSplit =0;
	std::string OutPutName;
	if (dst0)
	{
		OutPutName = "$OUTPUT_";
		OutPutName += SGE::ToCharStr(ChannelsToSplit);
		metatag += OutPutName + "=1//";
		metatag += OutPutName + "_C=r//";
		ChannelsToSplit++;
		Ogrp.AddTexture(GPUCV_GET_TEX(dst0));
	}
	if (dst1)
	{
		OutPutName = "$OUTPUT_";
		OutPutName += SGE::ToCharStr(ChannelsToSplit);
		metatag += OutPutName + "=1//";
		metatag += OutPutName + "_C=g//";
		ChannelsToSplit++;
		Ogrp.AddTexture(GPUCV_GET_TEX(dst1));
	}
	if (dst2)
	{
		OutPutName = "$OUTPUT_";
		OutPutName += SGE::ToCharStr(ChannelsToSplit);
		metatag += OutPutName + "=1//";
		metatag += OutPutName + "_C=b//";
		ChannelsToSplit++;
		Ogrp.AddTexture(GPUCV_GET_TEX(dst2));
	}
	if (dst3)
	{
		OutPutName = "$OUTPUT_";
		OutPutName += SGE::ToCharStr(ChannelsToSplit);
		metatag += OutPutName + "=1//";
		metatag += OutPutName + "_C=a//";
		ChannelsToSplit++;
		Ogrp.AddTexture(GPUCV_GET_TEX(dst3));
	}
	//special case, we extract only one channel
	if(ChannelsToSplit==1)
	{
		metatag+= "$SINGLE_OUTPUT=r//";
	}
	else
	{
		metatag+= "$SINGLE_OUTPUT=0//";
	}

	//call the shader
	TemplateOperator("cvgSplit", "FShaders/split_color", "",
		&Igrp, &Ogrp,NULL, NULL, GCV::TextureGrp::TEXTGRP_SAME_SIZE, metatag);



#else


	if (dst0)
	{
		TemplateOperator(FctName, "FShaders/pix2red", "",
			src, NULL, NULL,
			dst0);
		//chosen_filter = GetFilterManager()->GetFilterName("FShaders/pix2red.frag");
		//GpuFilter::FilterSize size = GetSize(dst0);
		//GetFilterManager()->Apply(chosen_filter, src, dst0, &size);
	}

	if (dst1)
	{
		TemplateOperator(FctName, "FShaders/pix2green", "",
			src, NULL, NULL,
			dst1);
		//chosen_filter = GetFilterManager()->GetFilterName("FShaders/pix2green.frag");
		//GpuFilter::FilterSize size = GetSize(dst1);
		//GetFilterManager()->Apply(chosen_filter, src, dst1, &size);
	}

	if (dst2)
	{
		TemplateOperator(FctName, "FShaders/pix2blue", "",
			src, NULL, NULL,
			dst2);
		/*chosen_filter = GetFilterManager()->GetFilterName("FShaders/pix2blue.frag");
		GpuFilter::FilterSize size = GetSize(dst2);
		GetFilterManager()->Apply(chosen_filter, src, dst2, &size);
		*/
	}
#endif
	GPUCV_STOP_OP(
		cvgSynchronize(dst3);//WE DO MANUAL SYNCHRONIZE HERE CAUSE MACRO ACCEPT ONLU 4 images...
	cvCvtPixToPlane( src, dst0, dst1, dst2, dst3 ),//in case if error this operator is called
		src, dst0, dst1, dst2//in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//========================================

//===================================================
//=>stop CVGXCORE_OPER_TRANSFORM_PERMUT_GRP
//===================================================
//___________________________________________________
//___________________________________________________
//___________________________________________________
//===================================================
//=>start CVGXCORE_OPER_ARITHM_LOGIC_COMP_GRP
//===================================================

//========================================
#if _GPUCV_DEVELOP_BETA
void cvgLUT(CvArr* src, CvArr* dst, CvArr* lut )
{
	GPUCV_START_OP(cvLUT(src, dst, lut),
		"cvgLUT",
		dst,
		GenericGPU::HRD_PRF_2);

	IplImage * Ipl_lut = (IplImage*)lut;
	SG_Assert(GLEW_ARB_imaging, "GL_ARB_imaging extension not found, using cvLut instead!");
	char SrcDepth = Ipl_lut->depth;
	//SG_Assert(SrcDepth==CV_8S || SrcDepth==CV_8U, "Src depth must be CV_8S or CV_8U!");

	//get texture objects
	CvgArr * cvg_dst, *cvg_src, *cvg_lut;

	DataDsc_GLBuff	* glbuff_lut=NULL;
	DataDsc_GLTex		* gl_src, *gl_dst;

	//get CvgArr
	cvg_src = dynamic_cast<CvgArr * > (GPUCV_GET_TEX(src));
	cvg_dst = dynamic_cast<CvgArr * > (GPUCV_GET_TEX(dst));
	cvg_lut = dynamic_cast<CvgArr * > (GPUCV_GET_TEX(lut));

	//get GL textures
#if 0
	glbuff_lut	= cvg_lut->GetDataDsc<DataDsc_GLBuff>();
	glbuff_lut->_SetType(DataDsc_GLBuff::PIXEL_UNPACK_BUFFER);

	cvg_lut->SetLocation<DataDsc_GLBuff>(true);
#endif
	//gl_src		= cvg_src->SetLocation<DataDsc_GLTex>(true);
	gl_dst		= cvg_dst->SetLocation<DataDsc_GLTex>(false);


	//check format:
#if 0
	SG_Assert(gl_src->_GetNChannels() == gl_src->_GetNChannels(), "Src and Dst have different number of channels");
	SG_Assert(gl_dst->GetPixelType() == glbuff_lut->GetPixelType(), "Dst and Lut have different depth(pixel type)");
#endif
	if(Ipl_lut->nChannels!=1)
	{
		SG_Assert(gl_dst->_GetNChannels() == Ipl_lut->nChannels, "Lut channel number must be 1 or equal to Dst channel number");
	}

	//start the operator
	GLuint ColorLUTType = GL_POST_COLOR_MATRIX_COLOR_TABLE;//GL_COLOR_TABLE;// GL_POST_CONVOLUTION_COLOR_TABLE
	gl_dst->SetRenderToTexture();
	gl_dst->InitGLView();

#if 1
	if(glbuff_lut)
	{//USE color table from GPU
		glbuff_lut->_Bind();
		glColorTable(ColorLUTType,
			cvgConvertCVInternalTexFormatToGL(Ipl_lut),
			glbuff_lut->_GetWidth()*glbuff_lut->_GetHeight(),
			glbuff_lut->GetPixelFormat(),
			glbuff_lut->GetPixelType(),
			NULL);
	}
	else//use color table from CPU
	{
		glColorTable(ColorLUTType,
			cvgConvertCVInternalTexFormatToGL(Ipl_lut),
			Ipl_lut->width,
			cvgConvertCVTexFormatToGL(Ipl_lut),
			cvgConvertCVPixTypeToGL(Ipl_lut),
			Ipl_lut->imageData);
	}
	glEnable(ColorLUTType);
#endif
	if(SrcDepth==CV_8S)
	{//apply a translate color of 128

	}
	//gl_src->DrawFullQuad(gl_src->_GetWidth(),gl_src->_GetHeight());
	glDrawPixels(512,512,GL_RGB, GL_UNSIGNED_BYTE, ((IplImage*)src)->imageData);
	if(SrcDepth==CV_8S)
	{

	}
#if 1
	glDisable(ColorLUTType);
	if(glbuff_lut)
		glbuff_lut->_UnBind();
#endif
	gl_dst->UnsetRenderToTexture();
	//


	//if(cvg_dst->GetOption(DataContainer::CPU_RETURN))
	cvg_dst->SetLocation<DataDsc_IplImage>(true);
	/*	if(cvg_src->GetOptions(DataContainer::CPU_RETURN)
	cvg_dst->SetLocation<DataDsc_IplImage>(true);
	if(cvg_lut->GetOptions(DataContainer::CPU_RETURN)
	cvg_dst->SetLocation<DataDsc_IplImage>(true);
	*/

	GPUCV_STOP_OP(
		cvLUT(src, dst, lut),//in case if error this operator is called
		src, dst, lut, NULL//in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
#endif
//========================================
void cvgConvertScale(CvArr* src, CvArr* dst, double scale/*=1*/, double shift/*=0*/ )
{
	GPUCV_START_OP(cvConvertScale(src, dst, scale, shift),
		"cvgConvertScale",
		dst,
		GenericGPU::HRD_PRF_2);

	float Params [2] = {scale*256., shift};
	std::string MetaOptions = "$DEF_SCALE=1//";
	if(shift!=0)
		MetaOptions += "$DEF_SHIFT=1//";


	TemplateOperator(FctName, "FShaders/convert_Scale", "",
		src, NULL, NULL,
		dst, Params, 2,
		TextureGrp::TEXTGRP_SAME_SIZE, MetaOptions);

	GPUCV_STOP_OP(
		cvConvertScale(src, dst, scale, shift),//in case if error this operator is called
		src, dst, NULL, NULL//in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgAdd( CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask/*=NULL*/)
{
	//KEYTAGS: TUTO_CREATE_OP_BASE_TAG__STP2__LAUNCHER
	GPUCV_START_OP(cvAdd(src1, src2, dst, mask),
		"cvgAdd",
		dst,
		GenericGPU::HRD_PRF_2);

	//KEYTAGS: TUTO_CREATE_OP_GLSL_TAG__STP1__LAUNCHER
	TemplateOperator(FctName, "FShaders/add_all", "",
		src1, src2, mask,
		dst, NULL, 0,
		TextureGrp::TEXTGRP_SAME_ALL, (mask)?"$DEF_MASK=1//":"");

	GPUCV_STOP_OP(
		cvAdd(src1, src2, dst, mask),//in case if error this operator is called
		src1, src2, mask, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgAddS( CvArr* src1, CvScalar value, CvArr* dst, CvArr* mask/*=NULL*/)
{
	GPUCV_START_OP(cvAddS(src1, value, dst, mask),
		"cvgAdds",
		dst,
		GenericGPU::HRD_PRF_2);

	float vfloat[4];
	for (int i=0;i<4;i++)
		vfloat[i] = value.val[i]/256;

	std::string MetaOptions= "$DEF_SCALAR=1//";
	if(mask)
		MetaOptions += "$DEF_MASK=1//";

	TemplateOperator(FctName, "FShaders/add_all", "",
		src1, NULL, mask,
		dst, vfloat, 4,
		TextureGrp::TEXTGRP_NO_CONTROL, MetaOptions);

	GPUCV_STOP_OP(
		cvAddS(src1, value, dst, mask),//in case if error this operator is called
		src1, NULL, mask, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgSub( CvArr* src1, CvArr* src2, CvArr* dst, CvArr* mask/*=NULL*/)
{
	GPUCV_START_OP(cvSub(src1, src2, dst, mask),
		"cvgSub",
		dst,
		GenericGPU::HRD_PRF_2);


	TemplateOperator(FctName, "FShaders/sub_all", "",
		src1, src2, mask,
		dst, NULL, 0,
		TextureGrp::TEXTGRP_SAME_ALL, (mask)?"$DEF_MASK=1//":"");

	GPUCV_STOP_OP(
		cvSub(src1, src2, dst, mask),//in case if error this operator is called
		src1, src2, mask, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgSubS( CvArr* src1, CvScalar value, CvArr* dst, CvArr* mask/*=NULL*/)
{
	GPUCV_START_OP(cvSubS(src1, value, dst, mask),
		"cvgSubS",
		dst,
		GenericGPU::HRD_PRF_2);

	float vfloat[4];
	for (int i=0;i<4;i++)
		vfloat[i] = value.val[i]/256;

	std::string MetaOptions= "$DEF_SCALAR=1//";
	if(mask)
		MetaOptions += "$DEF_MASK=1//";

	TemplateOperator(FctName, "FShaders/sub_all", "",
		src1, NULL, mask,
		dst, vfloat, 4,
		TextureGrp::TEXTGRP_NO_CONTROL, MetaOptions);

	GPUCV_STOP_OP(
		cvSubS(src1, value, dst, mask),//in case if error this operator is called
		src1, NULL, mask, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgSubRS( CvArr * src1, CvScalar value, CvArr* dst,  CvArr* mask)
{
	GPUCV_START_OP(cvSubRS(src1, value, dst, mask),
		"cvgSubRS",
		dst,
		GenericGPU::HRD_PRF_2);

	float vfloat[4];
	for (int i=0;i<4;i++)
		vfloat[i] = value.val[i]/256;

	std::string MetaOptions = "$DEF_SCALAR=1//$DEF_REVERSE_ORDER=1//";
	if(mask)
		MetaOptions += "$DEF_MASK=1//";

	TemplateOperator(FctName, "FShaders/sub_all", "",
		src1, NULL, mask,
		dst, vfloat, 4,
		TextureGrp::TEXTGRP_NO_CONTROL, MetaOptions);

	GPUCV_STOP_OP(
		cvSubRS(src1, value, dst, mask),//in case if error this operator is called
		src1, NULL, mask, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgMul(  CvArr* src1,  CvArr* src2, CvArr* dst, double scale/*=1*/)
{
	float factor[1] = {scale*256.};
	//cvgOp2ImgTemp(src1, src2, dst, (string) "cvgMul", "op_multiply", factor, 1);
	GPUCV_START_OP(cvMul(src1, src2, dst),
		"cvgMul",
		dst,
		GenericGPU::HRD_PRF_2);

	TemplateOperator(FctName, "FShaders/template_arithmetic", "",
		src1, src2, NULL,
		dst, factor, 1,
		TextureGrp::TEXTGRP_SAME_ALL_FORMAT, "$DEF_SRC2=1//$DEF_OPERATION=*//$DEF_GLOBAL_FACTOR=*Parameters[0]//$DEF_PARAMETER_NBR=1//");

	GPUCV_STOP_OP(
		cvMul(src1, src2, dst),//in case if error this operator is called
		src1, src2, NULL, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgDiv( CvArr* src1,  CvArr* src2, CvArr* dst, double scale/*=1*/)
{
	GPUCV_START_OP(cvDiv(src1, src2, dst),
		"cvgDiv",
		dst,
		GenericGPU::HRD_PRF_2);

	float factor[1] = {scale/256.};
	if(src1!=NULL)
	{
		TemplateOperator(FctName, "FShaders/template_arithmetic", "",
			src1, src2, NULL,
			dst, factor, 1,
			TextureGrp::TEXTGRP_SAME_ALL_FORMAT,
			"$DEF_SRC2=1//$DEF_OPERATION=/ //$DEF_GLOBAL_FACTOR=*Parameters[0]//$DEF_PARAMETER_NBR=1//");
	}
	else
	{
		TemplateOperator(FctName, "FShaders/template_arithmetic", "",
			src2, NULL, NULL,
			dst, factor, 1,
			TextureGrp::TEXTGRP_SAME_ALL_FORMAT,
			"$DEF_SRC2=1//$DEF_OPERATION=/ //$DEF_GLOBAL_FACTOR=*Parameters[0]//$DEF_PARAMETER_NBR=1//");
	}

	GPUCV_STOP_OP(
		cvDiv(src1, src2, dst),//in case if error this operator is called
		src1, src2, NULL, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================

/*=========================================================
Cmp operator
==============================================================*/
/*! \todo Use GLSL vector functions such as 'equal/greaterThan/...'
*/
void cvgCmp(CvArr* src1, CvArr* src2, CvArr* dst,int op)
{
	GPUCV_START_OP(cvCmp(src1, src2, dst, op),
		"cvgCmp",
		dst,
		GenericGPU::HRD_PRF_2);

	std::string MetaTag="$CMP_OPER=";//(a";
	switch (op)
	{
	case 0: MetaTag += "=="; break;
	case 1: MetaTag += ">"; break;
	case 2: MetaTag += ">="; break;
	case 3:	MetaTag += "<"; break;
	case 4: MetaTag += "<="; break;
	case 5: MetaTag += "!="; break;
	}
	MetaTag+="//";//b)

	MetaTag+="$DEF_IMG2=1//";
	if(GetCVDepth(src1) >= IPL_DEPTH_32F)
		MetaTag+="$PIXELTYPE=float//";//we use int to compare by default

	TemplateOperator(FctName, "FShaders/cmp", "",
		src1, src2,NULL,
		dst,NULL,0,
		TextureGrp::TEXTGRP_SAME_ALL_FORMAT, MetaTag);

	GPUCV_STOP_OP(
		cvCmp(src1, src2, dst, op),//in case if error this operator is called
		src1, src2, NULL, dst //in case of error, we get this images back to CPU, so the OpenCV operator can be called
		);
}
//===================================================
/*! \todo Use GLSL vector functions such as 'equal/greaterThan/...'
*/
void cvgCmpS(CvArr* src, double value, CvArr* dst,int op)
{
	GPUCV_START_OP(cvCmpS(src, value, dst, op),
		"cvgCmpS",
		dst,
		GenericGPU::HRD_PRF_2);

	std::string MetaTag="$CMP_OPER=";//(a";
	switch (op)
	{
	case 0: MetaTag += "=="; break;
	case 1: MetaTag += ">"; break;
	case 2: MetaTag += ">="; break;
	case 3:	MetaTag += "<"; break;
	case 4: MetaTag += "<="; break;
	case 5: MetaTag += "!="; break;
	}
	MetaTag+="//";//b)

	MetaTag+="$DEF_SCALAR=1//";

	if(GetCVDepth(src) >= IPL_DEPTH_32F)
		MetaTag+="$PIXELTYPE=float//";//we use int to compare by default

	float fl_Value = value/256.;
	TemplateOperator(FctName, "FShaders/cmp", "",
		src, NULL,NULL,
		dst, &fl_Value,1,
		TextureGrp::TEXTGRP_SAME_ALL_FORMAT, MetaTag);

	GPUCV_STOP_OP(
		,
		src, NULL, NULL, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgMax(  CvArr* src1,  CvArr* src2, CvArr* dst)
{
	GPUCV_START_OP(cvMax(src1, src2, dst),
		"cvgMax",
		dst,
		GenericGPU::HRD_PRF_2);

	TemplateOperator(FctName, "FShaders/max", "",
		src1, src2, NULL,
		dst, NULL, 0,
		TextureGrp::TEXTGRP_SAME_ALL_FORMAT, "");

	//cvgFct2ImgTemp(src1, src2, dst, (string) "cvgMax", (string) "max");

	GPUCV_STOP_OP(
		cvMax(src1, src2, dst),//in case if error this operator is called
		src1, src2, NULL, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgMin(  CvArr* src1,  CvArr* src2, CvArr* dst)
{
	//cvgFct2ImgTemp(src1, src2, dst, (string) "cvgMin", (string) "min");
	GPUCV_START_OP(cvMin(src1, src2, dst),
		"cvgMin",
		dst,
		GenericGPU::HRD_PRF_2);

	TemplateOperator(FctName, "FShaders/min", "",
		src1, src2, NULL,
		dst, NULL, 0,
		TextureGrp::TEXTGRP_SAME_ALL_FORMAT, "");

	GPUCV_STOP_OP(
		cvMin(src1, src2, dst),//in case if error this operator is called
		src1, src2, NULL, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgMaxS( CvArr * src, double value, CvArr* dst)
{
	GPUCV_START_OP(cvMaxS(src, value, dst),
		"cvgMaxS",
		dst,
		GenericGPU::HRD_PRF_2);

	float Value=value;
	//scale the float value to the src1 image range
	unsigned int TempDepth = GetGLDepth(src);
	if((TempDepth != GL_FLOAT)
		&&(TempDepth != GL_INT)
		&&(TempDepth != GL_UNSIGNED_INT)
		&&(TempDepth!= GL_DOUBLE))
	{
		Value = Value/256.;
	}

	TemplateOperator(FctName, "FShaders/maxs", "",
		src, NULL, NULL,
		dst, &Value, 1,
		TextureGrp::TEXTGRP_SAME_ALL, "");

	GPUCV_STOP_OP(
		cvMaxS(src, value, dst),//in case if error this operator is called
		src, NULL, NULL, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgMinS( CvArr * src, double value, CvArr* dst)
{
	GPUCV_START_OP(cvMinS(src, value, dst),
		"cvgMinS",
		dst,
		GenericGPU::HRD_PRF_2);

	float Value=value;
	//scale the float value to the src1 image range
	unsigned int TempDepth = GetGLDepth(src);
	if((TempDepth != GL_FLOAT)
		&&(TempDepth != GL_INT)
		&&(TempDepth != GL_UNSIGNED_INT)
		&&(TempDepth!= GL_DOUBLE))
	{
		Value = Value/256.;
	}

	TemplateOperator("cvgMinS", "FShaders/mins", "",
		src, NULL, NULL,
		dst, &Value, 1,
		TextureGrp::TEXTGRP_SAME_ALL, "");

	GPUCV_STOP_OP(
		cvMinS(src, value, dst),//in case if error this operator is called
		src, NULL, NULL, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgAbs( CvArr * src, CvArr* dst)
{
	GPUCV_START_OP(cvAbs(src, dst),
		"cvgAbs",
		dst,
		GenericGPU::HRD_PRF_2);

	TemplateOperator(FctName, "FShaders/abs", "",
		src, NULL, NULL,
		dst, NULL, 0,
		TextureGrp::TEXTGRP_SAME_ALL,
		"$DEF_OPER=abs(img0)//");

	GPUCV_STOP_OP(
		,
		src, NULL, NULL, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgAbsDiff( CvArr * src1, CvArr * src2, CvArr* dst)
{
	GPUCV_START_OP(cvAbsDiff(src1, src2, dst),
		"cvgAbsDiff",
		dst,
		GenericGPU::HRD_PRF_2);

	TemplateOperator(FctName, "FShaders/abs", "",
		src1, src2, NULL,
		dst, NULL, 0,
		TextureGrp::TEXTGRP_SAME_ALL,
		"$DEF_OPER=abs(img0-img1)//$DEF_IMG2=1//");

	GPUCV_STOP_OP(
		,
		src1, src2, NULL, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
void cvgAbsDiffS( CvArr * src, CvArr* dst, CvScalar value)
{
	GPUCV_START_OP(cvAbsDiffS(src, dst, value),
		"cvgAbsDiffS",
		dst,
		GenericGPU::HRD_PRF_2);

	float vfloat[4];
	for (int i=0;i<4;i++)
		vfloat[i] = value.val[i]/256;

	TemplateOperator(FctName, "FShaders/abs", "",
		src, NULL, NULL,
		dst, vfloat, 4,
		TextureGrp::TEXTGRP_SAME_ALL,
		"$DEF_OPER=abs(img0-img1)//$DEF_SCALAR=1//");

	GPUCV_STOP_OP(
		,
		src, NULL, NULL, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
//===================================================
//=>stop CVGXCORE_OPER_ARITHM_LOGIC_COMP_GRP
//===================================================
//___________________________________________________
//___________________________________________________
//___________________________________________________
//===================================================
//=>start CVGXCORE_OPER_STATS_GRP
//===================================================
//========================================
CvScalar cvgSum(CvArr* arr)
{
	CvScalar sum = cvgAvg(arr);
	int imgPixelsNbr = GetWidth(arr) * GetHeight(arr);
	sum.val[0] *= imgPixelsNbr;
	sum.val[1] *= imgPixelsNbr;
	sum.val[2] *= imgPixelsNbr;
	sum.val[3] *= imgPixelsNbr;
	return sum;
}
//========================================
CvScalar cvgAvg(  CvArr* src,  CvArr* mask/*=NULL*/)//, bool NotNullPixels/*=false*/)
{
	CvScalar result;
	GPUCV_START_OP(return cvAvg(src, mask);,
		"cvgAvg",
		src,
		GenericGPU::HRD_PRF_2);

	SG_Assert(!mask, "cvgAvg Not compatible with MASK");

	//declare size of data we read back.
	CvSize size;
	size.width = size.height = GPUCV_CXCOREG_AVG_READBACK_SIZE;

	//get source Container
	CvgArr * cvgSrc = dynamic_cast<CvgArr*>(GPUCV_GET_TEX(src));
	SG_Assert(cvgSrc, "no source");
	cvgSrc->PushSetOptions(DataContainer::UBIQUITY, true);

	//get source DataDsc
	DataDsc_GLTex * DDTex_src = cvgSrc->GetDataDsc<DataDsc_GLTex>();

	if (DDTex_src->HaveData())
	{//if image is already on GPU...we need to do some conversion manually
		DDTex_src->_SetAutoMipMap(true);
	}
	else if(cvgSrc->_IsLocation<DataDsc_CPU>())
	{//if image is only on CPU...no problem, just load it and mipmapping will be done automatically.
		//nothing to do now
		cvgSrc->SetLocation<DataDsc_GLTex>(true);
	}

	//set ubiquity flags for input texture + AUTO_MIPMAP...
#if _GPUCV_GL_USE_MIPMAPING

	//cvgSrc->GetDataDsc<DataDsc_GLTex>()->_SetAutoMipMap(true);
	//cvgSrc->SetLocation<DataDsc_GLTex>(true);
	//cvgSrc->Print();
#endif

	//Create temp texture that contains result...
	int SrcnChannels = GetnChannels(src);
	int TempnChannels = (SrcnChannels==1)?3:SrcnChannels;//can not render to luminance texture.
	IplImage * TempIlp = cvgCreateImage(size, IPL_DEPTH_32F, TempnChannels);
	DataContainer * DC_Temp = GPUCV_GET_TEX(TempIlp);//, DataContainer::LOC_GPU, false);


	//set mipmapping option
	//cvgSrc->PushSetOptions(DataContainer::AUTO_MIPMAP, true);
	//GLuint Loc = cvgSrc->_GetLocation();
	DataDsc_GLTex * DDTex_Temp = 	DC_Temp->SetLocation<DataDsc_GLTex>(false);

	DC_Temp->Print();
	float  *data = new float[size.width * size.height*sizeof(float)];
	//draw to textureFrame Buffer
	DDTex_Temp->SetRenderToTexture();
	InitGLView(*DDTex_Temp);
#if _GPUCV_USE_DATA_DSC
	DDTex_src->_Bind();
	DDTex_src->_SetTexParami(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	DDTex_src->DrawFullQuad(DDTex_Temp->_GetWidth(), DDTex_Temp->_GetHeight());
	DDTex_Temp->_ReadData((void**)&data, 0, DDTex_Temp->_GetWidth(), 0,DDTex_Temp->_GetHeight());

#else
	cvgSrc->_Bind();
	cvgSrc->_SetTexParami(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	DataContainer::DrawFullQuad(TempTex->_GetWidth(), TempTex->_GetHeight(), (DataContainer*)cvgSrc);
#endif

	//read back
#if !_GPUCV_USE_DATA_DSC
	TempTex->_CreateLocation(DataContainer::LOC_CPU);
	TempTex->_AddLocation(DataContainer::LOC_CPU);
	TempTex->FrameBufferToCpu(true);
#endif
	//==========
	DDTex_Temp->UnsetRenderToTexture();

	cvgSrc->Print();
	DC_Temp->Print();

	//format result


	int channels = DDTex_src->_GetNChannels();
	/*
	std::cout << "Average on " << size.height << "*" << size.width<< std::endl;
	for (int i = 0; i < size.height; i++)
	{
	for (int j = 0; j < size.width*channels; j++)
	printf("%d \t", data[i*size.width*channels+j]);
	std::cout << std::endl;
	}
	*/

	result.val[0] = result.val[1] = result.val[2] = result.val[3] = 0.;

	for (int i = 0; i < size.height; i++)
	{
		for (int j = 0; j < size.width; j++)//channels)
		{
			for(int c = 0; c < TempnChannels; c++)
			{
				result.val[c] += *data;
				//std::cout << "data:" << *data << std::endl;
				data ++;;
			}
			//[i*size.width*channels+j];
			//result.val[1] += *(data+1);//[i*size.width*channels+j+1];
			//result.val[2] += *(data+2);//data[i*size.width*channels+j+2];
			//result.val[3] += *(data+3);//data[i*size.width*channels+j+3];
			//data += channels;
		}
	}

	float dividende = size.width*size.height/256.;
	for (int i=0; i < SrcnChannels; i++)
	{
		result.val[i] /= dividende;
	}
	for (int i=SrcnChannels; i < TempnChannels; i++)
	{
		result.val[i] = 0;
	}

	cvgSrc->PopOptions();

	if(cvgSrc->GetOption(DataContainer::CPU_RETURN))
		cvgSrc->SetLocation<DataDsc_CPU>(true);

	cvgReleaseImage(&TempIlp);

	GPUCV_STOP_OP(
		return cvAvg(src, mask);, //is errors call opencv operator
		src, mask, NULL, NULL //and restore to CPU all given images.
		);
	return result;
}
//=======================================
#if 0//_GPUCV_DEVELOP_BETA
//! \todo must be updated!!!!
void cvgMinMaxLoc(CvArr* src, double* min_val, double* max_val,
				  CvPoint* min_loc/*=NULL*/, CvPoint* max_loc/*=NULL*/, CvArr* mask/*=NULL*/ )
{
	//test image format

	if (src==NULL)
	{
		GPUCV_ERROR("\nCritical : cvgMinMaxLoc() => src is NULL.");
		return cvMinMaxLoc(src,min_val, max_val,min_loc,max_loc,mask);
	}

	GLuint Depth = GetGLDepth(src);
	if ((GetnChannels(src)> 1)
		|| (Depth != GL_BYTE)
		|| (Depth != GL_UNSIGNED_BYTE)
		)
	{
		GPUCV_WARNING("\nWarning: cvgMinMaxLoc() => Source image is not single channel or not 8-bits.");
		return cvMinMaxLoc(src,min_val, max_val,min_loc,max_loc,mask);
	}
	else if (mask != NULL)
	{
		GPUCV_WARNING("\nWarning : cvgMinMaxLoc() => mask is not NULL, using original Opencv operator.");
		return cvMinMaxLoc(src,min_val, max_val,min_loc,max_loc,mask);
	}

	SetThread();

	CvSize Size1= {32,32};
	IplImage * dstGPU = cvgCreateImage(Size1,GetnChannels(src),GetCVDepth(src));
	DataContainer * TmpTex = new DataContainer(_GPUCV_FRAMEBUFFER_DFLT_FORMAT,Size1.width,Size1.height,GL_BGR,GL_FLOAT, 0);
	float *data=new float[Size1.width*Size1.height*3];
	int Ipix= 0;
	float Pixel;
	int PixY, PixX;

	//resize image using function 'maxloc'
	cvgUnsetCpuReturn(dstGPU);
	GLsizei width = GetWidth(src);
	GLsizei height = GetHeight(src);

	if (max_val)
	{
		RenderBufferManager()->Force(TmpTex, Size1.width,Size1.height);
		cvgResizeGLSLFct(src,dstGPU, "maxloc", NULL);

		cvgReadPixels(0,0,Size1.width,Size1.height,GL_BGR,GL_FLOAT, data);
		RenderBufferManager()->UnForce();
		*max_val = 0;
		float max_valf=0;
		CvPoint maxloc;

		Ipix = 0;
		for (PixY =0; PixY < Size1.height; PixY++)
		{
			//Ipix += Size1.width*3;
			for (PixX =0; PixX < Size1.width; PixX++)
			{
				Pixel = data[Ipix];//dstGPU->imageData[Ipix];
				if ( Pixel > max_valf)
				{
					max_valf = Pixel;
					maxloc.x = (int)data[Ipix+1]*width;//dstGPU->imageData[Ipix+1];
					maxloc.y = (int)data[Ipix+2]*height;//dstGPU->imageData[Ipix+2];

				}
				// 		if ( Pixel == max_valf)//candidate pixels
				//			printf("\nPos : %d-%d \t values : %f %f %f => new pos : %f %f",PixX, PixY, Pixel, data[Ipix+1],data[Ipix+2], data[Ipix+1]*src->width, data[Ipix+2]*src->width);
				Ipix += 3;
			}
		}
		//printf("\nmax val %f - %d - %lf", (max_valf*256), (unsigned int)(max_valf*256), (double)(max_valf*256));
		*max_val= (double) (max_valf*255);
		//printf("\nmax val %lf", *max_val);

		if(max_loc!=NULL)
		{
			max_loc->x = maxloc.x;
			max_loc->y = maxloc.y;
		}
	}

	//resize image using function 'minloc'
	if (min_val)
	{
		RenderBufferManager()->Force(TmpTex, Size1.width,Size1.height);
		cvgResizeGLSLFct(src,dstGPU, "minloc", NULL);

		cvgReadPixels(0,0,Size1.width,Size1.height,GL_BGR,GL_FLOAT, data);
		RenderBufferManager()->UnForce();

		*min_val = 100000;
		CvPoint minloc;
		Ipix = 0;
		float min_valf=1;

		for (PixY =0; PixY < Size1.height; PixY++)
		{
			//Ipix += Size1.width*3;
			for (PixX =0; PixX < Size1.width; PixX++)
			{
				Pixel = data[Ipix];//dstGPU->imageData[Ipix];
				if ( Pixel <= min_valf)
				{
					/*	min_valf = Pixel;
					minloc.x = data[Ipix+1]*src->width;//dstGPU->imageData[Ipix+1];
					minloc.y = data[Ipix+2]*src->height;//dstGPU->imageData[Ipix+2];
					printf("\nPos : %d-%d \t values : %f %f %f => new pos : %d %d",PixX, PixY, Pixel, data[Ipix+1],data[Ipix+2], minloc.x, minloc.y);
					*/

					if (min_valf == Pixel)
					{
						//max_valf = Pixel;
						if (minloc.x > data[Ipix+1]*width)
							minloc.x = (int)data[Ipix+1]*width;//dstGPU->imageData[Ipix+1];
						if (minloc.y > data[Ipix+2]*height)
							minloc.y = (int)data[Ipix+2]*height;//dstGPU->imageData[Ipix+2];
					}
					else
					{
						min_valf = Pixel;
						minloc.x = (int)data[Ipix+1]*width;//dstGPU->imageData[Ipix+1];
						minloc.y = (int)data[Ipix+2]*height;//dstGPU->imageData[Ipix+2];
					}
					//			printf("\nPos : %d-%d \t values : %f %f %f => new pos : %d %d",PixX, PixY, Pixel, data[Ipix+1],data[Ipix+2], minloc.x, minloc.y);
				}
				Ipix += 3;
			}
		}
		*min_val= (unsigned int) (min_valf*256);

		if(min_loc!=NULL)
		{
			min_loc->x = minloc.x;
			min_loc->y = minloc.y;
		}
	}
	delete TmpTex;
	UnsetThread();
	delete (data);
}
#endif
//===================================================
//=>stop CVGXCORE_OPER_STATS_GRP
//===================================================
//___________________________________________________
//___________________________________________________
//___________________________________________________
//===================================================
//=>start CVGXCORE_OPER_LINEAR_ALGEBRA_GRP
//===================================================
#if _GPUCV_DEVELOP_BETA
void cvgScaleAdd(CvArr* src1, CvScalar scale, CvArr* src2, CvArr* dst)
{
	GPUCV_START_OP(cvScaleAdd(src1, scale, src2, dst),
		"cvgScaleAdd",
		dst,
		GenericGPU::HRD_PRF_2);

	float vfloat[4];
	for (int i=0;i<4;i++)
		vfloat[i] = scale.val[i];

	std::string MetaOptions= "$DEF_SCALE=1//";

	TemplateOperator(FctName, "FShaders/add_all", "",
		src1, src2, NULL,
		dst, vfloat, 4,
		TextureGrp::TEXTGRP_SAME_ALL, MetaOptions);

	GPUCV_STOP_OP(
		cvScaleAdd(src1, scale, src2, dst),//in case if error this operator is called
		src1, src2, NULL, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}
#endif
//===================================================
//=>stop CVGXCORE_OPER_LINEAR_ALGEBRA_GRP
//===================================================
//___________________________________________________
//___________________________________________________
//___________________________________________________
//===================================================
//=>start CVGXCORE_OPER_MATH_FCT_GRP
//===================================================



void cvgPow(CvArr* src, CvArr* dst, double power)
{
	GPUCV_START_OP(cvPow(src, dst, power),
		"cvgPow",
		dst,
		GenericGPU::HRD_PRF_2);

	if (power==1.)
	{//img dst = img src
		//we performs simple copy.
		cvgCopy(src,dst);
		return;
	}

	float val=power;
	int factor = 0;
	std::string MetaOptions;
	if((int)val != (float)val)//we use absolute operator as describe in cvPow doc
		MetaOptions = "$DEF_ABS=1//";

	DataContainer * srcTex = GPUCV_GET_TEX(src);
	DataContainer * dstTex = GPUCV_GET_TEX(dst);
	DataContainer * newDest = NULL;

	//check if internal format has suffisant precision and range
	//pixels must be stored in memory as float, and not as int or byte.
	//copy all texture properties from destination, except texture ID and cpu data.
#if 1
	if(srcTex->GetDataDsc<DataDsc_GLTex>()->GetPixelType() != GL_FLOAT)
	{
		newDest  = new TextureTemp(dstTex);
		newDest->GetDataDsc<DataDsc_GLTex>()->_SetInternalPixelFormat(GL_RGBA_FLOAT32_ATI);
		newDest->SetLabel(dstTex->GetLabel() + "_TmpCpywithFLOATformat");
		//newDest->SetLocation<DataDsc_GLTex>(false);
		factor = 4;
	}
	else
#else
	factor = 64;
#endif
	{
		newDest = dstTex;
	}

	float Params[2] = {val, 1};
	if(factor)
	{
		Params[1] = factor*pow(2.,power);
		//	MetaOptions += "$DEF_FACTOR=";
		//	MetaOptions += SGE::ToCharStr(factor*pow(2.,power));
		//	MetaOptions +="//";
	}
	TemplateOperator(FctName, "FShaders/pow", "",
		srcTex, NULL, NULL,
		newDest, Params, 2,
		TextureGrp::TEXTGRP_NO_CONTROL, MetaOptions);

	//dstTex->PopOptions();

	if (newDest!=dstTex)
	{//copy back texture ID and properties
		//and CPU
#if !_GPUCV_USE_DATA_DSC
		dstTex->_SwitchTexID(*newDest);
		if(dstTex->GetOption(DataContainer::CPU_RETURN))
			dstTex->_SwitchTexData(*newDest);
#else
		dstTex->SwitchDataDsc<DataDsc_GLTex>(newDest);

		if(dstTex->GetOption(DataContainer::CPU_RETURN))
			dstTex->SetLocation<DataDsc_CPU>(true);
#endif
		delete newDest;
	}

	GPUCV_STOP_OP(
		cvPow(src, dst, power),//in case if error this operator is called
		src, NULL, NULL, dst //in case of error, we get this images back to CPU, so the opencv operator can be called
		);
}

//===================================================
//=>stop CVGXCORE_OPER_MATH_FCT_GRP
//===================================================
//___________________________________________________
//___________________________________________________
//___________________________________________________
//===================================================
//=>start CVGXCORE_OPER_RANDOM_NUMBER_GRP
//===================================================

//===================================================
//=>stop CVGXCORE_OPER_RANDOM_NUMBER_GRP
//===================================================
//___________________________________________________
//___________________________________________________
//___________________________________________________
//===================================================
//=>start CVGXCORE_OPER_DISCRETE_TRANSFORM_GRP
//===================================================

//===================================================
//=>stop CVGXCORE_OPER_DISCRETE_TRANSFORM_GRP
//===================================================
//___________________________________________________
//___________________________________________________
//___________________________________________________

