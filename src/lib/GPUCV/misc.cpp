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
#include <cvver.h>
namespace GCV{

void * CreateCvgArr()
{
	return new CvgArr();
}

/** \sa LibraryDescriptor;
*/
_GPUCV_EXPORT_C LibraryDescriptor *modGetLibraryDescriptor(void)
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
		pLibraryDescriptor->SetDllName("gpucv");
		pLibraryDescriptor->SetImplementationName("GLSL");
		pLibraryDescriptor->SetBaseImplementationID(GPUCV_IMPL_GLSL);
		pLibraryDescriptor->SetUseGpu(true);
	}
	return pLibraryDescriptor;
}


//=================================================
int cvgInit(unsigned char InitGLContext, unsigned char isMultiThread)
{

	int result = GpuCVInit((InitGLContext>0)?true:false, (isMultiThread>0)?true:false);
	SG_TRC::CL_TRACE_BASE_PARAMS * GlobalParams=SG_TRC::CL_TRACE_BASE_PARAMS::GetCommonParam();
	//opencv version:
		GlobalParams->AddChar("opencv-V", CV_VERSION);

	return result;
}
//=================================================
void cvgTerminate()
{
	GpuCVTerminate();
}
//===============================================================
void cvgSetCpuReturn(CvArr *img)
{
	GPUCV_START_OP(_GPUCV_NOP,
		"cvgSetCpuReturn",
		img,
		GenericGPU::HRD_PRF_1);

	cvgSetOptions(img, DataContainer::CPU_RETURN, true);
	cvgSynchronize(img);

	GPUCV_STOP_OP(
		_GPUCV_NOP,
		NULL, NULL, NULL, NULL
		);
}
//===============================================================
void cvgUnsetCpuReturn(CvArr *img)
{
	GPUCV_START_OP(_GPUCV_NOP,
		"cvgSetCpuReturn",
		img,
		GenericGPU::HRD_PRF_1);

	cvgSetOptions(img, DataContainer::CPU_RETURN, false);

	GPUCV_STOP_OP(
		_GPUCV_NOP,
		NULL, NULL, NULL, NULL
		);
}
//===============================================================
const char * cvgSetShaderPath(const char * _path)
{
	GetGpuCVSettings()->SetShaderPath(_path);
	return GetGpuCVSettings()->GetShaderPath().c_str();
}
//===============================================================
const char * cvgGetShaderPath()
{
	return GetGpuCVSettings()->GetShaderPath().c_str();
}
//==========================================================
const char * cvgRetrieveShaderPath(const char * appPath)
{
	//get application path
	std::string AppName= appPath;
	std::string AppPath;
	bool bPathFound = false;
	
	GPUCV_DEBUG("argv[0]: " << AppName);
	
	SGE::ReformatFilePath(AppName);
	SGE::ParseFilePath(AppName, &AppPath);

	GPUCV_NOTICE("Current application location: " << AppPath);

	std::string gpucv_data_path_list[]={
		"../../../../data/"//since rev 59x applications are in lib/target/arch/mode -> 4 level path
		,"../../../data/"	//in visual studio, default app path might be build/windows-vs-20xx/app/ -> 3 level
		,"../../data/"		//in visual studio, default app path might be build/windows-vs-20xx/ -> 2 level
		,"../data/"			//on a release package, data could be reachable from gpucv/bin/../data/
		,"data/"			//if we start it from the GpuCV root.
#ifdef _WINDOWS
	//	,"c:\\program/ files\\GpuCV\\data\\" //default setup folder
		//user folder?
#else
		//user folder?
#endif
		//folder set dynamicaly?
		};

	int iPathNbr= sizeof(gpucv_data_path_list) / sizeof(std::string);

	


	//try from relative path
	for(int iPath = 0; iPath <  iPathNbr; iPath++)
	{
		if(DirectoryExists(gpucv_data_path_list[iPath]))
		{
			AppPath = gpucv_data_path_list[iPath];
			bPathFound=true;
			break;
		}
	}

	if(bPathFound==false)
	{
		//try from absolute path
		std::string strAppPathtoTest;
		for(int iPath = 0; iPath <  iPathNbr; iPath++)
		{
			strAppPathtoTest = AppPath;
			strAppPathtoTest += gpucv_data_path_list[iPath];
			if(DirectoryExists(strAppPathtoTest))
			{
				AppPath = strAppPathtoTest;
				break;
			}
		}
	}

#if 0//Deprecated
	if(AppPath=="")
	{
		AppPath = "../../../../data/";//since rev 59x applications are in lib/target/arch/mode -> 4 level path
	}
	else
	{
		size_t iPos = AppPath.find("lib");
		if(iPos == std::string::npos)
			iPos = AppPath.find("bin");

		if(iPos != std::string::npos)
		{
			//GPUCV_DEBUG("String found in path at pos:" << iPos);
			AppPath = AppPath.substr(0, iPos);
			AppPath += "data";
			//GPUCV_NOTICE("Changing to: " << AppPath);
		}
	}
#endif
	if(bPathFound==false)
	{
		SG_Assert(AppPath!="", "Could not found GpuCV data path!! Application might not work!");
	}
	else
	{
		GPUCV_DEBUG("Found GpuCV data path: " << AppPath);
	}
	return cvgSetShaderPath(AppPath.c_str());
}
//=================================================
#if 0
void cvgSetLocation(CvArr* src, DataContainer::TextureLocation _location, bool _dataTransfer/*=true*/)
{
	GPUCV_START_OP(_GPUCV_NOP,
		"cvgSetLocation()",
		src,
		GenericGPU::HRD_PRF_1);

	SG_Assert(src, "Input image is NULL");
	GPUCV_GET_TEX_ON_LOC(src, _location, _dataTransfer);

	GPUCV_STOP_OP(
		_GPUCV_NOP,
		src, NULL, NULL, NULL
		);
}
#endif
//==========================================================
void cvgSynchronize(CvArr* _image)
{
	//SG_Assert(_image, "Input image is NULL");
	//GPUCV_GET_TEX(_image)->SetLocation<DataDsc_CPU>(true);
	cvgSetLocation<DataDsc_CPU>(_image, true);
}
//==========================================================
void cvgFlush(CvArr* _image)
{
	SG_Assert(_image, "Input image is NULL");
	GPUCV_GET_TEX(_image)->Flush();
}
//==========================================================
void cvgSetLabel(CvArr* _image, std::string _label)
{
#if 1
	if(_image==NULL)
		GPUCV_NOTICE("Could not add label to image: "<< _label)
	SG_Assert(_image, "Input image is NULL");
	GPUCV_GET_TEX(_image)->SetLabel(_label);
#endif
}
//==========================================================
const char * cvgGetLabel(CvArr* _image)
{
	SG_Assert(_image, "Input image is NULL");
	return GPUCV_GET_TEX(_image)->GetLabel().data();
}
//==========================================================
void cvgPushSetOptions(CvArr* _arr, CL_Options::OPTION_TYPE _opt, bool val)
{
	DataContainer *temp = GPUCV_GET_TEX(_arr);
	SG_Assert(temp, "No CvgArr* found for given CvArr*");
	temp->PushSetOptions(_opt, val);
}
//==========================================================
void cvgSetOptions(CvArr* _arr, CL_Options::OPTION_TYPE _opt, bool val)
{
	DataContainer *temp = GPUCV_GET_TEX(_arr);
	SG_Assert(temp, "No CvgArr* found for given CvArr*");
	temp->SetOption(_opt, val);
}
//==========================================================
int cvgGetOptions(CvArr* _arr, CL_Options::OPTION_TYPE _opt)
{
	DataContainer *temp = GPUCV_GET_TEX(_arr);
	SG_Assert(temp, "No CvgArr* found for given CvArr*");
	return temp->GetOption(_opt);
}
//==========================================================
void cvgPopOptions(CvArr* _arr)
{
	DataContainer *temp = GPUCV_GET_TEX(_arr);
	SG_Assert(temp, "No CvgArr* found for given CvArr*");
	temp->PopOptions();
}
//==========================================================
void cvgShowImageProperties(CvArr* _arr)
{
	DataContainer *temp = GPUCV_GET_TEX(_arr);
	if(temp)
		temp->Print();
}
//==========================================================
void cvgCreateHistImage(IplImage  * img, CvHistogram *hist, CvScalar color)
{// Creating Histogram Image
	cvgSynchronize(img);
	float val_histo, max_val_histo;
	int val_graph;
	CvSize ImgSize = cvGetSize(img);
	float maxValY = ImgSize.height-1;
	cvGetMinMaxHistValue(hist,0,&max_val_histo, 0, 0 );
	//cvZero(img);
	int nb_buckets = hist->mat.dim[0].size;
	for(int i=0;i<nb_buckets;i++)
	{
		val_histo = cvQueryHistValue_1D(hist,i);
		if(val_histo==0)
			continue;
		val_graph = cvRound(maxValY-(val_histo/(max_val_histo)*maxValY));
		cvLine(img,cvPoint(i,int(maxValY)),cvPoint(i,val_graph),color);
	}
}
//=========================================================
void cvgResizeGLSLFct(  CvArr* src, CvArr* dst, char * GLSLfunction,  CvArr* mask/*=NULL*/)
{
	GPUCV_START_OP(,
		"cvgResizeGLSLFct",
		src,
		GenericGPU::HRD_PRF_3);


	string FilterName="";

	if (mask!=NULL)
		FilterName="FShaders/resizeFct_Mask";
	else
		FilterName="FShaders/resizeFct";

	if (strcmp(GLSLfunction, "min")==0)
	{
		FilterName+="($FCT_INIT=color=vec4(1.);"\
			"$FCT=color=min(colorTemp, color);$FCT_FINAL=gl_FragColor = color;)";
	}
	else if (strcmp(GLSLfunction, "max")==0)
	{
		FilterName+="($FCT_INIT=color=vec4(0.);"\
			"$FCT=color=max(colorTemp, color);$FCT_FINAL=gl_FragColor = color;)";
	}
	else if (strcmp(GLSLfunction, "maxloc")==0)
		FilterName+="($FCT_INIT=color=vec4(0.);"\
		"$FCT=if(colorTemp.b> color.b){color.b = colorTemp.b;color.g = PP.x;color.r = PP.y;}"\
		"$FCT_FINAL=gl_FragColor = color;"\
		")";
	/* .b is the max value
	.g is the x pos of the max value
	.r is the y pos of the max value
	*/
	else if (strcmp(GLSLfunction, "minloc")==0)
		FilterName+="($FCT_INIT=color=vec4(1.);"\
		"$FCT=if(colorTemp.b<=color.b){color.b = colorTemp.b;color.g = PP.x;color.r = PP.y;}"\
		"$FCT_FINAL=gl_FragColor = color;"\
		")";
	/* .b is the min value
	.g is the x pos of the min value
	.r is the y pos of the min value
	*/
	else if (strcmp(GLSLfunction, "CV_INTER_LINEAR")==0)
		FilterName+="($FCT_INIT=;int PixelNbr=0;"\
		"$FCT=color+=colorTemp;PixelNbr+=1;"\
		"$FCT_FINAL=gl_FragColor = color/PixelNbr;"\
		")";
	/* .b is the min value
	.g is the x pos of the min value
	.r is the y pos of the min value
	*/
	else if (strcmp(GLSLfunction, "AVG_NONULL")==0)//Average of non NULL pixels
		FilterName+="($FCT_INIT= color=vec4(0.); float PixelNbr=0.;"\
		"$FCT= if(colorTemp.r!=0.)"\
		"{color+=colorTemp;PixelNbr++;}"\
		"$FCT_FINAL= color.r/=PixelNbr;"\
		"color.g=floor(PixelNbr/256.)/256.;"\
		"color.b=mod(PixelNbr,256.)/256.;"\
		"gl_FragColor=color;"\
		")";
	/* .R is the average value of the region
	.G is the higher part of the pixel number (G*256+B)
	.B is the lower part of the pixel number (G*256+B)
	*/
	else if (strcmp(GLSLfunction, "AVG")==0)//Average of all pixels
		FilterName+="($FCT_INIT= color=vec4(0.); int PixelNbr=0;"\
		"$FCT= color+=colorTemp;PixelNbr++;"\
		"$FCT_FINAL= color.r/=PixelNbr;"\
		"color.g=floor(PixelNbr/256.)/256.;"\
		"color.b=mod(PixelNbr,256.)/256.;"\
		"gl_FragColor=color;"\
		")";
	/* .R is the average value of the region
	.G is the higher part of the pixel number (G*256+B)
	.B is the lower part of the pixel number (G*256+B)
	*/
	else if (strcmp(GLSLfunction, "AVG_ACQ")==0)//Average of all pixels from carte(where src(x,y) < threshold) && mak !=0
		FilterName+="($FCT_INIT= color=vec4(0.); int PixelNbr=0;"\
		"$FCT= "\
		"if(color.r<35./256.)"\
		"{color+=colorTemp;PixelNbr++;}"\
		"$FCT_FINAL= color.r/=PixelNbr;"\
		"color.g=floor(PixelNbr/256.)/256.;"\
		"color.b=mod(PixelNbr,256.)/256.;"\
		"gl_FragColor=color;"\
		")";
	/* .R is the average value of the region
	.G is the higher part of the pixel number (G*256+B)
	.B is the lower part of the pixel number (G*256+B)
	*/
	else if (strcmp(GLSLfunction, "minmax")==0)
	{
		FilterName+="($FCT_INIT=color=vec4(0.);"\
			"$FCT=color.r=min(colorTemp.r,color.r);color.g = max(colorTemp.r,color.r);"\
			"$FCT_FINAL=gl_FragColor = color;"\
			")";
		/* .b is the max value
		.g is the x pos of the max value
		.r is the y pos of the max value
		*/
	}
	else
	{
		string Msg = "\nWarning in Fct cvgResizeGLSLFct :  the GLSL function '";
		Msg += GLSLfunction;
		Msg += "' has not been validated yet, unknown result...";
		GPUCV_WARNING(Msg.data());
		FilterName+="($FCT_X=";
		FilterName += GLSLfunction;
		FilterName += "(colorxTemp, colorx);$FCT_Y=";
		FilterName += GLSLfunction;
		FilterName += "(coloryTemp, colory);$FCT_FINAL=";
		FilterName += GLSLfunction;
		FilterName += "(colorx, colory);";
		FilterName += ")";
	}

	// Image Dimensions
	float params[4] = {
		GetWidth(src),
		GetHeight(src),
		GetWidth(dst),
		GetHeight(dst)
	};

	TemplateOperator("cvgResizeFct", FilterName.data(), "",
		src, mask, NULL,
		dst, params, 4);

	GPUCV_STOP_OP(,
		src, dst, mask, NULL
		);
}
//=============================
void cvgInitGLView(CvArr* src)
{
	SG_Assert(src, "cvgInitGLView()=>No input image!");
	DataContainer * CLTex_Dst = GetTextureManager()->Get<DataContainer>(src);
	SG_Assert(src, "cvgInitGLView()=>Could not get corresponding DataContainer!");
	
	//flush any processing on the image
	CLTex_Dst->Flush();
	//imag is not going to be changed
	CLTex_Dst->PushSetOptions(DataContainer::DEST_IMG, false);
	DataDsc_GLTex * DataGLDst = CLTex_Dst->SetLocation<DataDsc_GLTex>(true);
	SG_Assert(src, "cvgInitGLView()=>Could not get corresponding DataDescriptor!");
	DataGLDst->InitGLView();
	//restore image settings
	CLTex_Dst->PopOptions();
}

//=============================
void cvgDrawGLQuad(CvArr* src)
{
	SG_Assert(src, "cvgDrawGLQuad()=>No input image!");
	DataContainer * DC_Src		= GPUCV_GET_TEX(src);
	//imag is not going to be changed
	DC_Src->PushSetOptions(DataContainer::DEST_IMG, false);
	DC_Src->SetOption(DataContainer::UBIQUITY, 1);
	DataDsc_GLTex * DDGLTex_Src = DC_Src->SetLocation<DataDsc_GLTex>(true);
	DDGLTex_Src->DrawFullQuad(DDGLTex_Src->_GetWidth(), DDGLTex_Src->_GetHeight());
	DC_Src->PopOptions();
}
//=============================
void cvgSetRenderToTexture(CvArr* src)
{
	SG_Assert(src, "cvgSetRenderToTexture()=>No input image!");
	DataContainer * DC_Src		= GPUCV_GET_TEX(src);
	DC_Src->SetRenderToTexture();
}
//============================================================
void cvgUnsetRenderToTexture(CvArr* src)
{
	SG_Assert(src, "cvgUnSetRenderToTexture()=>No input image!");
	DataContainer * DC_Src		= GPUCV_GET_TEX(src);
	DC_Src->UnsetRenderToTexture();
}
//============================================================
void TemplateOperator(std::string _fctName, std::string _filename1, std::string _filename2,
					  CvArr * _src1, CvArr * _src2, CvArr * _src3,
					  CvArr *_dest,
					  const float *_params/*=NULL*/, unsigned int _param_nbr/*=0*/,
					  TextureGrp::TextureGrp_CheckFlag _controlFlag/*=0*/, std::string _optionalMetaTag/*=""*/,
					  FCT_PTR_DRAW_TEXGRP(_DrawFct)/*=NULL*/)
{
	SG_Assert(_dest && _src1, "No input or ouput image");

#if 0//_GPUCV_SUPPORT_GS
	ShaderProgramNames Names;
	Names.m_ShaderNames[0] = _filename1;
	Names.m_ShaderNames[1] = _filename2;

	TemplateOperator(_fctName, Names,
#else
	TemplateOperator(_fctName, _filename1, _filename2,
#endif
		GPUCV_GET_TEX(_src1),
		(_src2)?GPUCV_GET_TEX(_src2):NULL,
		(_src3)?GPUCV_GET_TEX(_src3):NULL,
		GPUCV_GET_TEX(_dest),
		_params, _param_nbr,
		_controlFlag, _optionalMetaTag,
		_DrawFct);
	ShowOpenGLError(__FILE__, __LINE__);
}


}//namespace GCV

