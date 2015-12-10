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
/**
*	\brief Supply some operators that are not part of OpenCV and that can be use to control and tweak GpuCV programs.
*	\author Yannick Allusse
*/

#ifndef __GPUCV_MISC_H
#define __GPUCV_MISC_H
#include <GPUCV/config.h>
#include <includecv.h>

#ifdef __cplusplus
	#include <GPUCV/toolscvg.h>
	#include <GPUCV/cvgArr.h>
#endif


/** @defgroup CVG_MISC_GRP Additional functions and operators relative to GpuCV.
*	@ingroup GPUCV_LIB_GRP
*	@{ */
#ifdef __cplusplus
	namespace GCV{
#endif// __cplusplus

/*!
*   \brief Initialize GpuCV library and framework, test extensions, and create or use a GL context.
*   \param InitGLContext -> define if library should create its own GL context or use an existing one
*   \param isMultiThread -> allow to use GPUCV library on multi-thread programs (in development)
*   \return int -> status[??]
*   \sa cvgTerminate(), GpuCVInit(), GpuCVTerminate()
*   \author Yannick Allusse
*/
_GPUCV_EXPORT_C
int  cvgInit(unsigned char InitGLContext CV_DEFAULT(true), unsigned char isMultiThread CV_DEFAULT(false));

/*!
*	\brief Close GpuCV library and framework, release manager and save benchmarks.
*	\sa cvgInit(), GpuCVInit(), GpuCVTerminate()
*   \author Yannick Allusse
*/
_GPUCV_EXPORT_C
void cvgTerminate();


/**
*	\brief Synchronize CvArr image between CPU and GPU, it makes sure the data are on CPU.
*	\param src	=> source array to synchronize.
*   \author Yannick Allusse
*/
_GPUCV_EXPORT_C
void cvgSynchronize(CvArr* src);

	/** Call the flush function of all Data Descriptors.
	*	\sa DataDsc_Base::Flush(), DataContainer::Flush()
	*/
_GPUCV_EXPORT_C
void cvgFlush(CvArr* _image);

/**
*	\brief Define the path where the library will find "VShaders" and "FShaders" folders.
*   \author Yannick Allusse
*	\return Path string.
*	\sa cvgGetShaderPath(), cvgRetrieveShaderPath()
*/
_GPUCV_EXPORT_C
const char * cvgSetShaderPath(const char * _path);

/**
*	\brief Return the defined path where the library will find "VShaders" and "FShaders" folders.
*   \author Yannick Allusse
*	\return Path string.
*	\sa cvgSetShaderPath(), cvgRetrieveShaderPath()
*/
_GPUCV_EXPORT_C
const char * cvgGetShaderPath();

/**
*	\brief Try to find the path to gpucv/data/ and set it as the Shader path (cvgSetShaderPath()).
Return the defined path where the library will find "VShaders" and "FShaders" folders.
*   \author Yannick Allusse
*	\return A reference to path string.
*	\sa cvgGetShaderPath(), cvgSetShaderPath()
Example:
\code
int main(int argc, char **argv) 
{
	...
	std::string DataPath;
	GPUCV_NOTICE("Current application path: " << argv[0]);
	std::string DataPath = cvgRetrieveShaderPath(argv[0]);
	GPUCV_NOTICE("GpuCV shader and data path: "<< DataPath);
	...
}
\endcode
*/
_GPUCV_EXPORT_C
const char * cvgRetrieveShaderPath(const char * appPath);


#ifdef __cplusplus
/**
*	\brief Smart move/copy the array data to the given location, allocate memory if required.
*	\param src	=> source array to synchronize.
*	\param TType => data destination, see DataContainer::TextureLocation.
*	\param _dataTransfer => specify if the data must be copied(true), or if we just allocate destination memory(false).
*   \author Yannick Allusse
*	\sa DataContainer::SetLocation()
*/
template<typename TType>
void cvgSetLocation(CvArr* src, bool _dataTransfer CV_DEFAULT(true))
{
#if 0
	std::string Name = std::string("cvgSetLocation<")+std::string(typeid(TType).name())+std::string(">()");
#else
	std::string Name = std::string("cvgSetLocation()");//previous declaration gives error with the benchmarking tools and class '::' definitions...
#endif
	GPUCV_START_OP(,
		Name.data(),
		src,
		GenericGPU::HRD_PRF_0);

	SG_Assert(src, "Input image is NULL");
	//add tracing parameters
#if 0//_GPUCV_PROFILE
	if(_PROFILE_PARAMS)
	{
		int ParamIntVal = (_dataTransfer)?1:0;
		_PROFILE_PARAMS->AddParam("data_transfer", ParamIntVal, 1);
	}
#endif
	//=======================
	CvgArr *tempImg = (CvgArr*)GPUCV_GET_TEX(src);
	SG_Assert(tempImg, "Could not find IplImage");
	tempImg->SetLocation<TType>(_dataTransfer);

	GPUCV_STOP_OP(
		_GPUCV_NOP,
		src, NULL, NULL, NULL
		);
}

template<typename TType>
long int cvgSetLocationEstimation(CvArr* src, char & _bestSrcId, bool _dataTransfer CV_DEFAULT(true))
{
	long int Result=0;
	std::string Name=std::string("cvgSetLocationEstimation<")+std::string(typeid(TType).name())+std::string(">()");
	GPUCV_START_OP(,
		Name.data(),
		src,
		GenericGPU::HRD_PRF_1);

	SG_Assert(src, "Input image is NULL");
	CvgArr *tempImg = (CvgArr*)GPUCV_GET_TEX(src);
	SG_Assert(tempImg, "Could not find IplImage");
	Result = tempImg->SetLocationEstimation<TType>(_bestSrcId, _dataTransfer);

	GPUCV_STOP_OP(
		_GPUCV_NOP,
		src, NULL, NULL, NULL
		);
	return Result;
}


/**
*	\brief Force the data flag location.
*	\param src	=> source array.
*	\param _dataFlag => value of the data flag, [have data/do not have data]
*	\param _forceUniqueData => force unique data flag, all other location that have data flag ON will be discarded.
*   \author Yannick Allusse
*	\sa DataContainer::SetDataFlag()
*/
template<typename TType>
void cvgSetDataFlag(CvArr* src, bool _dataFlag, bool _forceUniqueData CV_DEFAULT(false))
{
	SG_Assert(src, "Input image is NULL");
	CvgArr *tempImg = (CvgArr*)GPUCV_GET_TEX(src);
	SG_Assert(tempImg, "Could not find IplImage");
	tempImg->SetDataFlag<TType>(_dataFlag,  _forceUniqueData);
}

/**
*	\brief Define a label for the given image.
*	\sa DataContainer::SetLabel(), cvgGetLabel().
*   \author Yannick Allusse
*/
_GPUCV_EXPORT_C
void cvgSetLabel(CvArr* _image, std::string _label);

/**
*	\brief Get the image label.
*	\sa DataContainer::GetLabel(), cvgSetLabel().
*   \author Yannick Allusse
*/
_GPUCV_EXPORT_C
const char * cvgGetLabel(CvArr* _image);

/**
*	\brief Set the image option _opt to _val.
*	\sa CL_Option::SetOption().
*   \author Yannick Allusse
*/
_GPUCV_EXPORT_C
void cvgSetOptions(CvArr* _arr, GCV::CL_Options::OPTION_TYPE _opt, bool _val);

/**
*	\brief Push all image options and set option _opt to _val.
*	\sa CL_Option::PushSetOption().
*   \author Yannick Allusse
*/
_GPUCV_EXPORT_C
void cvgPushSetOptions(CvArr* _arr, GCV::CL_Options::OPTION_TYPE _opt, bool _val);

/**
*	\brief Get image option _opt.
*	\sa CL_Option::GetOption().
*   \author Yannick Allusse
*/
_GPUCV_EXPORT_C
int cvgGetOptions(CvArr* _arr, GCV::CL_Options::OPTION_TYPE _opt);

/**
*	\brief Pop all image options.
*	\sa CL_Option::PopOption().
*   \author Yannick Allusse
*/
_GPUCV_EXPORT_C
void cvgPopOptions(CvArr* _arr);

#endif//C++
_GPUCV_EXPORT_C
void cvgShowImageProperties(CvArr* _arr);

/*!
*   \brief This function set default comportment for CvArr to be recovered from GPU to CPU after all filter processing (and do this memory conversion now if necessary).
*   \param arr -> CvArr to manage
*   \author Jean-Philippe Farrugia
*	\sa cvgUnsetCpuReturn().
*	\deprecated Please use cvgSetOptions(arr, DataContainer::UBIQUITY, true) instead.
*/
_GPUCV_EXPORT_C
void cvgSetCpuReturn(CvArr *arr);

/*!
*   \brief This function unset default comportment for CvArr to be recovered from GPU to CPU after all filter processing.
*   \param arr -> CvArr to manage
*   \author Jean-Philippe Farrugia
*   \author Yannick Allusse
*	\sa cvgSetCpuReturn().
*	\deprecated Please use cvgSetOptions(arr, DataContainer::UBIQUITY, false) instead.
*/
_GPUCV_EXPORT_C
void cvgUnsetCpuReturn(CvArr *arr);

/*!
*	\brief The function draw the histogram into the image, no GPU processing is planed on this function.
*	This function draw the histogram into a 3 channels image using the given histogram. Lines are drawn for
*	every bins with given color.
*	\param img -> Destination image.
*	\param hist -> Histogram to draw.
*	\param color -> Color of the histogram lines.
*	\author Yannick Allusse
*	\todo Number of buckets is fixed to 256 => add automatic buckets detection using the histogram.
*/
_GPUCV_EXPORT_C
void cvgCreateHistImage(IplImage  * img, CvHistogram *hist, CvScalar color);

/*!
*	\brief Similar to <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_cv.htm#decl_cvResize" target=new>cvResize</a> function but use the supply GLSL function to determine pixels results.
*	\param src -> source image
*	\param dst -> destination image
*	\param GLSLfunction -> GLSL function having two 'vec4' parameters, examples:
* 	<ul><li>min - minimum nearest-neigbor,</li>
*	<li>max - maximum nearest-neigbor,</li>
*	</ul>
*	\param mask => mask array.
*	\warning Only the default interpolation method(CV_INTER_LINEAR) have been ported to GPU.
*	\author Yannick Allusse
*/
_GPUCV_EXPORT_C
void cvgResizeGLSLFct(  CvArr* src, CvArr* dst, char * GLSLfunction,  CvArr* mask CV_DEFAULT(NULL));


_GPUCV_EXPORT_C
void cvgInitGLView(CvArr* src);


/*!
*	\brief Draw a GL quad using the corresponding source array.
*	\param src -> source array
*	It make sure we have a OpenGL texture(DataDsc_GLTex) and draw a quad with vertexes coordinates GPUCVDftQuadVertexes.
*	It will be drawn as full screen, glScale/glRotate/glTranslate can be used to modify the quad size/orientation/position.
*	\author Yannick Allusse
*/
_GPUCV_EXPORT_C
void cvgDrawGLQuad(CvArr* src);


_GPUCV_EXPORT_C
void cvgSetRenderToTexture(CvArr* src);

_GPUCV_EXPORT_C
void cvgUnsetRenderToTexture(CvArr* src);


#ifdef __cplusplus

/**
*	\brief Generic function to call simple shader operator using 1 or two input images, an optional mask image, some parameters and some optional meta parameters for the shader.
*	TemplateOperator is a generic function to call simple shaders like arithmetic and algebra operators.
*	\param _fctName	=> Name of the function which is calling TemplateOperator, ex:"cvAdd"
*	\param _filename1 => Fragment shader filename, including path, but without the 'frag' extension.
*	\param _filename2 => Vertex shader filename, including path, but without the 'vert' extension.
*	\param _src1 => Main Input CvArr.
*	\param _src2 => Optional Input CvArr.
*	\param _src3 => Optional Input CvArr, often used as mask.
*	\param _dest => Destination CvArr.
*	\param _params => Optional float[] parameters.
*	\param _param_nbr => Number of optional float[] parameters.
*	\param _controlFlag	=> Control flag used to check that image need to have some properties in common, like size, format...
*	\param _optionalMetaTag => String that defines some macro or preprocessor to be used in the fragment shader file.
*	\note This function should be use only with simple operators, but complex operators should base their structure on it.
*	\author Yannick Allusse
*/
_GPUCV_EXPORT_C
void TemplateOperator(std::string _fctName,
					  std::string _filename1, std::string _filename2,
					  CvArr * _src1, CvArr * _src2, CvArr * _src3,
					  CvArr *_dest,
					  const float *_params CV_DEFAULT(NULL), unsigned int _param_nbr CV_DEFAULT(0),
					  GCV::TextureGrp::TextureGrp_CheckFlag _controlFlag CV_DEFAULT(GCV::TextureGrp::TEXTGRP_NO_CONTROL), std::string _optionalMetaTag CV_DEFAULT(""),
					  FCT_PTR_DRAW_TEXGRP(_DrawFct) CV_DEFAULT(NULL));


#endif//C++


#ifdef __cplusplus

}//namespace GCV
#endif//c++
/**	@} */

#endif
