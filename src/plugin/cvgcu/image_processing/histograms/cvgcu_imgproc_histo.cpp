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
#include <cvgcu/cvgcu.h>
//=================================
//CVG_IMGPROC__HISTOGRAM_GRP
//=================================
#if _GPUCV_COMPILE_CUDA

#include <cxcoreg/cxcoreg.h>
#include <vector_types.h>
#include <SugoiTracer/tracer_c.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <GPUCVTexture/fbo.h>

using namespace GCV;



_GPUCV_CVGCU_EXPORT_CU
	void gcuCalcHist(CvArr ** _src, 
		CvArr* hist, int accumulate=0, const CvArr* mask=NULL);
//============================================================
void cvgCudaCalcHist( IplImage** image, CvHistogram* hist, int accumulate/*=0*/, const CvArr* mask/*=NULL*/ )
{
	GPUCV_START_OP(cvCalcHist(image, hist, accumulate,  mask),
		"cvgCudaCalcHist", 
		*image,
		GenericGPU::HRD_PRF_CUDA);

	GCV_OPER_TODO_ASSERT(!mask, "mask not supported yet!");
	GCV_OPER_ASSERT(image, "no input images!");
	GCV_OPER_ASSERT(*image, "no input image!");
	GCV_OPER_ASSERT(hist, "no histogram!");

	int iBins = hist->mat.dim[0].size;
	GCV_OPER_COMPAT_ASSERT((iBins == 64) || (iBins == 256), "Histogram bins number must be 64 or 256 to run CUDA implementation.");
	GCV_OPER_COMPAT_ASSERT((*image)->nChannels == 1, "Input image is not single channel");

	//Create temporary image to store the histogram...
	CvSize HistoBinsSize;
	HistoBinsSize.width = iBins;
	HistoBinsSize.height=1;
	IplImage * pIplHistoBins = cvgCreateImage(HistoBinsSize, IPL_DEPTH_32S, 1);//no need to init...
	

	//call Cuda operator
	gcuCalcHist((void**)image, (void*)pIplHistoBins, accumulate, mask);


	//read back data to CPU
	//and filling histogram data
	int * pData = (int*)pIplHistoBins->imageData; //we get local data here without sync cause we know gcuCalcHist get data back to cpu
	unsigned int TotalHisto=0;
	for (int j=0; j< hist->mat.dim[0].size; j++)
	{
		((CvMatND*)(&hist->mat))->data.fl[j] = (unsigned int)pData[j];
//		GPUCV_DEBUG("cudaHisto:" << j << "=>" << ((CvMatND*)(&hist->mat))->data.fl[j]);
		TotalHisto += (unsigned int)((CvMatND*)(&hist->mat))->data.fl[j];
	}
	GPUCV_DEBUG("TotalHisto:" << TotalHisto);
	
	cvgReleaseImage(&pIplHistoBins);
	
	GPUCV_STOP_OP(
		cvCalcHist(image, hist, accumulate,  mask),
		NULL, NULL, NULL, NULL
		);
}


#endif//COMPILE_CUDA

