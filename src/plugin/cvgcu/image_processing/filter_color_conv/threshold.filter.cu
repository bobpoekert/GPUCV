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
#include <cvgcu/config.h>
#if _GPUCV_COMPILE_CUDA
#include <cvgcu/config.h>
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>
#include <cvgcu/image_processing/filter_color_conv/threshold.filter.h>


/*!
*  Execute a threshold on a grayscale image
*  \param src -> Source array (single-channel, 8-bit of 32-bit floating point).
*  \param dst -> Destination array, must be either the same type as src or 8-bit.
*  \param threshold -> Threshold value.
*  \param max_value -> Maximum value to use with CV_THRESH_BINARY and CV_THRESH_BINARY_INV thresholding types.
*  \param threshold_type -> Thresholding type 
*/
_GPUCV_CVGCU_EXPORT_CU 
	void gcuThreshold(CvArr* _src, CvArr* _dst, double _threshold, double _max_value, int _threshold_type )
	{    
		int src_width		= gcuGetWidth(_src);
		int src_height		= gcuGetHeight(_src);
		int src_Depth		= gcuGetGLDepth(_src);
		int src_nChannels	= gcuGetnChannels(_src);

		void * d_result = gcuPreProcess(_dst, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);
		void * d_src = gcuPreProcess(_src, GCU_INPUT, CU_MEMORYTYPE_DEVICE);

		dim3 threads(16,16,1);
		dim3 blocks = dim3(iDivUp(src_width,threads.x), iDivUp(src_height,threads.y), 1);

		int NewChannels = src_nChannels;
#if 0//
		if(IS_MULTIPLE_OF(width *channels, 4))
			NewChannels = 4;
		else if(IS_MULTIPLE_OF(width *channels, 2))
			NewChannels = 2;

		if(NewChannels != channels)
		{
			src_width* =  channels / NewChannels;
		}
		//printf("\nNbr of channels after: %d", NewChannels);
#endif  
		
#define GCU_THRESHOLD_SWITCH_FCT(CHANNEL, SRC_TYPE, DST_TYPE)\
		gcuThresholdKernel_1<DST_TYPE, SRC_TYPE> <<<blocks, threads>>> \
		((SRC_TYPE*)d_src, (DST_TYPE*)d_result, (float)_threshold,(float)_max_value, _threshold_type,src_width, src_height)


		//GCU_MULTIPLEX_ALLCHANNELS_ALLFORMAT(GCU_THRESHOLD_SWITCH_FCT, NewChannels, src_Depth);
		
		//! \todo only single channel threshold is supported now
		GCU_MULTIPLEX_1CHANNELS_ALLFORMAT(GCU_THRESHOLD_SWITCH_FCT,1,src_Depth); 

		//set back to normal shape...
#if 0	
	if(NewChannels != channels)
		{
			gcuUnsetReshapeObj(src,CU_MEMORYTYPE_DEVICE);
			gcuUnsetReshapeObj(dstArr,CU_MEMORYTYPE_DEVICE);
		}
#endif
		gcuPostProcess(_src);
		gcuPostProcess(_dst);
	}
#endif
