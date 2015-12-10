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
#include <cvauxg/cvauxg.h>
#include <cvauxg/config.h>
#include <cvg/cvg.h>
#include <cxcoreg/cxcoreg.h>



using namespace GCV;

_GPUCV_CVAUXG_EXPORT_C LibraryDescriptor *modGetLibraryDescriptor(void)
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
		pLibraryDescriptor->SetDllName("cvauxg");
		pLibraryDescriptor->SetImplementationName("GLSL");
		pLibraryDescriptor->SetBaseImplementationID(GPUCV_IMPL_GLSL);
		pLibraryDescriptor->SetUseGpu(true);
		pLibraryDescriptor->SetStartColor(GPUCV_IMPL_GLSL_COLOR_START);
		pLibraryDescriptor->SetStopColor(GPUCV_IMPL_GLSL_COLOR_STOP);
	}
	return pLibraryDescriptor;
}


#if _GPUCV_DEVELOP_BETA
void cvgEigenProjection( void*     eigInput,
						int       nEigObjs,
						int       ioFlags,
						void*     userData,
						float*    coeffs, 
						IplImage* avg,
						IplImage* proj )
{
	GPUCV_START_OP(cvEigenProjection(eigInput, nEigObjs, ioFlags, userData,
		coeffs, avg,proj),
		"cvEigenProjection", 
		NULL,
		GenericGPU::HRD_PRF_2);

	//opencv checkup :
	{
		float *avg_data;
		uchar *proj_data;
		int avg_step = 0, proj_step = 0;
		CvSize avg_size, proj_size;
		int i;

		cvGetImageRawData( avg, (uchar **) & avg_data, &avg_step, &avg_size );
		if( avg->depth != IPL_DEPTH_32F )
			CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
		if( avg->nChannels != 1 )
			CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );

		cvGetImageRawData( proj, &proj_data, &proj_step, &proj_size );
		if( proj->depth != IPL_DEPTH_8U )
			CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
		if( proj->nChannels != 1 )
			CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );

		if( proj_size != avg_size )
			CV_ERROR( CV_StsBadArg, "Different sizes of projects" );
	}//opencv Checkup


	if (ioFlags == CV_EIGOBJ_NO_CALLBACK)
	{
		IplImage **eigens = (IplImage**) (((CvInput *) & eigInput)->data);
		//float **eigs = (float**) cvAlloc( sizeof( float * ) * nEigObjs );
		int eig_step = 0, old_step = 0;
		CvSize eig_size = avg_size, old_size = avg_size;

		if( eigs == NULL )
			CV_ERROR( CV_StsBadArg, "Insufficient memory" );

		TextureGrp EigenObjTexGrp;
		EigenObjTexGrp.SetGrpType(TextureGrp::TEXTGRP_INPUT);
		TextureGrp EigenOutputTexGrp;
		EigenOutputTexGrp.SetGrpType(TextureGrp::TEXTGRP_OUTPUT);


		for( i = 0; i < nEigObjs; i++ )
		{
			IplImage *eig = eigens[i];			    
			EigenObjTexGrp.AddTexture( GetTextureManager()->Get(eig, DataContainer::LOC_GPU, true));

			//float *eig_data;

			cvGetImageRawData( eig, (uchar **) & eig_data, &eig_step, &eig_size );
			if( eig->depth != IPL_DEPTH_32F )
				CV_ERROR( CV_BadDepth, cvUnsupportedFormat );
			if( eig_size != avg_size || eig_size != old_size )
				CV_ERROR( CV_StsBadArg, "Different sizes of objects" );
			if( eig->nChannels != 1 )
				CV_ERROR( CV_BadNumChannels, cvUnsupportedFormat );
			if( i > 0 && eig_step != old_step )
				CV_ERROR( CV_StsBadArg, "Different steps of objects" );

			old_step = eig_step;
			old_size = eig_size;
			eigs[i] = eig_data;
		}

		float *FilterParams = NULL;
		int ParamNbr = 0;

		if(nEigObjs < EigenObjTexGrp.GetTexMaxNbr())
		{//first implementation: using multi-texturing when there less Eigen Obj than the maximum texture supported by hardware
			GPUCV_DEBUG("cvgEigenProjection(): Having " << nEigObjs << "Objects, can fit "
				<< " into " << EigenObjTexGrp.GetTexMaxNbr() << " textures");
		}
		else 
		{
			CvSize ObjPerTexture;
			ObjPerTexture.width		= (int)_GPUCV_TEXTURE_MAX_SIZE_X/eig_size.width;
			ObjPerTexture.height	= (int)_GPUCV_TEXTURE_MAX_SIZE_X/eig_size.height;
			int ObjNbrPerTex = ObjPerTexture.width * ObjPerTexture.height;
			int NbrTexneeded = nEigObjs / ObjNbrPerTex;
			if(ObjNbrPerTex >= nEigObjs)
			{//we try to fit as mush eigen objects into one texture(using subtextures)
				GPUCV_DEBUG("cvgEigenProjection(): Having " << nEigObjs << "Objects, can fit "
					<< ObjPerTexture.width	<<"*" << ObjPerTexture.height << " into a single texture");
			}
			else if (ObjNbrPerTex * EigenObjTexGrp.GetTexMaxNbr() >= nEigObjs)
			{//we try to fit as mush eigen objects into several texture(using subtextures)
				GPUCV_DEBUG("cvgEigenProjection(): Having " << nEigObjs << "Objects, can fit "
					<< ObjPerTexture.width	<<"*" << ObjPerTexture.height << " into a single texture, so we are using "
					<< (float)nEigObjs / ObjNbrPerTex << " textures");
			}
			else
			{//multi pass
				SG_Assert(0, "cvgEigenProjection(): to many Objects to be process on GPU");
			}
		}



		//all images are luminances... can it be changed to RGBA..?power of 4?
		float Params[] = {
			nEigObjs,	//nbr of eigen object
			1,			//nbr of eigen object per texture
			eig_size.width,	//width of the eigen object in its texture(in text coord value)
			eig_size.height//height of the eigen object in its texture(in text coord value)
		};
		TemplateOperator("cvgEigenProjection", "FShaders/eigen_projection", "",
			EigenObjTexGrp, EigenOutputTexGrp,
			Params, sizeof(Params),
			TextureGrp::TEXTGRP_NO_CONTROL/* TEXTGRP_SAME_ALL_FORMAT*/, 
			"");

		//GPU PART START=======================================
		/*icvEigenProjection_8u32fR( nEigObjs,(void*) eigs,
		eig_step, ioFlags,
		userData, coeffs,
		avg_data, avg_step,
		proj_data, proj_step,
		avg_size   );*/



		//GPU PART STOP========================================
		cvFree( &eigs );
	}

	GPUCV_STOP_OP(
		cvEigenProjection(eigInput, nEigObjs, ioFlags, userData,
		coeffs, avg,proj);,
		avg, NULL, NULL, NULL		
		);
}


#endif
