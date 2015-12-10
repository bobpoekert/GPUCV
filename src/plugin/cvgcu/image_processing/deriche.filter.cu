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
#include <GPUCVCuda/gpucv_wrapper_c.h>
#include <GPUCVCuda/base_kernels/config.kernel.h>

#if _GPUCV_COMPILE_CUDA

#include <cvgcu/image_processing/deriche.filter.h>
#include <cxcoregcu/oper_array/linear_algebra/transpose.filter.h>

#define CvArr void
void cvgcuDericheExtrema(CvArr* B, float* Gx, float* Gy);

#define  GCU_DERICHE_TYPE float

#define GCU_DERICHE_SHOW_IMAGE(NAME, PTR) 	gcuShowImage(NAME, width, height,gcuGetCVDepth(srcArr)/*IPL_DEPTH_32F:32*/ , src_nChannels, PTR, sizeof(GCU_DERICHE_TYPE), 1)
void cv_DericheExtrema(CvArr* B, float* Gx, float* Gy);

_GPUCV_CVGCU_EXPORT_CU
void gcuDeriche(CvArr* srcArr, CvArr* dstArr, double alpha ) 
{

	unsigned int height		= gcuGetHeight(srcArr);
	unsigned int width		= gcuGetWidth(srcArr);
	unsigned int src_nChannels= gcuGetnChannels(srcArr);

	GCU_DERICHE_TYPE *	d_dst, *d_src, *d_srcT;
	GCU_DERICHE_TYPE *	d_liss;
	GCU_DERICHE_TYPE *	d_lissT, *d_derivT;
	GCU_DERICHE_TYPE *	d_X, *d_Y;
	GCU_DERICHE_TYPE *	d_Extrema=NULL, *Tre;

	d_src = (GCU_DERICHE_TYPE * )	gcuPreProcess(srcArr, GCU_INPUT, CU_MEMORYTYPE_DEVICE);
	d_dst = (GCU_DERICHE_TYPE *)	gcuPreProcess(dstArr, GCU_OUTPUT, CU_MEMORYTYPE_DEVICE);

	GCU_DERICHE_SHOW_IMAGE("srcTest", d_src);
	//gcuShowImage("srcTest", width, height, gcuGetCVDepth(srcArr), src_nChannels, d_src);

	size_t pitch_temp=0;


	// allocate memory for tmp img
	gcudaMallocPitch((void **)&d_liss,	&pitch_temp, width*1*sizeof(GCU_DERICHE_TYPE), height);	
	gcudaMallocPitch((void **)&d_Y,		&pitch_temp, width*1*sizeof(GCU_DERICHE_TYPE), height);

	//transpose of source	
	gcudaMallocPitch((void **)&d_srcT , &pitch_temp, height*1*sizeof(GCU_DERICHE_TYPE), width); 
	gcudaMallocPitch((void **)&d_lissT,	&pitch_temp, height*1*sizeof(GCU_DERICHE_TYPE), width);	
	gcudaMallocPitch((void **)&d_derivT,&pitch_temp, height*1*sizeof(GCU_DERICHE_TYPE), width);	
	gcudaMallocPitch((void **)&d_X ,	&pitch_temp, width*1*sizeof(GCU_DERICHE_TYPE), height);
	gcudaMallocPitch((void **)&Tre,		&pitch_temp, (width)*1*sizeof(GCU_DERICHE_TYPE), height);

#if !DERICHE_OPTIMIZED
	dim3 threads(1,1,1);
	if(IS_MULTIPLE_OF(width,128)) threads.x = 128;
	else if(IS_MULTIPLE_OF(width,64)) threads.x = 64;
	else if(IS_MULTIPLE_OF(width,32)) threads.x = 32;
	else if(IS_MULTIPLE_OF(width,16)) threads.x = 16;
	else if (IS_MULTIPLE_OF(width,16)) threads.x = 8;
	else if (IS_MULTIPLE_OF(width,4)) threads.x = 4;
	else threads.x = width;
#else
	dim3 threads(128,2,1);
	while(threads.x > width/4)
		threads.x /=4;

	if(threads.x <2)
		printf("\n Error in gcuDeriche(): Image size is too small");
#endif

	dim3 threadsTranspose(16,16,1);
	//thread for transposition
	while(threadsTranspose.x > width/2)
	{
		threadsTranspose.x /=2;
		threadsTranspose.y *=2;
	}
	while(threadsTranspose.y > height/2)
	{
		threadsTranspose.y /=2;
	}
	if(threads.x =0)
		printf("\n Error in gcuDeriche(): Image size is too small");

	dim3 blocksTranspose = dim3(iDivUp(width,threadsTranspose.x),iDivUp(height,threadsTranspose.y), 1);


	size_t Pitch = gcuGetPitch(srcArr)/ sizeof(GCU_DERICHE_TYPE);
	//coeficients for lissage in X and Y
	float liss_k, liss_a0, liss_a1, liss_a2, liss_a3, liss_b1, liss_b2 ;
	liss_k = (1-exp(-alpha))*(1-exp(-alpha))
		/(1+2*alpha*exp(-alpha)-exp(-2*alpha));
	liss_a0 = liss_k;
	liss_a1 = liss_k*(alpha-1)*exp(-alpha);
	liss_a2 = liss_k*(alpha+1)*exp(-alpha);
	liss_a3 = -liss_k*exp(-2*alpha);
	liss_b1 = -2*exp(-alpha);
	liss_b2 = exp(-2*alpha);
#if 0// _DEBUG
	printf("\nLissage=================:");
	printf("\nalpha:%f", alpha);
	printf("\nk:%f", liss_k);
	printf("\na0:%f", liss_a0);
	printf("\na1:%f", liss_a1);
	printf("\na2:%f", liss_a2);
	printf("\na3:%f", liss_a3);
	printf("\nb1:%f", liss_b1);
	printf("\nb2:%f", liss_b2);
	printf("\n=======================");
#endif
	//===================================
	//coeficients for derivation in X and Y
	float deriv_k, deriv_a0, deriv_b1, deriv_b2 ;
	deriv_k = exp(-alpha);
	deriv_a0 = (1-deriv_k)*(1-deriv_k);
	deriv_b1 = -2*deriv_k;
	deriv_b2 = deriv_k*deriv_k;
#if 0// _DEBUG
	printf("\nDerivation=============:");
	printf("\nalpha:%f", alpha);
	printf("\nk:%f", deriv_k);
	printf("\na0:%f", deriv_a0);
	printf("\nb1:%f", deriv_b1);
	printf("\nb2:%f", deriv_b2);
	printf("\n=======================");
#endif
	//===================================



	//prepare transposition
#if 0//_DEBUG
	printf("\nTranspose block nbr:%d %d %d", blocksTranspose.x, blocksTranspose.y, blocksTranspose.z);
	printf("\nTranspose thread nbr:%d %d %d", threadsTranspose.x, threadsTranspose.y, threadsTranspose.z);
#endif
	gcuTransposeKernel_Shared<float1,float1, 1,16, 16> <<<blocksTranspose, threadsTranspose,0>>>(
		(float1*)d_src
		,(float1*) ((1)?d_srcT:d_dst)
		/*,Pitch*/, width, height);
	GCU_DERICHE_SHOW_IMAGE("transpose 1", d_srcT);


#if DERICHE_SIMPLE_PROC
	printf("\nDericheSimpleProc\n");
	// X =======================================
	// one lissage and one derivation on X
	//dim3 blocks = dim3(width/threads.x,height / threads.y, 1);
	dim3 blocks = dim3(	iDivUp(width,threads.x),
		1., 1);

	//OK
	gcuDericheLissageKernel<<<blocks, threads>>>((GCU_DERICHE_TYPE *) d_srcT
		,(GCU_DERICHE_TYPE *) (1)?d_lissT:d_dst
		,height, width
		,liss_k, liss_a0, liss_a1, liss_a2, liss_a3, liss_b1, liss_b2);

	//ok
	gcuDericheDerivationKernel<<<blocks, threads>>>(d_lissT
		,(1)?d_derivT:d_dst
		, width, height, deriv_k, deriv_a0, deriv_b1, deriv_b2) ;

	//ok, transpose back results
	gcuTransposeKernel_Shared<float1,float1, 1,16, 16> <<<blocksTranspose, threadsTranspose,0>>>((float1 *) d_derivT
		, (1)?(float1*)d_X:(float1*)d_dst
		, Pitch, width, height);


	// Y =======================================
	// one lissage and one derivation on Y

	//ok
	gcuDericheLissageKernel<<<blocks, threads>>>((GCU_DERICHE_TYPE *) d_src 
		,(GCU_DERICHE_TYPE *)d_liss
		, width, height
		, liss_k, liss_a0, liss_a1, liss_a2, liss_a3, liss_b1, liss_b2) ;
	//ok
	gcuDericheDerivationKernel<<<blocks, threads>>>(d_liss//Isy
		,(1)?d_Y:d_dst//d_dst//Tmp1
		,width, height
		,deriv_k, deriv_a0, deriv_b1, deriv_b2) ;

#else
	
	printf("\n!!DericheSimpleProc\n");
	// blocks and threads calculs for "dual processing"
	threads.x = 256/4;
	threads.y = 4;
	threads.z = 1;

	dim3 blocks = dim3(	iDivUp(width,threads.x),
		1./*height / threads.y*/, 1);


#if 0//it is not lissage X => derivation X but lissage X derivation Y!!!
	// lissage Y on d_src and d_srcT
	gcuDericheLissageKernel<<<blocks, threads>>>(d_src, d_srcT 
		,(1)?d_liss:d_dst
		,(1)?d_lissT:d_dst 
		,width, height
		,liss_k, liss_a0, liss_a1, liss_a2, liss_a3, liss_b1, liss_b2, 1.);


	// derivation Y on (lissage(d_src))t and (lissage(d_srcT))t
	gcuDericheDerivationKernel<<<blocks, threads>>>(d_liss,d_lissT
		,(1)?d_Y:d_dst
		,(1)?d_derivT:d_dst
		,width, height
		,deriv_k, deriv_a0, deriv_b1, deriv_b2);

	gcuTransposeKernel<float1,float1, 1,16, 16> <<<blocksTranspose, threadsTranspose,0>>>(
		(float1 *) d_derivT//d_liss
		,(1)?(float1*)d_X:(float1*)d_dst
		/*,Pitch*/, width, height);

#else//lissage X derivation Y!!!
	
	GCU_DERICHE_TYPE *	d_tempX, *d_tempY;
	// allocate memory for tmp img
	gcudaMallocPitch((void **)&d_tempX,	&pitch_temp, width*1*sizeof(GCU_DERICHE_TYPE), height);	
	gcudaMallocPitch((void **)&d_tempY,	&pitch_temp, height*1*sizeof(GCU_DERICHE_TYPE), width);


	printf("\nLissage 1\n");
	//perform both lissage on d_src and d_srcT
	gcuDericheLissageKernel<<<blocks, threads>>>(d_src, d_srcT 
		,(1)?d_liss:d_dst
		,(1)?d_lissT:d_dst 
		,width, height
		,liss_k, liss_a0, liss_a1, liss_a2, liss_a3, liss_b1, liss_b2, 1.);
	//GCU_DERICHE_SHOW_IMAGE("Lissage1-d_liss", d_liss);
	//GCU_DERICHE_SHOW_IMAGE("Lissage1-d_lissT", d_lissT);

	printf("\nTranspose 1\n");
	gcuTransposeKernel_Shared<float1,float1, 1,16, 16> <<<blocksTranspose, threadsTranspose,0>>>(
		(float1 *) d_liss//d_liss
		,(float1*)((1)?d_tempX:d_dst)
		/*,Pitch*/, width, height);
	//GCU_DERICHE_SHOW_IMAGE("Transpose 1-d_tempX", d_tempX);


	printf("\nTranspose 2\n");
	gcuTransposeKernel_Shared<float1,float1, 1,16, 16> <<<blocksTranspose, threadsTranspose,0>>>(
		(float1 *) d_lissT//d_liss
		,(float1*)((1)?d_tempY:d_dst)
		/*,Pitch*/, width, height);
	//GCU_DERICHE_SHOW_IMAGE("Transpose 2-d_tempY", d_tempY);


	printf("\nLissage 2\n");
	// derivation Y on (lissage(d_src))t and (lissage(d_srcT))t
	gcuDericheDerivationKernel<<<blocks, threads>>>(d_tempX,d_tempY
		,(1)?d_derivT:d_dst
		,(1)?d_Y:d_dst
		,width, height
		,deriv_k, deriv_a0, deriv_b1, deriv_b2);
	GCU_DERICHE_SHOW_IMAGE("Lissage2-d_derivT", d_derivT);
	GCU_DERICHE_SHOW_IMAGE("Lissage2-d_Y", d_Y);



	printf("\nTranspose 3\n");
	gcuTransposeKernel_Shared<float1,float1, 1,16, 16> <<<blocksTranspose, threadsTranspose,0>>>(
		(float1 *) d_derivT//d_liss
		,(float1*)((1)?d_X:d_dst)
		/*,Pitch*/, width, height);
	//GCU_DERICHE_SHOW_IMAGE("Transpose 3-d_X", d_X);
#endif
#endif


#if 1//extrema on GPU
	// extrema of gradient
	dim3 ExtremaSize = dim3(1,16,1);
	threads.x = 16;
	threads.y = 16;

	while(threads.x*ExtremaSize.x> width)
	{
		threads.x /=2;
		threads.y *=2;
	}
	while(threads.y*ExtremaSize.y> height)
	{
		threads.y /=2;
	}
	dim3 blocksExtram = dim3(iDivUp(width,threads.x*ExtremaSize.x),
		iDivUp(height,threads.y*ExtremaSize.y), 1);

#if 0//_DEBUG
	printf("\nExtrema block nbr:%d %d %d", blocksExtram.x, blocksExtram.y, blocksExtram.z);
	printf("\nExtrema thread nbr:%d %d %d", threads.x, threads.y, threads.z);
	printf("\nExtrema total results nbr:%d %d %d", blocksExtram.x*threads.x, blocksExtram.y*threads.y, blocksExtram.z*threads.z);
#endif

	gcudaMalloc((void **)&d_Extrema ,blocksExtram.x*threads.x*1*sizeof(GCU_DERICHE_TYPE) *  blocksExtram.y*threads.y);
	float Max = 0;	
#if 1	 
	gcuDericheExtremaKernel<<<blocksExtram, threads>>>(d_X, d_Y, 
		(1)?Tre:d_dst,
		d_Extrema,
		width, height, 1.45
		,ExtremaSize.x,ExtremaSize.y
		) ;
	GCU_DERICHE_SHOW_IMAGE("Extrema-Tre", d_dst);
	GCU_DERICHE_SHOW_IMAGE("Extrema-d_Extrema", d_Extrema);

	int memsize = blocksExtram.x*threads.x*blocksExtram.y*threads.y*1;
	//printf("\nExtrema memory:%d", memsize, blocksExtram.y*threads.y, blocksExtram.z*threads.z);
#else
	gcuDericheExtremaKernel<<<blocksTranspose, threadsTranspose>>>(d_X, d_Y,
		(0)?Tre:d_dst,
		d_Extrema,
		width, height, 1.45
		,ExtremaSize.x,ExtremaSize.y
		);
	int memsize = width*height*1;
	GCU_DERICHE_SHOW_IMAGE("Extrema-Tre", d_dst);
	GCU_DERICHE_SHOW_IMAGE("Extrema-d_Extrema", d_Extrema);
#endif		
	gcuPostProcess(dstArr);
#else //extrema on CPU
	
	float *h_X = new float [width*height*1*sizeof(GCU_DERICHE_TYPE)];
	float *h_Y = new float [width*height*1*sizeof(GCU_DERICHE_TYPE)];//(float *)malloc(width*height*1*sizeof(GCU_DERICHE_TYPE));
	gcudaThreadSynchronize();
	gcudaMemCopyDeviceToHost(h_X, (1)?d_X:Tre, width*height*1*sizeof(GCU_DERICHE_TYPE));
	gcudaMemCopyDeviceToHost(h_Y, (1)?d_Y:Tre, width*height*1*sizeof(GCU_DERICHE_TYPE));
	gcudaThreadSynchronize();
	cv_DericheExtrema(dstArr, h_X, h_Y);
	delete h_X;
	delete h_Y;

#endif

#if 1
	float * ExtremaBuffer = (float*) malloc(memsize*sizeof(float));
	gcudaMemCopyDeviceToHost(ExtremaBuffer, (1)?d_Extrema:Tre, memsize*sizeof(float));
	for(int i =0; i < memsize; i++)
	{
		//printf("\n %d Cur val: %f", i, ExtremaBuffer[i]);
		if(Max < ExtremaBuffer[i])
		{
			//			printf("\n %d New Max: %f -> %f", i, Max, ExtremaBuffer[i]);
			Max = ExtremaBuffer[i];
		}
	}
	printf("\n Final Max: %f", Max);
	//=================================
 


	// normalisation : check that result is allright
	gcuDericheExtensionKernel<<<blocksTranspose, threadsTranspose>>>(
		Tre,d_dst
		,width, height
		,Max);
#endif
	gcuPostProcess(srcArr);

	gcudaFree(d_srcT);
	gcudaFree(d_liss);
	gcudaFree(d_lissT);
	gcudaFree(d_derivT);
	gcudaFree(d_X);
	gcudaFree(d_Y);
	gcudaFree(Tre);
	gcudaFree(d_Extrema);
}


typedef float DERICHE_TYPE;

#if 0
void cv_DericheExtrema(CvArr* B, float* Gx, float* Gy)
{
	DERICHE_TYPE* B_Data	= (DERICHE_TYPE*)((IplImage*)B)->imageData;
	unsigned int dst_Width	= ((IplImage*)B)->width;
	unsigned int dst_Height = ((IplImage*)B)->height;

	DERICHE_TYPE* Gx_Data	= Gx;
	unsigned int Gx_Width	= dst_Width;
	unsigned int Gx_Height  = dst_Height;

	DERICHE_TYPE* Gy_Data	= Gy;
	unsigned int Gy_Width	= dst_Width;
	unsigned int Gy_Height  = dst_Height;



	int i,j;
	int in,jn,im,jm;
	double k;
	float Ixval, Iyval, IxvalInJn, IyvalInJn, IxvalImJm, IyvalImJm, MulTemp, MulTempN, MulTempM;
	int dx, dy;
	float dist = 1.45;
	int Gxstep  = Gx_Width;//Gx->step/sizeof(float);
	// float *Gxdata = Gx->data.fl;
	int Gystep  = Gy_Width;//Gy->step/sizeof(float);
	// float *Gy_Data = Gy->data.fl;

	// float*B_data = (float*)((IplImage*)B)->imageData;

	for (i=0;i<dst_Height;i++)
	{
		for (j=0;j<dst_Width;j++)
		{

			Iyval = (Gy_Data+i*Gystep)[j];  //cvmGet(Gy,i,j); //Gy[i][j];
			Ixval = (Gx_Data+i*Gxstep)[j];  //cvmGet(Gx,i,j); //Gx[i][j];

			MulTemp = sqrt(Iyval*Iyval + Ixval*Ixval);
			dx = (int)(Ixval/MulTemp*dist);
			dy = (int)(Iyval/MulTemp*dist);

			in = i - (int) (dy);
			if ((in<0)|| (in>dst_Height-1)) in=0;
			im = i + (int ) (dy);
			if ((im>dst_Height-1) || (im<0)) im=dst_Height-1;
			jn = j - (int ) (dx);
			if ((jn<0) || (jn>dst_Width-1)) jn=0;
			jm = j + (int ) (dx);
			if ((jm>dst_Width-1) || (jm<0)) jm=dst_Width-1;

			IyvalInJn = (Gy_Data+in*Gystep)[jn];  //cvmGet(Gy,in,jn); //Gy[in][jn];
			IxvalInJn = (Gx_Data+in*Gxstep)[jn];  //cvmGet(Gx,in,jn); //Gx[in][jn];
			MulTempN = sqrt(IyvalInJn*IyvalInJn + IxvalInJn*IxvalInJn);

			IyvalImJm = (Gy_Data+im*Gystep)[jm];   //cvmGet(Gy,im,jm); //Gy[in][jn];
			IxvalImJm = (Gx_Data+im*Gxstep)[jm];   //cvmGet(Gx,im,jm); //Gx[in][jn];
			MulTempM = sqrt(IyvalImJm*IyvalImJm + IxvalImJm*IxvalImJm);

			if ((MulTemp>MulTempN) && (MulTemp>=MulTempM))
			{
				B_Data[i*dst_Width+j]=MulTemp;
			}
			else
			{
				B_Data[i*dst_Width+j]=0;
			}	
		}
	}//fin for
}
#endif


#endif//_GPUCV_COMPILE_CUDA
