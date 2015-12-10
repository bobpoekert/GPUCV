#include "StdAfx.h"

#include <includecv.h>
#include <highgui.h>
#include <GPUCV/misc.h>
#include <GPUCV/cv_new.h>
#include <GPUCV/cxtypesg.h>
#define FILTRE_DERICHE_CIMG 1			//!< USE THE FILTRE DERICHE FROM THE LIBRARY CIMG

#if FILTRE_DERICHE_CIMG
#include "CImg.h"
using namespace cimg_library;
#endif
 

//Calculation of the edge using the Deriche Filter
typedef float DERICHE_TYPE;
void cvDeriche(CvArr *src, CvArr *dst, float alpha)
{
	DERICHE_TYPE* src_Data	= GCV::GetCVData<DERICHE_TYPE>(src);
	unsigned int src_Width	= GCV::GetWidth(src);
	unsigned int src_Height = GCV::GetHeight(src);
	unsigned int src_Depth = GCV::GetGLDepth(src);

	if(src_Depth!=GL_FLOAT)
	{
		//GPUCV_WARNING("CVDeriche() do support float only");
		return;
	}

#if FILTRE_DERICHE_CIMG
	// FILTRE DERICHE IMMPLEMENTED USING CIMG
	CImg<float> image(src_Width,src_Height);
	CImg<float> gx, gy, gradnorm, lx, ly; 

	CvScalar ValNeg,ValForeground;
	ValNeg.val[0] = 0.0;
	ValForeground.val[0] = /*intThresholdFGEdge*/100+1;

	//int stepiTemp = ((IplImage*)src)->widthStep/sizeof(DERICHE_TYPE);

	float * img_data = image.data;
	int index=0;
	int maxVal = src_Width*src_Height;
	for(int index=0;index<maxVal;index++)
	{
		img_data[index] /*image(j,i)*/= src_Data[index]*255;
		index++;
		img_data[index] /*image(j,i)*/= src_Data[index]*255;
		index++;
		img_data[index] /*image(j,i)*/= src_Data[index]*255;
		index++;
		img_data[index] /*image(j,i)*/= src_Data[index]*255;
	}
	//image.display("Prueba");

	//float alpha = 1.7;
#if 1
	ly = image.get_deriche(alpha,0,'y');
	gx = ly.get_deriche(alpha,1,'x'); 

	lx = image.get_deriche(alpha,0,'x');
	gy = lx.get_deriche(alpha,1,'y'); 
#else
	ly = image.get_deriche(alpha,0,'y');
	gy = ly.get_deriche(alpha,1,'y'); 

	lx = image.get_deriche(alpha,0,'x');
	gx = lx.get_deriche(alpha,1,'x'); 
#endif	

	gradnorm = (gx.pow(2) + gy.pow(2)).sqrt(); 

	//create Temp Images to store derivation results
	IplImage *Yx, *Yy;
	Yx = cvCreateImage(cvSize(src_Width,src_Height), IPL_DEPTH_32F,1); 
	Yy = cvCreateImage(cvSize(src_Width,src_Height), IPL_DEPTH_32F,1); 

	//Switch pointers to datas
	char* imgX_back_data = (char*)Yx->imageData;
	char* imgY_back_data = (char*)Yy->imageData;
	Yx->imageData = (char*)gx.data ;
	Yy->imageData = (char*)gy.data ;
	//=========================


	cvDericheExtrema(dst, Yx, Yy);

	//		hysteresis(dst, 50, 100); //15, 45 used    //TEST: 20,40   50,100   100, 200
	//cvCanny(dst, dst, 150, 200);
	//gy.display("Prueba");
	//cvShowImage("test", I_x);

#if 0//DEBUG
	lx.display("Cimg_LX");
	gx.display("Cimg_GX");
	ly.display("Cimg_LY");
	gy.display("Cimg_GX");
	cvNamedWindow("DericheX", 1);
	cvNamedWindow("DericheY", 1);
	cvShowImage("DericheX", Yx);
	cvShowImage("DericheY", Yy);
	cvWaitKey(0);
#endif

	//restore previous pointers
	Yx->imageData = (char*)imgX_back_data;
	Yy->imageData = (char*)imgY_back_data;
	//===========================

	cvReleaseImage(&Yx);
	cvReleaseImage(&Yy);

#else//david implementation
	IplImage *I_x, *I_y, *aux1, *aux2;

	//float **Yx, **Yy;
	CvMat *Yx, *Yy;

	int i, j, nrows, ncolumns;
	CvScalar valorResult;
	nrows = GCV::GetHeight(src);
	ncolumns = GCV::GetWidth(src);

	int stepiContour = ((IplImage*)dst)->widthStep/sizeof(uchar);
	uchar* dataiContour = (uchar *)((IplImage*)dst)->imageData;
	int indexY;

	aux1 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	aux1->origin = IPL_ORIGIN_BL;

	aux2 = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
	aux2->origin = IPL_ORIGIN_BL;

	Yx = cvCreateMat(nrows, ncolumns, CV_32FC1); 
	Yy = cvCreateMat(nrows, ncolumns, CV_32FC1); 

	int Yxstep  = Yx->step/sizeof(float);
	float *Yxdata = Yx->data.fl;
	int Yystep  = Yy->step/sizeof(float);
	float *Yydata = Yy->data.fl;

	//Filtrage 1
	cvDericheLissageY(src, aux1, alpha);
	//Lissage 1
	cvDericheDeriveX(aux1, alpha, Yx);
	//Filtrage 2
	cvDericheLissageX(src, aux2, alpha);
	//Lissage 2
	cvDericheDeriveY(aux2, alpha, Yy);

	float max = 0;
	float min = 0;
	float grado = 0;
	float gradX, gradY;
	for (i=0;i<nrows;i++)
	{
		indexY = i*stepiContour;
		for (j=0;j<ncolumns;j++)
		{
			gradY = (Yydata+i*Yystep)[j];
			gradX = (Yxdata+i*Yxstep)[j];
			dataiContour[indexY+j] = sqrt(gradY*gradY + gradX*gradX);
		}
	}//fin for 
	cvDericheExtrema(dst, Yx, Yy);
	cvReleaseImage(&aux1); 
	cvReleaseImage(&aux2); 
	cvReleaseMat(&Yx);
	cvReleaseMat(&Yy);
#endif
}

void cvDericheExtrema(CvArr* B, CvArr* Gx, CvArr* Gy)
{

	DERICHE_TYPE* B_Data	= GCV::GetCVData<DERICHE_TYPE>(B);
	unsigned int dst_Width	= GCV::GetWidth(B);
	unsigned int dst_Height = GCV::GetHeight(B);

	DERICHE_TYPE* Gx_Data	= GCV::GetCVData<DERICHE_TYPE>(Gx);
	unsigned int Gx_Width	= GCV::GetWidth(Gx);
	unsigned int Gx_Height = GCV::GetHeight(Gx);

	DERICHE_TYPE* Gy_Data	= GCV::GetCVData<DERICHE_TYPE>(Gy);
	unsigned int Gy_Width	= GCV::GetWidth(Gy);
	unsigned int Gy_Height = GCV::GetHeight(Gy);

	cvNamedWindow("DericheX");
	cvNamedWindow("DericheY");
	cvShowImage("DericheX", Gx);
	cvShowImage("DericheY", Gy);

	unsigned int i,j;
	unsigned int in,jn,im,jm;
//	double k;
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



#if !FILTRE_DERICHE_CIMG
//Calcul de la derivate dans l'image pour le filtre de Deriche
//Calculation of the derivate in the image for Deriche's filter
void cvDericheDeriveY(CvArr* Ima, float alpha, CvMat* Y)
{
	IplImage* YImage;
	IplImage* YT;
	CvMat* YTemp; //float** YTemp;
	int nrows, ncolumns;
	nrows = GCV::GetWidth(Ima);
	ncolumns = GCV::GetHeight(Ima);

	YTemp = cvCreateMat(nrows, ncolumns, CV_32FC1);
	YImage = cvCreateImage(cvGetSize(Ima), IPL_DEPTH_8U, 1);
	YImage->origin = IPL_ORIGIN_BL;
	IplImage *ImaT = cvCreateImage(cvGetSize(Ima), IPL_DEPTH_8U, 1);
	ImaT->origin = IPL_ORIGIN_BL;
	cvTranspose(Ima, ImaT);
	cvDericheDeriveX(ImaT,alpha, YTemp);

	float value;

	for(int i=0; i<nrows; i++)
		for(int j=0;j<ncolumns;j++)
		{
			value = cvmGet(YTemp,i,j);//Y[j][i]=YTemp[i][j]; 
			cvmSet(Y,j,i,value);
		}

		cvReleaseMat(&YTemp);
}

//Calcul de la derivate dans l'image pour le filtre de Deriche
//Calculation of the derivate in the image for Deriche's filter
void cvDericheDeriveX(CvArr* Ima, float alpha, CvMat* Y)
{
	float k = exp(-alpha);
	float a = (1-k)*(1-k);
	float b1 = -2*k;
	float b2 = k*k;


	int mgx, mgy;
	mgx = mgy = 2;
	int width = GCV::GetWidth(Ima);
	int height = GCV::GetHeight(Ima);

	int Sx = width - 2*mgx;
	int Sy = height - 2*mgy;

	CvMat *Yp, *Ym;//float Yp[200][200]; float Yp[200][200]; 
	//float Ym[200][200]; 

	CvScalar aval; 
	float b1val, b2val;

	float value, MulTemp, valueYm, valueYp;
	value = 0.0;

	Yp = cvCreateMat(height, width, CV_32FC1); 
	Ym = cvCreateMat(height,width, CV_32FC1); 

	for(int i=0; i<height; i++)
	{
		for(int j=0;j<width;j++)
		{
			cvmSet(Yp,i,j,value); //Yp[i][j]=0; 
			cvmSet(Ym,i,j,value); //Ym[i][j]=0;
			cvmSet(Y,i,j,value);
		}
	}

	float max = 0;
	float min = 0;

	for (int j = mgx; j < Sx+mgx ; j++)
	{
		// Filtrage Yp
		for(int i = mgy; i < Sy+mgy; i++)
		{
			aval = cvGet2D(Ima,i-1,j);
			b1val = cvmGet(Yp,i-1,j); //Yp[i-1][j];
			b2val = cvmGet(Yp,i-2,j);  //Yp[i-2][j];
			MulTemp = a*aval.val[0] - b1*b1val - b2*b2val;
			cvmSet(Yp,i,j,MulTemp); //Yp[i][j] = MulTemp;
		}

		// Filtrage Ym
		for(int i = Sy+mgy-1; i >= mgy; i--)
		{
			aval = cvGet2D(Ima,i+1,j);
			b1val = cvmGet(Ym,i+1,j);  //Ym[i+1][j];
			b2val = cvmGet(Ym,i+2,j);  //Ym[i+2][j];
			MulTemp = -a*aval.val[0] - b1*b1val - b2*b2val;
			cvmSet(Ym,i,j,MulTemp); //Ym[i][j] = MulTemp;		   
		}

		// Resultat Y
		for(int i = mgy; i < Sy+mgy; i++)
		{
			valueYm = cvmGet(Ym,i,j);
			valueYp = cvmGet(Yp,i,j);
			value = valueYp + valueYm; //Y[i][j] = Yp[i][j] + Ym[i][j];	
			cvmSet(Y,i,j,value);
		}
	}

	cvReleaseMat(&Yp);
	cvReleaseMat(&Ym);
}

/// Calcul du lissage pour le filtre de Deriche
void cvDericheLissageX(CvArr* Ima, CvArr *YImage, float alpha)
{
	float k = (1-exp(-alpha))*(1-exp(-alpha))/(1+2*alpha*exp(-alpha)-exp(-2*alpha));
	float a0 = k;
	float a1 = k*(alpha-1)*exp(-alpha);
	float a2 = k*(alpha+1)*exp(-alpha);
	float a3 = -k*exp(-2*alpha);
	float b1 = -2*exp(-alpha);
	float b2 = exp(-2*alpha);

	int mgx, mgy;
	mgx = mgy = 2;

	int width_Ima = GCV::GetWidth(Ima);
	int height_Ima = GCV::GetHeight(Ima);
	int width_Y = GCV::GetWidth(YImage);
	int height_Y = GCV::GetHeight(YImage);

	int Sx = width_Ima - 2*mgx;
	int Sy = height_Ima - 2*mgy;

	int stepYImage = ((IplImage*)YImage)->widthStep/sizeof(uchar);
	uchar* dataYImage = (uchar *)((IplImage*)YImage)->imageData;
	int indexYImage;

	CvMat *Yp, *Ym; 

	CvScalar a0val, a1val, a2val, a3val, valorResult; 
	float MulTemp, b1val, b2val, valueYm, valueYp;

	Yp = cvCreateMat(height_Ima, width_Ima, CV_32FC1); 
	Ym = cvCreateMat(height_Ima, width_Ima, CV_32FC1);

	float value = 0.0;

	for(int i=0; i<height_Ima; i++)
		for(int j=0;j<width_Ima;j++)
		{
			cvmSet(Yp,i,j,value); //Yp[i][j]=0; 
			cvmSet(Ym,i,j,value); //Ym[i][j]=0;
		}

		float max,min;
		max = min = 0;


		for (int i = mgy; i < Sy+mgy ; i++)
		{
			indexYImage = i*stepYImage;
			for(int j = mgx; j < Sx+mgx; j++)
			{
				a0val = cvGet2D(Ima,i,j);
				a1val = cvGet2D(Ima,i,j-1);
				b1val = cvmGet(Yp,i,j-1); //Yp[i-1][j];
				b2val = cvmGet(Yp,i,j-2);  //Yp[i-2][j];
				MulTemp = a0*a0val.val[0] + a1*a1val.val[0] - b1*b1val - b2*b2val;
				cvmSet(Yp,i,j,MulTemp); //Yp[i][j] = MulTemp;
			}

			for(int j = Sx+mgx-1; j >= mgx; j--)
			{
				a2val = cvGet2D(Ima,i,j+1);
				a3val = cvGet2D(Ima,i,j+2);
				b1val = cvmGet(Ym,i,j+1);  //Ym[i+1][j];
				b2val = cvmGet(Ym,i,j+2);  //Ym[i+2][j];
				MulTemp = a2*a2val.val[0] + a3*a3val.val[0] - b1*b1val - b2*b2val;
				cvmSet(Ym,i,j,MulTemp); //Ym[i][j] = MulTemp;	
			}

			for(int j = mgx; j < Sx+mgx; j++)
			{
				valueYm = cvmGet(Ym,i,j);
				valueYp = cvmGet(Yp,i,j);
				dataYImage[indexYImage+j] = valueYp + valueYm; 
			}//fin for

		}

		cvReleaseMat(&Yp);
		cvReleaseMat(&Ym);
}

// Calcul du lissage pour le filtre de Deriche
void cvDericheLissageY(CvArr* Ima, CvArr* Y, float alpha)
{
	IplImage *ImaT = cvCreateImage(cvGetSize(Ima), IPL_DEPTH_8U, 1);
	ImaT->origin = IPL_ORIGIN_BL;
	IplImage *YT = cvCreateImage(cvGetSize(Ima), IPL_DEPTH_8U, 1);
	YT->origin = IPL_ORIGIN_BL;
	cvDericheTranspose(Ima, ImaT); 
	cvDericheLissageX(ImaT,YT,alpha);
	cvDericheTranspose(YT,Y);
	cvReleaseImage(&YT);
	cvReleaseImage(&ImaT);
}
#endif//!FILTRE_DERICHE_CIMG

#if _GPUCV_DEVELOP_BETA
void cvLocalSum(CvArr* src1,CvArr* dst, int box_height , int box_width)
{		

	int height = GCV::GetHeight(src1);
	int width = GCV::GetWidth(src1);


	for( int _iy = 0; _iy < height; _iy++ )
	{
		int _ix, _xstep = 1;
		for( _ix = 0; _ix < width; _ix += _xstep )

		{
			int lu = (int)(src1->imageData[_iy*width+_ix]);
			int ll = (int)(src1->imageData[(_iy+box_height)*width+_ix]);
			int ru = (int)(src1->imageData[_iy*width+_ix+box_width]);
			int rl = (int)(src1->imageData[(_iy+box_height)*width+_ix+box_width]);
			dst->imageData[_iy*width+_ix] = (lu - ll - ru + rl)/width*height ;
		}
	}
}
#endif

