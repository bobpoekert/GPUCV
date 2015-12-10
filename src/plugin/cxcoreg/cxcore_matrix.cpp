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
//#include <cvg/cvg.h>
#include <cxcoreg/cxcoreg.h>

using namespace GCV;
/////////////////////////////////////////////////////////
//---------------Function Declarations--------------////
///////////////////////////////////////////////////////

#if _GPUCV_DEVELOP_BETA
IplImage ** splitInto4(IplImage *A);
IplImage ** splitInto2(IplImage *A);
//void		customMultiply(IplImage *A,IplImage *B,double alpha, IplImage *C);
IplImage * customAdd(IplImage *A,IplImage *B);
IplImage * customSubtract(IplImage *A,IplImage *B);
IplImage * combineMatrices(IplImage *A,IplImage *B);
IplImage * matrixMultiply(IplImage *A,IplImage *B);
IplImage * convertToMatrixFormat(IplImage *A);
IplImage * multiMatrixAvg(IplImage **images,int num);
IplImage * customMatrixAvg (IplImage *A,IplImage* B,IplImage *C,IplImage* D,IplImage *E,IplImage* F,IplImage *G,IplImage* H);

/////////////////////////////////////////////////////////
//-------------Definitons of Constants--------------////
///////////////////////////////////////////////////////
#define PIXEL_CHANNELS 4 // No of components in each pixel

/////////////////////////////////////////////////////////
//-----------------------End------------------------////
///////////////////////////////////////////////////////
#endif



#if _GPUCV_SUPPORT_CVMAT
void  cvgGEMM(CvArr* src1,
			  CvArr* src2, 
			  double alpha,
			  CvArr* src3, 
			  double beta, 
			  CvArr* dst, 
			  int tABC/*=0*/ )
{
	GPUCV_START_OP(cvGEMM(src1, src2, alpha, src3, beta, dst, tABC),
		"cvgGEMM",
		dst,
		GenericGPU::HRD_PRF_2);

	SG_Assert(tABC==0, "Matrix transposition not done yet"); 

	DataContainer * Tex1 = GPUCV_GET_TEX(src1);
	DataContainer * Tex2 = GPUCV_GET_TEX(src2);
	DataContainer * Tex3 = NULL;
	DataContainer * TexDest = GPUCV_GET_TEX(dst);

	if(src3!=NULL && beta != 0)
	{//perform matrix multiplication + addition
		Tex3 = GPUCV_GET_TEX(src3);
	}
	DataDsc_CvMat * Mat1 = Tex1->GetDataDsc<DataDsc_CvMat>();
	DataDsc_CvMat * Mat2 = Tex2->GetDataDsc<DataDsc_CvMat>();
	DataDsc_CvMat * Mat3 = (Tex3)?Tex3->GetDataDsc<DataDsc_CvMat>():NULL;
	DataDsc_CvMat * MatDst = TexDest->GetDataDsc<DataDsc_CvMat>();

	// Parameters passed to the shader for computation
	float Params[7]=
	{	
		Mat1->_GetWidth() * Mat1->_GetNChannels(),// Params[0] is number of columns in A
		Mat2->_GetHeight(),// Params[1] is number of rows in A
		//Tex1->_GetNChannels(),// Params[2] is number of channels in the images
		Mat1->_GetHeight(),// Params[3] is number of rows in A
		Mat2->_GetWidth(), // Params[4] is number of columns in A
		(float)alpha,
		(float)beta
	};

	if(Mat1->_GetNChannels() == 1)//don't know why yet, but we must divide restults by 9
	{
		Params[4] = (float)(alpha/9.0);
		Params[5] = (float)(beta/9.0);
	}

	SG_Assert((GLsizei)Mat1->_GetNChannels()*Mat1->_GetWidth() == Mat2->_GetHeight(), "Matrices incompatible for multiplication.");

	std::string MetaOption;
	for(int i = Mat1->_GetNChannels()+1; i<=4;  i++)
	{
		MetaOption+= "$DEF_CMPNT_";
		MetaOption+= '0' + i;
		MetaOption+= "=0//";
	}//deactivate component one by one...

	MetaOption += (Mat3)?"$DEF_ADD=1//":"";


	TemplateOperator("cvgGEMM", "FShaders/mat/matmul_add", "",
		src1, src2, src3,
		dst, Params, (Tex3)?7:6,
		TextureGrp::TEXTGRP_SAME_ALL_FORMAT, MetaOption);

	GPUCV_STOP_OP(
		cvGEMM(src1, src2, alpha, src3, beta, dst, tABC),
		src1, src2, src3, dst);
}



#endif



#if _GPUCV_DEVELOP_BETA
/** 
* \brief Calculates Average of matrices
*
* This function calculates the average of a number of matrices.
* The function shaders to calculate average per channel. 
* However the limitation of shaders is that one can only load 8 images at a time. 
* Thus, in this function one can input only upto 8 images or less. For less than 8 images, set the remaining
* images to NULL.If one needs to use more than 8 images in the training set, the function (multiMatrixAvg) is useful.
*
* \param A,B,C,D,E,F,G,H -> Input Matrices(May be set to null in case one uses less that 8 Matrices) 
* \return The result is written to an IplImage and is returned
* \author Kshitiz Singh
*/
IplImage * customMatrixAvg(IplImage *A,IplImage* B,IplImage *C,IplImage* D,IplImage *E,IplImage* F,IplImage *G, IplImage* H)
{
	int checker=0;
	// check to see whether input images satisfy shader limits or not
	IplImage ** tab ;// array of IplImages for storing the input
	int nb = 7;

	tab = new IplImage*[7];
	tab[0] = (IplImage*)B;
	tab[1] = (IplImage*)C;
	tab[2] = (IplImage*)D;
	tab[3] = (IplImage*)E;
	tab[4] = (IplImage*)F;
	tab[5] = (IplImage*)G;
	tab[6] = (IplImage*)H;

	if((A->width)>_GPUCV_TEXTURE_MAX_SIZE_X|| (A->height)>_GPUCV_TEXTURE_MAX_SIZE_Y)
	{ 
		checker=1;

	}
	if(checker==0)
	{
		int ck;
		for(ck=0;ck<7;ck++)
		{
			if((int)(tab[ck]->width)>(_GPUCV_TEXTURE_MAX_SIZE_X/PIXEL_CHANNELS) || (int)(tab[ck]->height)>_GPUCV_TEXTURE_MAX_SIZE_Y)
			{

				checker=2;
			}
		}
	}
	if(checker==1)
	{
		printf("\n Image(s) violate shader limits \n");
		exit(0);
	}
	else if(checker==2)
	{
		printf("\n Image(s) violate shader limits \n");
		exit(0);
	}

	else 
	{				

		//GPUCV check for Hardware Compatibility
		if (!GetHardProfile()->IsCompatible(GenericGPU::HRD_PRF_2))
		{
			//cvAdd(A, B, C, mask);
			printf("Not Compatible");
			return NULL; 
		}
		//The result is stored in Output and returned
		IplImage *Output = NULL;
		SetThread();
		// Correcting : if dst = one of the sources, creating temp image
		// This temp image will be replacing C. By the end, C will be replaced.
		// Warning : performances issues. If possible, do not use A or B for dst.
		// Using ForceTexture here could lead to hazardous results...


		string chosen_filter;
		chosen_filter = GetFilterManager()->GetFilterName("FShaders/matrix/matrix/avgmat.frag");// We choose the desired shader that computes the average
		GetFilterManager()->Apply(chosen_filter,A,Output,A->width,A->height,tab,nb);
		// We apply the chosen shader, with the first image being A and the rest being stored in tab.
		//Size of the output is same as A, so we set the parameters similarly


		delete [] tab;// We are done with the images, so release them by deleting the 
		return Output;
		UnsetThread();
	}

	return NULL;
}

/** 
*\brief Calculating average of more than 8 matrices at a time
*
* This function overcomes the limitation of loading only 8 images at a time
* One can input more than 8 images and the result is written to the output.
* \param images-> An array of IplImages that contains the input images
* \param num-> The number of images in the array
* \todo => Mipmapping
* \author Kshitiz Singh
*/
IplImage * multiMatrixAvg(IplImage **images,int num)
{
	int checker=1;
	if(num<1){checker=-1;printf("\n Number of images cannot be less than 1\n");}
	int ck;
	for(ck=0;ck<num;ck++)
	{
		if((int)(images[ck]->width)>(_GPUCV_TEXTURE_MAX_SIZE_X/PIXEL_CHANNELS) || (int)(images[ck]->height)>_GPUCV_TEXTURE_MAX_SIZE_Y)
		{
			checker=0;
			//return images[ck];
		}
	}
	// check to see whether input images satisfy shader limits or not
	if(checker==-1){return images[0];/* Default return*/}
	else if(checker==0)
	{
		printf("\nInput Image dimension(s) exceeds shader limit\n");
		return images[0];//Default return

	}

	else 
	{				
		// GPUCV Check
		if (!GetHardProfile()->IsCompatible(GenericGPU::HRD_PRF_2))
		{
			//cvAdd(A, B, C, mask);
			printf("Not Compatible");
			return NULL; 
		}

		IplImage *temp=images[num-1];// we use this image for all our intermediate result storage, initialize it to the last image in the array
		float Param[1]={1.};//	

		/*		
		DataContainer * TexTest = GetTextureManager()->cvgGetGpuImage(A);// LoadImageIntoFrameBuffer(A,0);
		TexTest->WriteToFrameBuffer();
		TexTest->ReadFromFrameBuffer(0,0);
		//GetTextureManager()->cvgGetCpuImage(A);

		cvCopyImage(A, C);
		return;
		if(GetGpuCVSettings()->GetGlutDebug())
		glutSwapBuffers();
		*/
		SetThread();
		string chosen_filter;
		chosen_filter = GetFilterManager()->GetFilterName("FShaders/matrix/multimatrixavg.frag");
		GetFilterManager()->AddParam(chosen_filter, Param[0]);
		// Correcting : if dst = one of the sources, creating temp image
		// This temp image will be replacing C. By the end, C will be replaced.
		// Warning : performances issues. If possible, do not use A or B for dst.
		// Using ForceTexture here could lead to hazardous results...

		while(num >7)
		{
			IplImage ** tab ;
			int nb = 7;
			IplImage *A=(IplImage*)images[num-1];
			tab = new IplImage*[nb];
			tab[0] = (IplImage*)images[num-2];
			tab[1] = (IplImage*)images[num-3];
			tab[2] = (IplImage*)images[num-4];
			tab[3] = (IplImage*)images[num-5];
			tab[4] = (IplImage*)images[num-6];
			tab[5] = (IplImage*)images[num-7];
			tab[6] = (IplImage*)images[num-8];



			GetFilterManager()->Apply(chosen_filter,A,temp,A->width,A->height,tab,nb);
			//GetTextureManager()->set_last(C, LAST_IS_GPU);
			delete [] tab;
			Param[0]+=8.;
			GetFilterManager()->SetParamI(chosen_filter,1,Param[0]);

		}

		while(num >0 && num<7)
		{
			IplImage ** tab ;
			int nb = 1;
			IplImage *A=(IplImage*)images[num-1];
			tab = new IplImage*[nb];
			tab[0] = (IplImage*)images[num-2];
			GetFilterManager()->Apply(chosen_filter,A,temp,A->width,A->height,tab,nb);
			//GetTextureManager()->set_last(C, LAST_IS_GPU);
			delete [] tab;
			Param[0]+=1.;
			GetFilterManager()->SetParamI(chosen_filter,1,Param[0]);

		}

		return temp;
	}
}
/** 
* \brief Multiply Two matrices
*
* We input a 4096x4096 matrix as A
* and the other 4096x3 as B. the output is given on C.
* We first split A into 4 textures A1,A2,A3,A4, each with dimensions 2048x512(*4 because of packing 4 channels in a pixel).
* We also assume that packing in pixels is bgra in a 1x4 vector.We also split B into 2 textures B1 and B2, with dimensions 2048x1(*4 with the alpha channel stuffed with 1 because it will not be used).
* Then we multiply A1 and B1, A2 and B2,Add this and this becomes the upper half of our result. The Lower
* half is got by multiplying A3 and B1 and A4 and B2, and then adding them.
* We use the custom multiply function for multiplication, cvgAdd for adding and simple assingment for assigning top and bottom halves.
* Thus, this function is highly customized for the PCA calculation, any other large matrix multipliction would require a separate, but similar function.  
* \param A -> First Matrix to be multiplied
* \param B -> Second Matrix to be multiplied
* \return The result of the multiplication written to an IplImage
* \author Kshitiz Singh
*/
IplImage * matrixMultiply(IplImage *A,IplImage *B)
{

	IplImage **A1=new IplImage*[4];
	//Split The first Matrix, ie the one having the eigenfaces into 4 parts so as to satisfy the shader limits
	A1=splitInto4(A);
	IplImage **B1=new IplImage*[2];
	//Split The second Matrix, ie the normalised image into 2 parts to satisfy shader limits
	B1=splitInto2(B);
	//Multiply various fragments using the custom multiply function
	IplImage *R1=NULL;IplImage *R2=NULL;IplImage *R3=NULL;IplImage *R4=NULL;
	customMultiply(A1[0],B1[0],1., R1);
	customMultiply(A1[1],B1[1],1., R2);
	customMultiply(A1[2],B1[0],1., R3);
	customMultiply(A1[3],B1[1],1., R4);
	IplImage *H1,*H2,*Result;
	//Combine fragments to generate a bigger matrix
	H1=customAdd(R1,R2);
	H2=customAdd(R3,R4);
	Result=combineMatrices(H1,H2);
	return Result; 

}

/** 
* \brief Multiply Two matrices Using shaders
* This function uses a custom matrix multiplication algorithm, designed keeping in mind the bgra packing of IplImages.
* This uses a shader, that proceses a row from the first matrix and 4 columns from the second matrix per pixel in the output.
* Originally intended to multiply a 2048x2048 matrix stored as 2048x512 texture and a 2048x4 matrix stored as 2048x1 texture
* \param A-> First matrix to be multiplied
* \param B-> Second matrix to be multiplied
* \param C-> Result of AxB is put in C
* \todo => Genralise the multiplication, although it is already pretty much generalised as compared to other functions.
* \author Kshitiz Singh
*/
void customMultiply(CvArr *A,CvArr *B, double alpha, CvArr *C)
{
	DataContainer * TexA = GPUCV_GET_TEX(A);
	DataContainer * TexB = GPUCV_GET_TEX(B);
	DataContainer * TexC = GPUCV_GET_TEX(C);


	// Checking if any of the input matrices violate the shader limits
	//image size is already checked into the texture...

	SG_Assert(TexA->_GetNChannels() == TexB->_GetNChannels(), "Images don't have compatible channels \n");
	SG_Assert(TexA->_GetNChannels() == 4 && TexB->_GetNChannels()==4, 
		"One or more CvMat don't have 4 channels. Currently only 4 channel images supported \n");


	SG_Assert(TexA->_GetNChannels()*TexA->_GetWidth() == TexB->_GetHeight(), "Matrices incompatible for multiplication.");

	//GPUCV check
	if (!GetHardProfile()->IsCompatible(GenericGPU::HRD_PRF_2))
	{
		printf("Not Compatible");
	}

	//IplImage *Output;
	SetThread();

	// for choosing the particular shader
	string chosen_filter;
	chosen_filter = GetFilterManager()->GetFilterName("FShaders/matrix/custom_multiply.frag");

	// Parameters passed to the shader for computation

	float Params[6]=
	{	
		TexA->_GetWidth() * TexA->_GetNChannels(),// Params[0] is number of columns in A
		TexB->_GetHeight(),// Params[1] is number of rows in A
		TexA->_GetNChannels(),// Params[2] is number of channels in the images
		TexA->_GetHeight(),// Params[3] is number of rows in A
		TexB->_GetWidth(), // Params[4] is number of columns in A
		alpha
	};

	// Actual code for passing the parameters to the shader
	if(GetFilterManager()->GetNbParams(chosen_filter) == 0) 
	{
		GetFilterManager()->AddParam(chosen_filter, Params[0]);
		GetFilterManager()->AddParam(chosen_filter, Params[1]);
		GetFilterManager()->AddParam(chosen_filter, Params[2]);
		GetFilterManager()->AddParam(chosen_filter, Params[3]);
		GetFilterManager()->AddParam(chosen_filter, Params[4]);
		GetFilterManager()->AddParam(chosen_filter, Params[5]);
	}
	else
	{
		GetFilterManager()->SetParamI(chosen_filter, 0, Params[0]);
		GetFilterManager()->SetParamI(chosen_filter, 1, Params[1]);
		GetFilterManager()->SetParamI(chosen_filter, 2, Params[2]);
		GetFilterManager()->SetParamI(chosen_filter, 3, Params[3]);
		GetFilterManager()->SetParamI(chosen_filter, 4, Params[4]);
		GetFilterManager()->SetParamI(chosen_filter, 5, Params[5]);
	}

	//This applies the shader and outputs the result 
	GetFilterManager()->Apply(chosen_filter/*apply the current shader*/,
		TexA/*First image for the operation*/,
		TexC/*Output Image*/,
		int(Params[4])/*Width of output image*/,
		int(Params[3])/*Height of Output image*/,
		&TexB/*Second operand*/,
		1/*No of operands other than A*/);

	UnsetThread();
}
/** 
* \brief Compute sum of two matrices	
* 
* This function adds two IplImages,A and B, per channel, and stores the result in Output
* \param A->first input matrix
* \param b->second input matrix
* \return Matrix in IplImage format containing the result of A+B, per channel
* \author Kshitiz Singh
*/
IplImage * customAdd(IplImage *A,IplImage* B)
{
	int checker=0;
	// Checking if any of the input matrices violate the shader limits
	if((int)(A->width)>_GPUCV_TEXTURE_MAX_SIZE_X || (int)(A->height)>_GPUCV_TEXTURE_MAX_SIZE_Y || (int)(B->width)>_GPUCV_TEXTURE_MAX_SIZE_X ||(int)(B->height)>_GPUCV_TEXTURE_MAX_SIZE_Y)
	{ 
		checker=1;
	}
	if((A->nChannels != B->nChannels) || (A->nChannels > 4) || (A->nChannels < 0)||(B->nChannels > 4) || (B->nChannels < 0))
	{ 
		checker=2;
	}


	if(checker==1)
	{
		printf("\nInput matrix dimension(s) violate shader limits\n");
		exit(0);
	}

	else if(checker==2)
	{
		printf("\nImage channels either dont match or are not valid \n");
		exit(0);
	}

	else
	{
		if (!GetHardProfile()->IsCompatible(GenericGPU::HRD_PRF_2))
		{
			printf("Not Compatible");
			return NULL; 
		}


		SetThread();
		IplImage *Output = NULL;
		// Correcting : if dst = one of the sources, creating temp image
		// This temp image will be replacing C. By the end, C will be replaced.
		// Warning : performances issues. If possible, do not use A or B for dst.
		// Using ForceTexture here could lead to hazardous results...


		string chosen_filter;
		chosen_filter = GetFilterManager()->GetFilterName("FShaders/matrix/custom_add.frag");
		GetFilterManager()->Apply(chosen_filter,A,Output,A->width,A->height,&B,1);

		return Output;
		UnsetThread();
	}

	return NULL;
}
/** 
* \brief Divide a given texture into parts so as to satisfy shader limits
*
* This function takes a big matrix(texture/image) A and divides it into 4 parts, which are saved to textures B,C,D,E.
* Pictorially it looks like
* \code
*   -----------               ----------------------
*  |           |             |           |          |
*  |           |             |     B     |     C    |
*  |     A     |    ---->    |           |          | 
*  |           |              ----------------------   
*  |           |             |           |          |
*   -----------              |     D     |     E    |
*                            |           |          |
*                             ----------------------
* \endcode
* \param A-> Input texture, ie the one that needs to be split
* \return An array of IplImages containing the texture fragments obtained from input texture 
* \todo =>  Generalise the algo for usage of dividing into n fragments, currently dividing into 4 parts only
*
* \author Kshitiz Singh
*
*/

IplImage ** splitInto4(IplImage *A)
{

	int checker=0;
	// Checking if any of the input matrices violate the shader limits
	if((int)(A->width)>_GPUCV_TEXTURE_MAX_SIZE_X || (int)(A->height)>_GPUCV_TEXTURE_MAX_SIZE_Y )
	{ 
		checker=1;
	}
	if((int)(A->width)%4 /* take modulo n in case the function is extended to split matrix into n parts*/ !=0 || (int)(A->height)%4 !=0/* take modulo n in case the function is extended to split matrix into n parts*/) 
	{
		printf("Dimensions of input matrix are not divisble by 4, thus dividing it into 4 parts may result in \n");
		printf(" unwanted results. Advisable to check matrix A.");
	}

	if(checker==1)
	{
		printf("\nInput matrix dimension(s) violate shader limits\n");
		exit(0);// Default return
	}


	else
	{

		IplImage **result=new IplImage*[4];// The array that will contain the output
		IplImage *B=NULL;IplImage *C=NULL;IplImage *D=NULL;IplImage *E=NULL;// The images that will contain fragments of the input matrix
		int i,j;
		// First Quarter
		for(i=0;i<_GPUCV_TEXTURE_MAX_SIZE_X;i++)
		{
			for(j=0;j<_GPUCV_TEXTURE_MAX_SIZE_Y;j++)
			{	
				B->imageData[i*_GPUCV_TEXTURE_MAX_SIZE_X+j]=A->imageData[i*_GPUCV_TEXTURE_MAX_SIZE_X+j];
			}
		}
		// Second Quarter
		for(i=0;i<_GPUCV_TEXTURE_MAX_SIZE_X;i++)
		{
			for(j=_GPUCV_TEXTURE_MAX_SIZE_X;j<A->width*4;j++)
			{	
				C->imageData[i*_GPUCV_TEXTURE_MAX_SIZE_X+(j-_GPUCV_TEXTURE_MAX_SIZE_X)]=A->imageData[i*_GPUCV_TEXTURE_MAX_SIZE_X+j];
			}
		}
		// Third Quarter
		for(i=_GPUCV_TEXTURE_MAX_SIZE_X;i<A->width*4;i++)
		{
			for(j=0;j<_GPUCV_TEXTURE_MAX_SIZE_X;j++)
			{	
				D->imageData[(i-_GPUCV_TEXTURE_MAX_SIZE_X)*_GPUCV_TEXTURE_MAX_SIZE_X+j]=A->imageData[i*_GPUCV_TEXTURE_MAX_SIZE_X+j];
			}
		}
		// Fourth Quarter
		for(i=_GPUCV_TEXTURE_MAX_SIZE_X;i<A->width*4;i++)
		{
			for(j=_GPUCV_TEXTURE_MAX_SIZE_X;j<A->width*4;j++)
			{	
				E->imageData[(i-_GPUCV_TEXTURE_MAX_SIZE_X)*_GPUCV_TEXTURE_MAX_SIZE_X+(j-_GPUCV_TEXTURE_MAX_SIZE_X)]=A->imageData[i*_GPUCV_TEXTURE_MAX_SIZE_X+j];
			}
		}
		result[0]=(IplImage *)B;
		result[1]=(IplImage *)C;
		result[2]=(IplImage *)D;
		result[3]=(IplImage *)E;
		return result;
	}
}

/** 
* \brief Used to split a matrix with a small num of columns column and large number of rows
* 
* \code
*   -----------               -----------
*  |           |             |           |
*  |           |             |     B     |
*  |     A     |  ------>    |           |
*  |           |              -----------   
*  |           |             |           |     
*   ----------               |     D     |     
*                            |           |
*                             -----------
* \endcode
* \todo => A more generalised approach would be to divide it into n such parts
*
* \author Kshitiz Singh____//s
*/
IplImage ** splitInto2(IplImage *A)
{
	int checker=0;
	// Checking if any of the input matrices violate the shader limits
	if((A->width)>_GPUCV_TEXTURE_MAX_SIZE_X || (A->height)>_GPUCV_TEXTURE_MAX_SIZE_Y )
	{ 
		checker=1;
	}
	if((int)(A->height)%2 /* take modulo n in case the function is extended to split matrix into n parts*/) 
	{
		printf("Dimensions of input matrix are not divisble by 4, thus dividing it into 4 parts may result in \n");
		printf(" unwanted results. Advisable to check matrix A.");
	}

	if(checker==1)
	{
		printf("\nInput matrix dimension(s) violate shader limits\n");
		exit(0);// Default return
	}


	else
	{
		IplImage ** result =new IplImage*[2];
		IplImage *B = new IplImage;IplImage *C = new IplImage;
		int i,j;
		// First Half
		for(i=0;i<_GPUCV_TEXTURE_MAX_SIZE_X;i++)
		{
			for(j=0;j<1;j++)
			{	
				B->imageData[i+j]=A->imageData[i+j];
			}
		}
		// Second Half
		for(i=_GPUCV_TEXTURE_MAX_SIZE_X;i<A->width*4;i++)
		{
			for(j=0;j<1;j++)
			{	
				B->imageData[(i-_GPUCV_TEXTURE_MAX_SIZE_X)+j]=A->imageData[i+j];
			}
		}
		result[0]=(IplImage *)B;
		result[1]=(IplImage *)C;
		return result;
	}
}
/** 
* \brief This function is used to combine smaller matrices to give a bigger matrix
* \code
*
*   -----------
*  |           |
*  |     B     |
*  |           | 
*   -----------                    --------------
*                                 |              |
*        +       --------->       |              |   
*                                 |      A       |
*   -----------                   |              |
*  |           |                  |              |
*  |     D     |                   -------------- 
*  |           |
*   -----------
* \endcode
* Using this function, we combine the two input matrices to give a bigger matrix
* \param B-> Input Matrix( Upper Half of the resultant)
* \param C-> Input Matrix (Lower Half of the resultant)
* \return A matrix in the form of an IplImage with the input matrices as its sub-components  
* \todo => A more generalised approach would be to combine n such smaller matrices and create the bigger matrix
*
* \author Kshitiz Singh____//s
*/
IplImage * combineMatrices(IplImage *B,IplImage *C)
{
	int checker=0;
	// Checking if any of the input matrices violate the shader limits
	if(B->width>_GPUCV_TEXTURE_MAX_SIZE_X || B->height>_GPUCV_TEXTURE_MAX_SIZE_Y || C->width>_GPUCV_TEXTURE_MAX_SIZE_X || C->height>_GPUCV_TEXTURE_MAX_SIZE_Y)
	{ 
		checker=1;
	}
	if(B->width != C->width )
	{

		checker =2;
	}

	if(B->nChannels != C->nChannels)
	{
		printf("\n Number of channels don't match \n");
		checker =3;
	}

	if(checker==1)
	{
		printf("\nInput matrix dimension(s) violate shader limits\n");
		exit(0);// Default return
	}

	else if(checker==2)
	{
		printf("\nInput matrix dimensions dont match\n");
		exit(0);// Default return
	}
	//printf("\nDimensions of input matrices dont match\n");

	else
	{
		IplImage *A=NULL;
		int i,j;
		//Top Half
		for(i=0;i<_GPUCV_TEXTURE_MAX_SIZE_X;i++)
		{
			for(j=0;j<B->width*4;j++)
			{	
				A->imageData[i*(B->width*4)+j]=B->imageData[i*(B->width*4)+j];
			}
		}
		//Bottom Half
		for(i=_GPUCV_TEXTURE_MAX_SIZE_X;i<A->height;i++)
		{
			for(j=0;j<1;j++)
			{	
				A->imageData[i*(B->width*4)+j]=B->imageData[(i-_GPUCV_TEXTURE_MAX_SIZE_X)*(B->width*4)+j];
			}
		}
		return A;
	}
}
/**  
* \brief Used to subtract two matrices
* 
* This function is used to subtract elements of 2 matrices, channel by channel.
* \param A-> First Input Matrix
* \param B-> Second Input Matrix
* \return A matrix in the IplImage format containing the result A-B
* \author Kshitiz Singh
*/
IplImage * customSubtract(IplImage *A,IplImage *B)
{
	int checker=1;
	// Checking if any of the input matrices violate the shader limits
	if((int)(A->width)>_GPUCV_TEXTURE_MAX_SIZE_X || (int)(A->height)>_GPUCV_TEXTURE_MAX_SIZE_Y || (int)(B->width)>_GPUCV_TEXTURE_MAX_SIZE_X || (int)(B->height)>_GPUCV_TEXTURE_MAX_SIZE_Y)
	{ 
		checker=0;
	}
	if(A->width != B->width || A->height != B->height)
	{
		printf("\nDimensions of input matrices dont match\n");
		checker =-1;
	}

	if(checker==0)
	{
		printf("\nInput matrix dimension(s) violate shader limits\n");
		return A;// Default return
	}

	else if(checker==-1)
	{
		printf("\nInput matrix dimensions dont match\n");
		return A;// Default return
	}

	else
	{
		if (!GetHardProfile()->IsCompatible(GenericGPU::HRD_PRF_2))
		{
			printf("Not Compatible");
			return NULL; 
		}
		IplImage *Output = NULL;

		SetThread();
		// Correcting : if dst = one of the sources, creating temp image
		// This temp image will be replacing C. By the end, C will be replaced.
		// Warning : performances issues. If possible, do not use A or B for dst.
		// Using ForceTexture here could lead to hazardous results...


		string chosen_filter;
		chosen_filter = GetFilterManager()->GetFilterName("FShaders/matrix/custom_subtract.frag");

		GetFilterManager()->Apply(chosen_filter,A,Output,A->width,A->height,&B,1);

		return Output;
		UnsetThread();
	}

	return NULL;
}

/** 
* \brief  To convert a given image into desired texture format.
* 
* The function converts an input IplImage to a desired texture format,
* which is more convenient for usage in our matrix multiplication function. 
* We convert image A with dimensions widthxheight into a width*heightx1(*4 for pixels) texture 
* which is the image we return. A note of caution is that this function should only be applied 
* to an image, ie the normalised or residual image, this function should NOT be used on the eigne face 
* matrix.
* \author Kshitiz Singh
*/
IplImage * convertToMatrixFormat(IplImage *A)
{

	int checker=1;
	// Checking if any of the input matrices violate the shader limits
	if((int)(A->width)>_GPUCV_TEXTURE_MAX_SIZE_X || (int)(A->height)>_GPUCV_TEXTURE_MAX_SIZE_Y )
	{ 
		checker=0;
	}

	if(checker==0)
	{
		printf("\nInput matrix dimension(s) violate shader limits\n");
		return A;// Default return
	}

	else
	{
		IplImage *B;
		B=cvCreateImage(cvSize(A->width*A->height,1),IPL_DEPTH_32F,4);
		int lk;//variable used only for the "for" loops
		switch(A->nChannels)// We choose variable stuffing according to the type of image at hand
		{

		case(1):// In case of a single channel image

			for(lk=0;lk<A->width*A->height*4;lk++)
			{
				if(lk%4==0)B->imageData[lk]=A->imageData[lk];
				if(lk%4==1)B->imageData[lk]=1;
				if(lk%4==2)B->imageData[lk]=1;
				if(lk%4==3)B->imageData[lk]=1;
			}
			return B;
			break;

		case(3):// In case of a three channel image

			for(lk=0;lk<A->width*A->height*4;lk++)
			{
				if(lk%4==0)B->imageData[lk]=A->imageData[lk];
				if(lk%4==1)B->imageData[lk]=A->imageData[lk];
				if(lk%4==2)B->imageData[lk]=A->imageData[lk];
				if(lk%4==3)B->imageData[lk]=1;
			}
			return B;
			break;
		case(4):// In case of a 4 channel rgba image

			for(lk=0;lk<A->width*A->height*4;lk++)
			{
				B->imageData[lk]=A->imageData[lk];
			}
			return B;
			break;
		}
	}

	return NULL;
}

#endif
