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
*	\brief Header file containg definitions for the GPU equivalent OpenCV/Highgui functions.
*	\author Jean-Philippe Farrugia
*	\author Yannick Allusse
*/
#ifndef __GPUCV_HIGHGUI_H
#define __GPUCV_HIGHGUI_H
#include <highgui.h>
#include <highguig/config.h>
#ifdef __cplusplus
#	include <GPUCV/misc.h>
#	include <GPUCVHardware/GlobalSettings.h>
#endif


//Highgui reference =============================================================
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @ingroup HIGHGUI__LOAD_SAVE_IMG_GRP
*  @{
*	\todo fill with opencv functions list.
*/
/*! 
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_highgui.htm#decl_cvShowImage" target=new>cvShowImage</a> function.
*  If image is stored into texture in GPU memory, texture is copied back into ram causing lower performances.
*	\todo => when image is sored into a texture, open a glut windows to show texture....
*	\param name -> Name of the window.
*	\param image -> Image to be shown.
*	\param _mipmapLevel -> mipmap LOD to show, image size shown will be reduced by 2^_mipmapLevel. 
*	\sa DataContainer::AUTO_MIPMAP for details.
*	\author Yannick Allusse
*/
_GPUCV_HIGHGUIG_EXPORT_C 
void cvgShowImage(const char* name, CvArr* image);//, unsigned int _mipmapLevel CV_DEFAULT(0));

/*! 
*	\brief - A function mainly to be used for debugging and analysing.Displays all the properties of the image on the image itself.
*	\param name -> Name of the window.
*	\param image -> Image to be shown.
*	\author Ankit Agarwal
*/
_GPUCV_HIGHGUIG_EXPORT_C 
void cvgShowImage2(const char* name,IplImage* image);

/*! 
*	\brief - A function mainly to be used for debugging and analysing.Displays all the pixel value so of image on console..
*	\param name -> Name of the window.
*	\param image -> Image to be shown.
*	\param start -> Starting pixel
*	\param width -> box width
*	\param height-> box height
*	\author Ankit Agarwal
*/
_GPUCV_HIGHGUIG_EXPORT_C 
void cvgShowImage3(const char* name,IplImage *image, unsigned int start_x CV_DEFAULT(0), unsigned int start_y CV_DEFAULT(0), unsigned int width CV_DEFAULT(0), unsigned int height CV_DEFAULT(0));


/*! 
*	\brief - A function mainly to be used for debugging and analysing.Displays the contents of a matrix
*	\param label -> label for the contents.
*	\param mat -> matrix to be displayed.
*	\author Ankit Agarwal
*/

_GPUCV_HIGHGUIG_EXPORT_C 
void cvgShowMatrix(const char* label,CvMat *mat,CvPoint start  CV_DEFAULT(cvPoint(0,0)),int width CV_DEFAULT(0),int height CV_DEFAULT(0));


#ifdef __cplusplus
/*! 
*	\brief - A function mainly to be used for debugging and analysing.Displays all the pixel value so of image on console..
*	\param name -> Name of the window.
*	\param image -> image to be shown.
*	\param start -> Starting pixel
*	\param width -> box width
*	\param height-> box height
*	\author Ankit Agarwal
*/
template <typename imgtype,int channels,typename displaytype>
void cvgShowImaget(const char* name,IplImage *image, CvPoint start,int width, int height,int box_width,int box_height , int offset)
{
	using namespace GCV;
	GPUCV_NOTICE("_____________________________"<<name<<"______________________");
	imgtype * Databuff = (imgtype*)image->imageData;
	int SumPitch = image->width*image->nChannels;
	int x_pos,y_pos;

	for (int j = 0 ; j < box_height; j++)
	{
		y_pos = j + start.y;

		std::cout << "\n";
		std::cout <<  "Line:" << y_pos << "\t";

		for (int i= 0 ; i < box_width * channels ; i+=channels)
		{	
			x_pos = i + start.x;
			std::cout << "("; 
			for (int k=0 ; k < channels ; k++)

			{	//char *str;
				displaytype val = (displaytype)Databuff[y_pos*SumPitch+x_pos]+offset;
				std::string s = SGE::ToCharStr(val);
				s.resize(4);
				std::cout << s
					<< ",";


			}
			std::cout << ")"
				<< "\t";

		}
		std::cout << "\n";
	}
}



/*! 
*	\brief - A function mainly to be used for debugging and analysing.Displays all the pixel value so of matrix on console..
*	\param name -> Name of the window.
*	\param matrix -> matrix to be shown.
*	\param start -> Starting pixel
*	\param width -> box width
*	\param height-> box height
*	\author Ankit Agarwal
*/
template <typename mattype,int channels>
void cvgShowMatrixt(const char* label,CvMat* matrix, CvPoint start,int width, int height,int box_width,int box_height )
{
	using namespace GCV;
	GPUCV_NOTICE("\n\n_____________________________"<<label<<"______________________");
	mattype * Databuff = (mattype*)matrix->data.ptr;
	int SumPitch = matrix->step;
	int x_pos,y_pos, pos;

	for (int j = 0 ; j < box_height; j++)
	{
		y_pos = j + start.y;
		std::cout << std::endl << "Line:" << y_pos << "\t";
		for (int i= 0 ; i < box_width; i+=channels)
		{
			x_pos = i + start.x;
			pos = y_pos*width+x_pos;
			std::cout << "(";
			switch (channels)
			{
				case 1: 
					std::cout << Databuff[pos];
					break;
				case 2:
					std::cout <<Databuff[pos]<< ","
					   <<	Databuff[pos+1];
				   break;
				case 3:
					std::cout <<Databuff[pos]<< ","
						<<	Databuff[pos+1]<< ","
						<<	Databuff[pos+2];
					break;
				case 4: 
					std::cout <<Databuff[pos]<< ","
						<<	Databuff[pos+1]<< ","
						<<	Databuff[pos+2]<< ","
						<<	Databuff[pos+3];
					break;
			}
			std::cout << ")\t";
		}
	}
	GPUCV_NOTICE("_________________________________________________________");
}
#endif//__cplusplus

/*! 
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_highgui.htm#decl_cvSaveImage" target=new>cvSaveImage</a> function.
*  If image is stored into texture in GPU memory, texture is copied back into ram and then saved.
*	\param filename -> name of the file target.
*	\param image -> Image to be saved.
*	\author Yannick Allusse
*/
_GPUCV_HIGHGUIG_EXPORT_C
int
cvgSaveImage(const char* filename, CvArr* image );
/** @}*///HIGHGUI__LOAD_SAVE_IMG_GRP
//_______________________________________________________________
//_______________________________________________________________
//_______________________________________________________________
/** @ingroup HIGHGUI__VIDEO_IO_GRP GPUCV
*  @{
*	\todo fill with opencv functions list.
*/
/*! 
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_highgui.htm#decl_cvRetrieveFrame" target=new>cvSaveImage</a> function.
*  Call the OpenCV cvRetrieveFrame function and tell GpuCV that data are now on CPU. It does not transfer any data
*	\param capture -> capture object
*	\return Return the captured IplImage.
*	\warning !!!DO NOT RELEASE or MODIFY the retrieved frame!!! 
*	\author Yannick Allusse
*/
_GPUCV_HIGHGUIG_EXPORT_C
IplImage* cvgRetrieveFrame(CvCapture* capture, int streamIdx);

/*! 
*	\brief GPUCV correspondence of <a href="http://picoforge.int-evry.fr/projects/svn/gpucv/opencv_doc/ref/opencvref_highgui.htm#decl_cvRetrieveFrame" target=new>cvSaveImage</a> function.
*  Call the OpenCV cvgswQueryFrame function and tell GpuCV that data are now on CPU. It does not transfer any data
*	\param capture -> capture object
*	\return Return the captured IplImage.
*	\warning !!!DO NOT RELEASE or MODIFY the retrieved frame!!! 
*	\author Yannick Allusse
*/
_GPUCV_HIGHGUIG_EXPORT_C
IplImage* cvgQueryFrame(CvCapture* capture);

/** @}*///HIGHGUI__VIDEO_IO_GRP
//_______________________________________________________________
//_______________________________________________________________

#endif
