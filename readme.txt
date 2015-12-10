=======================================================================
=======================================================================
			GpuCV library:
		Version: GpuCV 1.0.0 rev 598

License: CeCILL-B (http://www.cecill.info/index.en.html)
URL: https://picoforge.int-evry.fr/cgi-bin/twiki/view/Gpucv/Web/WebHome
Contact: gpucv-developers (at) picoforge.int-evry.fr
OS: MS Windows XP, LINUX.
ARCH: x86 32/64 bits.
=======================================================================
=======================================================================


1-Description:
===============================================
GpuCV is an open-source GPU-accelerated image processing and Computer Vision library. It offers an Intel's OpenCV-like programming interface for easily porting existing OpenCV applications, while taking advantage of the high level of parallelism and computing power available from recent graphics processing units (GPUs). It is distributed as free software under the CeCILL-B license.

It is designed to be easily integrated into a existing OpenCV application with minor code changes. It also supply a framework and a toolchain to design new operator accelerated on GPU using OpenGL+GLSL or NVIDIA CUDA libraries.


2-Features:
===============================================

2-A- OpenCV porting 
_______________________________________________
Full compatibility with OpenCV:
- GpuCV operators definitions are based on OpenCV definitions. 
- Support of native OpenCV structures like CvArr, CvMat, IplImage... 
- Automatic data transfer management between central memory and graphics memory. 

Dynamic switching mechanism:
- Custom switching mechanisms between GpuCV and OpenCV operators are available to fit your application needs. 

Graphic programming level abstraction:
- Graphics programming layer based on OpenGL and GLSL is handled by the framework.. 
- Automatic format conversion between OpenCV image format/type and OpenGL image format/type. 
- Automatic data management between GpuCV and OpenCV 

Correspondances of OpenCV operators ported with GpuCV:
- CXCORE library 
	- Operation on array: 
		Initialization: cvCreateImage, cvCreateMat, cvReleaseImage, cvReleaseMat, cvCloneImage, cvCloneMat, cvGetRawData, cvSetData. 
		Copying and Filling: cvCopy, cvSetZero 
		Transforms and Permutations: cvSplit 
		Arithmetic, Logic and Comparison: cvgAdd, cvAddS, cvConvertScale, cvDiv, cvMax, cvMaxS, cvMin, cvMinS, cvMul, cvSub, cvSubRS, cvSubS 
		Statistics: cvAvg, cvSum, cvMinMaxLoc. 
		Linear Algebra: cvScaleAdd, cvGEMM 
		Math Functions: cvPow 
- CV library 
	- Image Processing: 
		Sampling, Interpolation and Geometrical Transforms: cvResize 
		Morphological Operations: cvDilate, cvErode, cvMorphologyEx 
		Filters and Color Conversion: cvCvtColor, cvThreshold 
		Histograms: cvQueryHistValue_*D 
- HIGHUI library
	cvShowImage, cvSaveImage.
And more...

2-B- Examples applications 
_______________________________________________
- GpuCVConsole: a console demo application to run and test all the GpuCV operators, it includes a benchmark system to compare GpuCV and OpenCV operators performances. 
- GpuCVCamDemo: a demo application to view real time morphologic processing on a video stream using OpenCV or GpuCV. 


2-C Project architecture 
_______________________________________________
The project is divided into two main parts: 
- GpuCV main library: contains all the OpenCV functions already ported on GPU. 
- GpuCV core libraries: contains a framework (GPUCVCore/GPUCVTexture/GPUCVHardware) for GPGPU (General-Purpose computation on GPUs) based on OpenGL and GLSL shader language.



3-GpuCV framework 
===============================================
GpuCV framework is design to offers tools to developers that wish to create new GpuCV or GPGPU operators. 

3-A Shading processing 
_______________________________________________
GpuCV is using shader programs on the graphics card to process the data. Here are the main characteristics: 
- A shader manager to load and manipulate shader programs. 
- Support of GLSL version 2.0?? for vertex and fragment shaders. 
- Meta-tag mechanisms for custom shader compiling. 
- Shader input surfaces are simple/multi textured quad, drawing list, or custom drawing function. 
- Automatic shader reloading after shader file edition during runtime. 


3-B Texture support: 
_______________________________________________
GpuCV is using OpenGL textures to store data to process and results. Here are the main characteristics: 
- A texture manager to load and manipulate textures. 
- Support for GL_TEXTURE_2D and GL_TEXTURE_REACTANGLE_ARB texture type. 
- Memory allocation=> "On request" approach 
- Multiple texture locations with Ubiquity mechanism. 
- Render to texture manager based on FBO and RenderBuffer? OpenGL objects. 
- Texture group objects to manipulate several texture simultaneously. 
- Texture recycler to avoid texture allocations/deallocations. 
- Custom texture coordinates to allow processing on ROI 5region Of Interest). 


3-C Global
_______________________________________________
- Embedded benchmarking tools. 
- Multi platforms building scripts. 
- Stack push/pop mechanisms to manage objects states. 
- Automatic GPU(s) models and hardware compatibilities detection. 
- Automatic OpenGL context saving/switching for multi-threaded applications. 
- Flexible framework architecture independent from OpenCV to be used in other GPGPU applications.


