//CVG_LicenseBegin============================================================================
// Short license for shaders:
//	Copyright@ Institut TELECOM 2005 http://www.institut-telecom.fr/en_accueil.html
//	This file is part of GpuCV library.
//	Contacts : gpucv@picoforge.int-evry.fr or gpucv-developers@picoforge.int-evry.fr
//	Project's Home Page : https://picoforge.int-evry.fr/cgi-bin/twiki/view/Gpucv/Web/WebHome
//	License: CeCILL-B license "http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html". 
//==============================================================================CVG_LicenseEnd
//CVG_METASHADERBegin
//! Defined to 1 when compiled into GpuCV
//! Defined to 0 when compile by external applications such as "Shader Designer"
#define GPUCV_FILTER 0 
#if GPUCV_FILTER 
	#define GETTEX 			// function to get texture
	#define IMGTYPE 		// texture type
#else
	#define GETTEX 		texture2D
	#define IMGTYPE 	uniform sampler2D
#endif
//!!!!! WARNING DO NOT WRITE ANYTHING IN THIS LICENSE OR METASHADER PART !!!!!!!!!!!!
//!!!!! CAUSE IT IS SELF GENERATED !!!!!!!!!!!!!!!!
//---------------------CVG_METASHADEREnd
IMGTYPE BaseImage; // source image

#define DEF_SRC2	0
#define DEF_SCALAR	0
#define DEF_MASK	0
#define DEF_SCALE_1	0
#define DEF_SCALE_2	0
#define DEF_GLOBAL_FACTOR	
#define DEF_PARAMETER_NBR	0
#define DEF_OPERATION	
#define SWIZZLE_ORDER bgra
#define REVERSE_SWIZZLE_ORDER rgba



#if DEF_SCALAR && DEF_SCALE_1
	#define DEF_PARAMETER_NBR 8
	#define SCALE_PARAMETER vec4(Parameters[4], Parameters[5], Parameters[6], Parameters[7]).REVERSE_SWIZZLE_ORDER
#elif DEF_SCALAR
	#define DEF_PARAMETER_NBR 4	
#elif DEF_SCALE_1
	#define DEF_PARAMETER_NBR 1
	#define SCALE_PARAMETER *Parameters[0]
#endif

#if DEF_PARAMETER_NBR
	uniform float Parameters[DEF_PARAMETER_NBR];
#endif

#if DEF_SCALAR && DEF_MASK
	IMGTYPE Image0;
	#define MASK_IMAGE Image0
#elif DEF_SRC2 && DEF_MASK
	IMGTYPE Image0;//src2
	IMGTYPE Image1;//mask
	#define MASK_IMAGE Image1
#elif DEF_MASK
	IMGTYPE Image0;
	#define MASK_IMAGE Image0
#elif DEF_SRC2
	IMGTYPE Image0;//src2
#endif

#define FCT //function to proccess neighboorhoods pixels
#define FCT_INIT //function to proccess neighboorhoods pixels
#define FCT_FINAL //function to proccess neighboorhoods pixels

void main(void)
{
//get main source
	vec4 img0 = GETTEX(BaseImage, gl_TexCoord[0].st).SWIZZLE_ORDER;

//get seconde source [scalar|image]
	#if DEF_SCALAR
		vec4 img1 = vec4 (Parameters[0],Parameters[1],Parameters[2],Parameters[3]).REVERSE_SWIZZLE_ORDER;
	#elif DEF_SRC2
		vec4 img1 = GETTEX(Image0, gl_TexCoord[0].st).SWIZZLE_ORDER;
	#endif
	
//get Mask
	#if DEF_MASK
		vec4 imgMask = GETTEX(MASK_IMAGE, gl_TexCoord[0].st).SWIZZLE_ORDER;
	#endif

//process operation
#if DEF_MASK
	if (imgMask.r != 0.)
#endif
		gl_FragColor = ( img0 DEF_OPERATION  img1 ).SWIZZLE_ORDER DEF_GLOBAL_FACTOR; //default addition
#if DEF_MASK
	else 
		gl_FragColor = vec4 (0.).SWIZZLE_ORDER; //mask
#endif
}
 


