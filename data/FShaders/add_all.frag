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

#define DEF_SCALAR	0
#define DEF_MASK	0
#define DEF_SCALE	0
#define SWIZZLE_ORDER bgra
#define REVERSE_SWIZZLE_ORDER rgba
#define SCALE_PARAMETER 1



#if DEF_SCALAR && DEF_SCALE
	#define PARAMETER_NBR 8
	#define SCALE_PARAMETER vec4(Parameters[4], Parameters[5], Parameters[6], Parameters[7]).REVERSE_SWIZZLE_ORDER
#elif DEF_SCALAR
	#define PARAMETER_NBR 4	
#elif DEF_SCALE
	#define PARAMETER_NBR 4
	#define SCALE_PARAMETER vec4(Parameters[0], Parameters[1], Parameters[2], Parameters[3]).REVERSE_SWIZZLE_ORDER
#else
	#define PARAMETER_NBR	0
#endif

IMGTYPE BaseImage; // source imag 

#if PARAMETER_NBR
	uniform float Parameters[PARAMETER_NBR];
#endif

#if DEF_SCALAR && DEF_MASK
	IMGTYPE Image0;
	#define MASK_IMAGE Image0
#elif DEF_MASK
	IMGTYPE Image0;//src2
	IMGTYPE Image1;//mask
	#define MASK_IMAGE Image1
#else
	IMGTYPE Image0;//src2
#endif

	
	
	

void main(void)
{
//get main source
	vec4 img0 = GETTEX(BaseImage, gl_TexCoord[0].st).SWIZZLE_ORDER;

//get seconde source [scalar|image]
	#if DEF_SCALAR
		vec4 img1 = vec4 (Parameters[0],Parameters[1],Parameters[2],Parameters[3]).REVERSE_SWIZZLE_ORDER;
	#else
		vec4 img1 = GETTEX(Image0, gl_TexCoord[0].st).SWIZZLE_ORDER;
	#endif
	
//get Mask and process addition	
#if DEF_MASK
	vec4 imgMask = GETTEX(MASK_IMAGE, gl_TexCoord[0].st).SWIZZLE_ORDER;
	if (imgMask.r == 0.)
		gl_FragColor = vec4 (0.);//.SWIZZLE_ORDER; //mask
	else
#endif
		gl_FragColor = (img0 * SCALE_PARAMETER + img1).SWIZZLE_ORDER; //default addition

}
