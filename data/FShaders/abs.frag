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
#define GPUCV_FILTER 1 
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
#define DEF_IMG2	0
#define DEF_OPER	//ex:abs(img0+img1)
#define SWIZZLE_ORDER bgra
#define REVERSE_SWIZZLE_ORDER rgba

#if DEF_SCALAR
	#define PARAMETER_NBR 4	
#else
	#define PARAMETER_NBR	0
#endif

IMGTYPE BaseImage; // source imag 

#if PARAMETER_NBR
	uniform float Parameters[PARAMETER_NBR];
#endif

#if DEF_IMG2
	IMGTYPE Image0;//src2
#endif


void main(void)
{
//get main source
	vec4 img0 = GETTEX(BaseImage, gl_TexCoord[0].st).SWIZZLE_ORDER;

//get seconde source [scalar|image]
	#if DEF_SCALAR
		vec4 img1 = vec4 (Parameters[0],Parameters[1],Parameters[2],Parameters[3]).REVERSE_SWIZZLE_ORDER;
	#elif DEF_IMG2
		vec4 img1 = GETTEX(Image0, gl_TexCoord[0].st).SWIZZLE_ORDER;
	#endif
	
//process operator
	gl_FragColor = DEF_OPER.SWIZZLE_ORDER; //default addition
}
