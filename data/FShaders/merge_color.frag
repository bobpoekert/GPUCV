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
#define INPUT_R 0
#define INPUT_G 1
#define INPUT_B 0
#define INPUT_A 0
/*
#define INPUT_0_C
#define INPUT_1_C
#define INPUT_2_C
#define INPUT_3_C
*/
//#define SINGLE_INPUT 1
/*
IMGTYPE BaseImage; // not used
IMGTYPE Image0; // not used
IMGTYPE Image1; // not used
*/
#if INPUT_R
	IMGTYPE inputR;
#endif
#if  INPUT_G
	IMGTYPE inputGreen;
#endif
#if INPUT_B
	IMGTYPE inputB;
#endif
#if INPUT_A
	IMGTYPE inputA;
#endif


void main(void)
{
	vec4 mergePixel=vec4(0.);
#if INPUT_R
	mergePixel.r = GETTEX(inputR, gl_TexCoord[0].st).r;
#endif
#if INPUT_G
	mergePixel.g = GETTEX(inputGreen, gl_TexCoord[0].st).r;
#endif
#if INPUT_B
	mergePixel.b = GETTEX(inputB, gl_TexCoord[0].st).r;
#endif
#if INPUT_A
	mergePixel.a = GETTEX(inputA, gl_TexCoord[0].st).r;
#endif
	gl_FragColor = mergePixel;
}