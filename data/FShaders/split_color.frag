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
#define OUTPUT_0 0
#define OUTPUT_1 0
#define OUTPUT_2 0
#define OUTPUT_3 0
#define OUTPUT_0_C
#define OUTPUT_1_C
#define OUTPUT_2_C
#define OUTPUT_3_C
#define SINGLE_OUTPUT 1
IMGTYPE BaseImage; // source image

void main(void)
{
#if 0//SINGLE_OUTPUT
	gl_FragColor = vec4(GETTEX(BaseImage, gl_TexCoord[0].st).SINGLE_OUTPUT);
#else
	vec4 val = GETTEX(BaseImage, gl_TexCoord[0].st).bgra;
	#if OUTPUT_0
		gl_FragData[0] = vec4(val.OUTPUT_0_C);
	#endif
	#if OUTPUT_1
		gl_FragData[1] = vec4(val.OUTPUT_1_C);
	#endif
	#if OUTPUT_2
		gl_FragData[2] = vec4(val.OUTPUT_2_C);
	#endif
	#if OUTPUT_3
		gl_FragData[3] = vec4(val.OUTPUT_3_C);
	#endif
#endif
}