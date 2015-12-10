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

void main(void)
{
/*
|X|   |0.412411  0.357585  0.180454| |R|
|Y| = |0.212649  0.715169  0.072182|*|G|
|Z|   |0.019332  0.119195  0.950390| |B|
*/    
   mat3 matrix = mat3(0.412411, 0.212649, 0.019332,
                        0.357585, 0.715169, 0.119195,
                        0.180454, 0.072182, 0.950390);
   
	vec3 Color = GETTEX(BaseImage, gl_TexCoord[0].st);
	
	//gl_FragColor = vec4(Color.bgr, 1.).rgba;
	gl_FragColor = vec4((matrix * GETTEX(BaseImage, gl_TexCoord[0].st).rgb).bgr, 1.).rgba;
	//gl_FragColor = vec4((matrix * GETTEX(BaseImage, gl_TexCoord[0].st).bgr).b,0.,0., 1.).bgra;
}