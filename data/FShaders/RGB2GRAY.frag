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
RGB[A]->Gray: Y=0.212671*R + 0.715160*G + 0.072169*B + 0*A
Gray->RGB[A]: R=Y G=Y B=Y A=0
*/   

//    vec3 color = GETTEX(BaseImage, gl_TexCoord[0].st).rgb;
    vec3 color = GETTEX(BaseImage, gl_TexCoord[0].st).rgb;
    float Y = color.r * .212671 + color.g * .715160 + color.b * .072169;
    
    gl_FragColor = vec4(Y, 0, 0, 0).rgba;
}


