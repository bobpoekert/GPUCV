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
#define DELTA .5

void main(void)
{
/*
R=Y + 1.403*(Cr - 128)
G=Y - 0.344*(Cr - 128) - 0.714*(Cb - 128)
B=Y + 1.773*(Cb - 128)
*/   
    vec4 color = GETTEX(BaseImage, gl_TexCoord[0].st).bgra;
//	float Y = 0.299*color.r + 0.587 *color.g + 0.114*color.b;
    gl_FragColor = vec4(color.r + 1.403*(color.g - DELTA),
                        color.r - .344*(color.g-DELTA) - .714*(color.b - DELTA),
                        color.r + 1.773*(color.b-DELTA), 
                        1.).rgba;
}
