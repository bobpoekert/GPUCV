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

uniform float Parameters[2]; // 0->min 1->max



//define minrange
//define step

void main(void)
{

//float pos  = (float)minrange + Parameters[0] * (float)step;
//float lum  = (GETTEX(BaseImage, gl_TexCoord[0].st).r) * 255.;

float lum = GETTEX(BaseImage, gl_TexCoord[0].st).g;
if ((lum < Parameters[0]) || (lum >= Parameters[1])) discard;

//gl_FragColor = vec4(1., 0., 0., 1.);


//if ((lum < pos) || (lum > (pos+(float)step))) discard;

/*
if ((lum >= pos) && (lum < (pos+(float)step)))
   gl_FragColor = vec4(1., 0., 0., 1.);
else discard;
*/
}


