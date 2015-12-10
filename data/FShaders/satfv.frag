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
uniform float Parameters[4];	//1-width
								//2-height
								//3-factor

	
void main(void)
{	
	vec4 img1 = GETTEX(BaseImage, gl_TexCoord[0].st).rgba;
	vec2 offset = gl_TexCoord[0].st;
	vec4 img2;
	vec4 img3;
	offset.y-=Parameters[3]/Parameters[1];
	if (offset.y <= 0) 
	img2=0;
	else
	img2 = GETTEX(BaseImage, offset).rgba;
	img3 = img1 + img2;
	gl_FragColor = img3;
}
