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
#define DEF_MASK	0
#define SWIZZLE_ORDER bgra
#define REVERSE_SWIZZLE_ORDER rgba


IMGTYPE BaseImage; 	// source imag 
#if DEF_MASK
	IMGTYPE Image0;	//mask
#endif

void main(void)
{
    vec4 img0 = GETTEX(BaseImage, gl_TexCoord[0].st).SWIZZLE_ORDER;

#if DEF_MASK
	vec4 imgMask = GETTEX(Image0, gl_TexCoord[0].st).SWIZZLE_ORDER;
	if (imgMask.r != 0.)
#endif
		gl_FragColor = img0.SWIZZLE_ORDER; //default addition
#if DEF_MASK
	else 
		discard;
#endif
}
