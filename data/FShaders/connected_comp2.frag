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
//#define Pas 2


void main(void)
{
    vec4 color = GETTEX(BaseImage, gl_TexCoord[0].st);
    vec4 color_vois;
    float Pas = 1.;

	vec2 PP = gl_TexCoord[0].st - vec2(1.,1.);
	int i,j;
	
	if((color.r!=0)||(color.g!=0)||(color.b!=0))
	{
		for(i=1.;i>=0;i--)
			for(j=1.;j>=0;j--)
			{
				color_vois = GETTEX(BaseImage, PP+vec2(j,i));
				if((color_vois.r!=0)||(color_vois.g!=0)||(color_vois.b!=0))
					color = min(color_vois,color);
			}
		gl_FragColor = color;
	}
	else discard;
}


