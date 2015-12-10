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
uniform samplerRect BaseImage;
uniform float Parameters[4];


void main(void)

{
	int i,j;
	float decx = 1./Parameters[0];
	float decy = 1./Parameters[1];
    int Ksizex = Parameters[2];
    int Ksizey = Parameters[3];

	vec4 color0;
	vec2 Origin = gl_TexCoord[0].st - vec2(Ksizex/2 * decx, Ksizey/2 * decy);

	color0 = GETTEX(BaseImage, gl_TexCoord[0].st);

	for(i=0; i < Ksizey;i++)
		for(j=0; j < Ksizex;j++)
			color0 = min(GETTEX(BaseImage, Origin+vec2(float(j)*decx,float(i)*decy)),color0);

	gl_FragColor = color0;
}
