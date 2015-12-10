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
uniform float Parameters[2]; // Parameter : Image width


void main(void)
{

    vec4 color = GETTEX(BaseImage, gl_TexCoord[0].st);
	vec2 PP = gl_TexCoord[0].st;
	float r,g,b;
	r=0.;
	g=0.;
	b=0.;
	
	// Pixels are tagged
	float offset = PP.y*Parameters[1]*Parameters[0]+PP.x*Parameters[0];
//	float value = PP.y/256.;
	
	if(color.r != 0)
	{
		r = floor(offset/(65536.0))/256;
		g = floor((offset-r*65536.0)/256.0)/256;
		b = mod(offset,256)/256;
	}
	
	gl_FragColor = vec4(r,g,b,1.0);
}


