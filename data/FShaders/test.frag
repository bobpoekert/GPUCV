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
//---------------------CVG_METASHADEREnd//ARToolkit on GPU
//Yannick ALLUSSE
//HitLab NZ
//=====================================


//Create ColorMap pattern from black and white map pattern
//CVG_METASHADERBegin
#define GETTEX //function to get texture
#define IMGTYPE //texture type
//---------------------CVG_METASHADEREnd

IMGTYPE BaseImage; // source imag
uniform float Parameters[2];
void main(void)
{
	vec4 pix;
	
	vec2 Pos = gl_TexCoord[0].st;
	
	vec2 Xoffset;
	Xoffset.x = (floor(Pos.x*Parameters[0]) /Parameters[0])/Parameters[0];
	Xoffset.y = (floor(Pos.y*Parameters[1]) /Parameters[1])/Parameters[1];
	
	
	
	pix.r = GETTEX(BaseImage, Pos).r;
	pix.g = GETTEX(BaseImage, vec2(Xoffset.y - Pos.y, Pos.x)).r;
	pix.b = GETTEX(BaseImage, vec2(Xoffset.x - Pos.x, Xoffset.y - Pos.y)).r;
	pix.a = GETTEX(BaseImage, vec2(Pos.y, Xoffset.x - Pos.x)).r;

         
    gl_FragColor = 1- pix;
}
