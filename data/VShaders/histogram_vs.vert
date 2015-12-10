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
uniform float Parameters[2];

void main(void)
{
	// Texture position
	gl_TexCoord[0]  = gl_MultiTexCoord0;

	// Vertex texture fetch
	vec2 PP = gl_MultiTexCoord0.st;
	vec4 color0 = GETTEX(BaseImage, PP);
	
	// Line computing
	float x = PP.x*Parameters[0];
	float y = PP.y*Parameters[1];
	float offset_decal = floor((y*Parameters[1]+x)/2048.)/Parameters[0]*2.0;
	
	// Coordinates Computation
	float lum =	color0.r*Parameters[1]/(Parameters[1]+1.0);
	
//	vec4 vert = vec4(gl_Vertex[0],gl_Vertex[1],gl_Vertex[2],gl_Vertex[3]);

//	vec4 vert = vec4(lum,0,gl_Vertex[2],gl_Vertex[3]);
    vec4 vert = vec4(lum,offset_decal,gl_Vertex[2],gl_Vertex[3]);

	// Vertex position, perturbed with r channel
	gl_Position = gl_ModelViewProjectionMatrix * vert;
}
