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
V=max(R,G,B)
S=(V-min(R,G,B))*255/V   if V!=0, 0 otherwise

       (G - B)*60/S,  if V=R
H= 180+(B - R)*60/S,  if V=G
   240+(R - G)*60/S,  if V=B

if H<0 then H=H+360
*/   

    vec4 color = GETTEX(BaseImage, gl_TexCoord[0].st).rgba;
    float R = color.r * 255.;
    float G = color.g * 255.;
    float B = color.b * 255.;
    
    float H;
    float V = max(max(R, G), B);
    float S;

    if (V == 0.) S = 0.;
    else S = (V-min(min(R, G), B))* 255. / V;
    
    if (int(V) == int(R))        H = (G-B)*60./S;
    else if (int(V) == int(G))   H = 180. + (B-R)*60./S;
    else                         H = 240. + (R-G)*60./S;


    if (H < 0.) H+=360.;

//The hue values calcualted using the above formulae vary from 0° to 360° 
//so they are divided by 2 to fit into 8-bit destination format.
    gl_FragColor = vec4(H/512., S/255., V/255., 1.).bgra;
}


