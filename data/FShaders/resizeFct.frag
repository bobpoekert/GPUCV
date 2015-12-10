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
uniform float Parameters[4]; // Parameter : Image width
#define FCT //function to proccess neighboorhoods pixels
#define FCT_INIT //init function
#define FCT_FINAL //final function

void main(void)
{

	vec2 delta		= vec2(Parameters[0]/Parameters[2]/2., Parameters[1]/Parameters[3]/2.);
	vec2 PPdelta 	= vec2(1./Parameters[0], 1./Parameters[1]);
	vec2 PP = gl_TexCoord[0].st;
	vec4 colorTemp = vec4(0.);
	vec4 color=vec4(0.);// = GETTEX(BaseImage, gl_TexCoord[0].st);

	float minX, maxX, minY, maxY;
	maxX = PPdelta.x*delta.x;
	minX = -maxX;
	maxY = PPdelta.y*delta.y;
	minY = -maxY;
	
	if (PP.x + minX <=0.) minX = -PP.x;
	if (PP.x + maxX >=1.) maxX = 1.-PP.x;
	if (PP.y + minY <=0.) minY = -PP.y;
	if (PP.y + maxY >=1.) maxY = 1.-PP.y;
	
	
	FCT_INIT

   for (float x= minX; x < maxX; x+=PPdelta.x)
   {
    for (float y= minY; y < maxY; y+=PPdelta.y)
    {
     PP = gl_TexCoord[0].st+vec2(x,y);
     //if ((PP.x>=0) && (PP.x <=1))
     // if ((PP.y>=0) && (PP.y <=1))
	 // {
   	   colorTemp = GETTEX(BaseImage, PP);
   	   FCT
	 // }
    }
   }
   FCT_FINAL
}
 


