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
#define SWIZZLE_ORDER bgra
#define REVERSE_SWIZZLE_ORDER rgba

#define CMP_OPER   // operator
#define DEF_SCALAR	0
#define DEF_IMG2	0


#if DEF_SCALAR
	#define PARAMETER_NBR	1	
#else
	#define PARAMETER_NBR	0
#endif

IMGTYPE BaseImage; // source imag 

#if PARAMETER_NBR
	uniform float Parameters[PARAMETER_NBR];
#endif

#if DEF_IMG2
	IMGTYPE Image0;//src2
#endif

#define PIXELTYPE int //comparing == with float does not give always accurate results when input are char or int...
//it is set to float when required


void main(void)
{
	vec4 imgA= GETTEX(BaseImage, gl_TexCoord[0].st);

//get seconde source [scalar|image]
	vec4 imgB=0.;
	#if DEF_SCALAR
		imgB.r=Parameters[0];
	#elif DEF_IMG2
		imgB = GETTEX(Image0, gl_TexCoord[0].st);
	#endif
	
	//convert to a temporary format...
	ivec4 A=0;
	ivec4 B=0;
	A.r = int(256.*imgA.r);
	B.r = int(256.*imgB.r);
	
//process operator
	if((A.r CMP_OPER B.r))
		gl_FragColor = 1.;
	else
		gl_FragColor = 0.;
}