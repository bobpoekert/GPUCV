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


bool isValue(float value, float compare_value)
{
	if ((value>=compare_value)&&(value<=compare_value+1))
		return(true);
	else
		return(false);
}

void main(void)
{
	vec2 PP = gl_TexCoord[0].st;
	vec2 PP2;
	float x = PP.x*Parameters[0];
	float y = PP.y*Parameters[1];
	vec4 sum = vec4(0.0,0.0,0.0,0.0);

	if ((x >= 255)&&(isValue(y,254)))
	{
		for(int i=0;i<140;i++)
		{
			PP2 = PP+vec2(0.0,(float)i/Parameters[1]);
			sum = sum + GETTEX(BaseImage, PP2);
		}
		gl_FragColor = sum;
	}
	else discard;
	//gl_FragColor = vec4(0.8,0.8,0.8,0.0);	
}


