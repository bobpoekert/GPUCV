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
uniform float Parameters[11];


void main(void)

{
vec2 Loop;
	vec2 Psize = vec2(Parameters[0], Parameters[1]);
	Psize = 1./Psize;//get pixelsize
	
	vec4 color0;
	vec2 PP = gl_TexCoord[0].st - Psize.st;

	color0 = GETTEX(BaseImage, gl_TexCoord[0].st);

	for(Loop.t=0.;Loop.t<3.;Loop.t++)
		for(Loop.s=0.;Loop.s<3.;Loop.s++)
		{
		
			if(Parameters[Loop.t*3.+Loop.s+2.]==1.0)
				color0 = min(GETTEX(BaseImage, PP+ Loop*Psize),color0);	//PP+vec2(j/sizex,i/sizey))
		}
	
	gl_FragColor = color0;	
#if 0
	int i,j;
	vec2 Psize = vec2(Parameters[0], Parameters[1]);
	Psize = 1./Psize;//get pixelsize
	

	vec4 color0;
	vec2 PP = gl_TexCoord[0].st - Psize.st;
	color0 = GETTEX(BaseImage, gl_TexCoord[0].st);

	for(i=3;i>=0;i--)
		for(j=3;j>=0;j--)
		{
			if(Parameters[i*3+j+2]==1.0)
				color0 = min(GETTEX(BaseImage, PP+vec2(j*Psize.s,i*Psize.t)),color0);
		}

    gl_FragColor = color0;
#endif
}
