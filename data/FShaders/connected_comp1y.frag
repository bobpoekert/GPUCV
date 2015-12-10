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
    vec4 colorNext = GETTEX(BaseImage, gl_TexCoord[0].st);
	vec2 PP = gl_TexCoord[0].st;
	vec2 PPdeltaText = {1/Parameters[0], 1/Parameters[1]};

	float r,g,b,a;
	r=0.;
	g=0.;
	b=0.;
	a=0.;

//	float offset = PP.y*Parameters[1]*Parameters[0]+PP.x*Parameters[0];
	
	if(vec3(color.y) != 0)
	{
		b = PP.x;
		g = PP.y;
		r = PP.x;
		a = PP.y;
		gl_FragColor = vec4(r,g,0.,0.);//vec4(color.y,color.y,color.y,0.);

		
/*		
		for (int i=0;i<50;i++)
		{
		    PP+= PPdeltaText;
		 	colorNext = GETTEX(BaseImage, PP);
			if(colorNext.rgb==vec3(0.)) i=1000;
		}

		PP-= 5*PPdeltaText;
		PPdeltaText.x = -PPdeltaText.x;
		color = colornext;
		for (i=0;i<10;i++)
		{
		    PP+= PPdeltaText;
		 	colorNext = GETTEX(BaseImage, PP);
			if(colorNext.rgb==vec3(0.)) i=1000;
			if ((color.g < colorNext.g&))
			   
		}
*/
		


/*		for (i=0;i<50;i++)
		{
		    PP+= PPdeltaText;
		 	colorNext = GETTEX(BaseImage, PP);
			if(colorNext.rgb==vec3(0.)) i=1000;
		}
	*/	
/*
		r = PP.x;
		g = PP.y;
		b = PP.x;
		a = PP.y;
		

		for(int i=0;i<3;i++)
			if ((PP.y + i/sizey) < ((sizey-1)/sizey))     
				for(j=0;j<3;j++)
					if ((PP.x + j/sizex) < ((sizex-1)/sizex))
					{
					  //colorNext.rgba= GETTEX(BaseImage, PP+vec2(j/sizex,i/sizey)).rgba;
					  colorNext= GETTEX(BaseImage, PP+vec2(j/sizex,i/sizey));
					  if(IsMax(color.rg, colorNext.rg))color.rg = colorNext.rg;
				//	  if(IsMin(color.ra, colorNext.ra))color.ra = colorNext.ra;
					}

		gl_FragColor = vec4(r,g,0.,0.);//vec4(color.y,color.y,color.y,0.);
*/
	}
	else 
		 discard;


	
}


