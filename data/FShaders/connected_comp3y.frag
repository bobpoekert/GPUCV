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
uniform float Parameters[3]; // Parameter : Image width


/*
vec2 GetPixelPos(vec4 PixelPtr)//, vec2 Pos)
{
	vec2 Pos;

		//Pos.x= PixelPtr.r;//*Parameters[0];
		Pos.x= floor(PixelPtr.r*Parameters[0])/Parameters[0];
		//Pos.y= PixelPtr.g;//*Parameters[1];
		Pos.y= floor(PixelPtr.g*Parameters[1]+0.)/Parameters[1];

		//correct error...
		if (Pos.y > (0.94))///Parameters[1])*(Parameters[1]))
		   Pos.y -= 1/Parameters[1];
		  
		if (Pos.x > (0.94))///Parameters[0])*(Parameters[0]))
		   Pos.x -= 1/Parameters[0];
		//=================================   
	return Pos;
}

vec4 GetPixelPtr(vec2 PixelPos)//, vec4 Ptr)
{
	vec4 Ptr;
	vec2 PosTemp;
	
	//clean position, cause of using floats
	//PosTemp.x = (floor(Parameters[0] * PixelPos.x))/Parameters[0];
	//PosTemp.y = (floor(Parameters[1] * PixelPos.y))/Parameters[1];
	Ptr = GETTEX(BaseImage, PosTemp.xy);
	return Ptr;
}
*/

vec2 GetPixelPos(vec2 PixelPtr)//, vec2 Pos)
{
	vec2 Pos;

		//Pos.x= PixelPtr.r;//*Parameters[0];
		Pos.x= floor(PixelPtr.x*Parameters[0])/Parameters[0];
		//Pos.y= PixelPtr.g;//*Parameters[1];
		Pos.y= floor(PixelPtr.y*Parameters[1]+0.)/Parameters[1];

		//correct error...
		if (Pos.y > (0.94))///Parameters[1])*(Parameters[1]))
		   Pos.y -= 1/Parameters[1];
		  
		if (Pos.x > (0.94))///Parameters[0])*(Parameters[0]))
		   Pos.x -= 1/Parameters[0];
		//=================================   
	return Pos;
}

vec4 GetPixelPtr(vec2 PixelPos)//, vec4 Ptr)
{
	vec4 Ptr;
	vec2 PosTemp;
	
	//clean position, cause of using floats
	//PosTemp.x = (floor(Parameters[0] * PixelPos.x))/Parameters[0];
	//PosTemp.y = (floor(Parameters[1] * PixelPos.y))/Parameters[1];
	Ptr = GETTEX(BaseImage, PosTemp.xy).bgra;
	return Ptr;
}

bool IsMax(vec2 Color, vec2 ColorNext)
{
 	 float Temp, TempNext;
	 Temp = Color.x*Parameters[0]+Color.y;
 	 TempNext = ColorNext.x*Parameters[0]+ColorNext.y;

	 if (TempNext>Temp)
	 	return true;
	 else
	 	 return false;
}


bool IsMin(vec2 Color, vec2 ColorNext)
{
 	 float Temp, TempNext;
	 Temp = Color.x*Parameters[0]+Color.y;
 	 TempNext = ColorNext.x*Parameters[0]+ColorNext.y;

	 if (TempNext<Temp)
	 	return true;
	 else
	 	 return false;
}

void main(void)
{
	int i,j;	
	float sizex = Parameters[0];
	float sizey = Parameters[1];
	
    vec4 color = GETTEX(BaseImage, gl_TexCoord[0].st);
    vec4 colorNext;
    float offset;

	vec2 PPPtr;
	float r,g,b;
	r=0.;
	g=0.;
	b=0.;
	PPPtr.x=0.;
	PPPtr.y=0.;
		
	// Pixels are tagged
//	float offset = PP.y*Parameters[0]+PP.x;
//	float value = PP.y/256.;	
//	PPPtr = GetPixelPos(color, PPPtr);

	vec2 PP = gl_TexCoord[0].st;
 	PP.x = floor(PP.x*Parameters[0])/Parameters[0];
 	PP.y = floor(PP.y*Parameters[1]+0.)/Parameters[1];
	
	PPPtr = GetPixelPos(color.bg);//, PPPtr);

if(color.bgr == vec3(0.))  
			discard;
else
{
		 vec2 LoopOffset;
		
		//control if pixel is a border pixel to avoir looking at the wrong place in image...(ex: next line, end of line...)
		if ((PP.y) < ((sizey-1)/sizey) && (PP.y) > ((1)/sizey))
		   LoopOffset.y = -1;
		else if (PP.y > ((sizey-1)/sizey))
		   LoopOffset.y = -2;
		else if (PP.y < (1/sizey))
		   LoopOffset.y = +1;

		if ((PP.x) < ((sizex-1)/sizex) && (PP.x) > ((1)/sizex))
		   LoopOffset.x = -1;
		else if (PP.x > ((sizex-1)/sizex))
		   LoopOffset.y = -2;
		else if (PP.x < (1/sizex))
		   LoopOffset.x = +1;
		   

		PP = gl_TexCoord[0].st + vec2(LoopOffset.x/sizex,LoopOffset.y/sizey);
		bool NewValue = false;
		//look for max arround the pixel
		//if (PP.y > j/sizey)
		for(i=0;i<3;i++)
			if ((PP.y + i/sizey) < ((sizey-1)/sizey))     
				for(j=0;j<3;j++)
					if ((PP.x + j/sizex) < ((sizex-1)/sizex))
					{
					 colorNext = GETTEX(BaseImage, PP+vec2(j/sizex,i/sizey)).bgra;
					  if(IsMax(color.bg, colorNext.bg))
					  {
					       color.bg = colorNext.bg;
					       NewValue=true;
					  }
					}
	if (NewValue)
	{
		color.ra = GETTEX(BaseImage, PP).ra;//save old value of pixel
		gl_FragColor = color;
	}
	else discard;
}

}


