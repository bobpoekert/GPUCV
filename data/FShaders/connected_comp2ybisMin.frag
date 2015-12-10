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

vec2 GetPixelPos(vec4 PixelPtr)//, vec2 Pos)
{
	vec2 Pos;

		Pos.x= PixelPtr.r;//*Parameters[0];
		Pos.x= floor(Pos.x*Parameters[0])/Parameters[0];
		Pos.y= PixelPtr.g;//*Parameters[1];
		Pos.y= floor(Pos.y*Parameters[1]+0.)/Parameters[1];

		//correct error...
		if (Pos.y > (0.94))///Parameters[1])*(Parameters[1]))
		   Pos.y -= 1/Parameters[1];
		  
		if (Pos.x > (0.94))///Parameters[0])*(Parameters[0]))
		   Pos.x -= 1/Parameters[0];
		//=================================   
	return Pos;
}

/*
vec2 GetPixelPos(vec4 PixelPtr)//, vec2 Pos)
{
	vec2 Pos;

		Pos.x= PixelPtr.b;//*Parameters[0];
		Pos.x= floor(Pos.x*Parameters[0])/Parameters[0];
		Pos.y= PixelPtr.g;//*Parameters[1];
		Pos.y= floor(Pos.y*Parameters[1]+0.)/Parameters[1];

		//correct error...
		if (Pos.y > (0.94))///Parameters[1])*(Parameters[1]))
		   Pos.y -= 1/Parameters[1];
		  
		if (Pos.x > (0.94))///Parameters[0])*(Parameters[0]))
		   Pos.x -= 1/Parameters[0];
		//=================================   
	return Pos;
}
*/
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
	 Temp = Color.y*Parameters[0]+Color.x;
 	 TempNext = ColorNext.y*Parameters[0]+ColorNext.x;

	 if (TempNext>Temp)
	 	return true;
	 else
	 	 return false;
}


bool IsMin(vec2 Color, vec2 ColorNext)
{
 /*	 
 	 float Temp, TempNext;
	 
	 Temp = Color.y*Parameters[0]+Color.x;
 	 TempNext = ColorNext.y*Parameters[0]+ColorNext.x;

	 if (TempNext==0.)
	 	return false;
	 else if (TempNext<Temp)
	 	return true;
	 else
	 	 return false;
*/
return false;	
}

void main(void)
{
	int i,j;	
	float sizex = Parameters[0];
	float sizey = Parameters[1];
	
//    vec4 color = GETTEX(BaseImage, gl_TexCoord[0].st).bgra;
    vec4 color = GETTEX(BaseImage, gl_TexCoord[0].st);//.bgra;
    vec4 colorNext;

	vec2 PPPtr;
	float r,g,b;
	r=0.;
	g=0.;
	b=0.;
	PPPtr.x=0.;
	PPPtr.y=0.;
		
	vec2 PP = gl_TexCoord[0].st;
 	PP.x = floor(PP.x*Parameters[0])/Parameters[0];
 	PP.y = floor(PP.y*Parameters[1]+0.)/Parameters[1];
	
	PPPtr = GetPixelPos(color);

if(color.bgr == vec3(0.))  
			discard;
#if 0	
	else
	{
	 /*	PPPtr.x= color.b;//*Parameters[0];
		PPPtr.x= floor(PPPtr.x*Parameters[0])/Parameters[0];
		PPPtr.y= color.g;//*Parameters[1];
		PPPtr.y= floor(PPPtr.y*Parameters[1]+0.)/Parameters[1];

		//correct error...
		if (PPPtr.y > (1/Parameters[1])*(Parameters[1]*0.94))
		   PPPtr.y -= 1/Parameters[1];
		  
		if (PPPtr.x > (1/Parameters[0])*(Parameters[0]*0.94))
		   PPPtr.x -= 1/Parameters[0];
		//=================================   
		*/
		
		color = GETTEX(BaseImage, PPPtr).bgra;//GetPixelPtr(PPPtr, color);
		
		PPPtr = GetPixelPos(color);//, PPPtr);
		//colorNext = GETTEX(BaseImage, PPPtr);

		//test is pointer are equal
		if ((color.b ==0) || (color.g==0))
		   gl_FragColor = vec4(0.,1.,1.,0.);
		else if ((color.b == GETTEX(BaseImage, gl_TexCoord[0].st).b) && (color.g == GETTEX(BaseImage, gl_TexCoord[0].st).g))
		   gl_FragColor = vec4(1.);  
		else if (PPPtr.xy == PP.xy)
	   	   gl_FragColor = vec4(1.);
		else if (PPPtr.x==0)
			  gl_FragColor = vec4(1.,1.,0.,0.); //discard;//
		else if (PPPtr.y==0)
			   gl_FragColor = vec4(0.,1.,1.,0.);
	   	else
		{
		 	if (PPPtr.y > PP.y)
				g= PPPtr.y  - PP.y;
			else
				r= PP.y - PPPtr.y ;
				
			if (PPPtr.x > PP.x)
				b= PPPtr.x  - PP.x;
			else
				b= PP.x - PPPtr.x ;	
		/*	
			if (b<= 1./Parameters[1])
			   b=0;
			if (g<= 1./Parameters[0])
			   g=0;
			if (r<= 1./Parameters[0])
   			   r=0;
		*/
				b*=128.;
				g*=128.;
				r*=128.;

				gl_FragColor = vec4(b,g,r,0);
		}
	}
#else
	else /*if (PPPtr.xy==PP.xy)*/
	{	
		//if (color.a>0) //(PP.x > 250/256) && (PP.y > 250/256))
		//{
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
		
		//look for max arround the pixel
		//if (PP.y > j/sizey)
	
 	    for(i=0;i<3;i++)
			if ((PP.y + i/sizey) < ((sizey-1)/sizey))     
				for(j=0;j<3;j++)
					if ((PP.x + j/sizex) < ((sizex-1)/sizex))
					{
					  //colorNext.rgba= GETTEX(BaseImage, PP+vec2(j/sizex,i/sizey)).rgba;
					  colorNext= GETTEX(BaseImage, PP+vec2(j/sizex,i/sizey));
					  if(IsMax(color.rg, colorNext.rg))color.rg = colorNext.rg;
				//	  if(IsMin(color.ra, colorNext.ra))color.ra = colorNext.ra;
					}

		//color.a= color.a - 0.1;
		//gl_FragColor = color;
		//}
		PP = gl_TexCoord[0].st;
		PPPtr = GetPixelPos(color);
  		
		colorNext.bgra = GETTEX(BaseImage, PPPtr).bgra;
		
	/*	if(colorNext.bgra == vec4(0.))
		{
		//	color=vec4(0.,0.,1.,0.);
		}
		else if (PP != PPPtr)
		{
		*/
			//pixel is not the first max pixel of the component
			//looking into the PPTR pixel
				for(i=0;i<4;i++)
				{
						 if(color==colorNext) i= 1000;
						 
						 if (IsMax(color.rg, colorNext.rg)) color = colorNext;

					//	 if (IsMin(color.ra, colorNext.ra)) color.ra = colorNext.ra;
						
						PPPtr = GetPixelPos(color);//, PPPtr);					
						colorNext = GETTEX(BaseImage, PPPtr);
				}
	
	/*
		}
		else
		{
		 //color = vec4(1.,0,0,0.);
		}
*/	     
	//show vector range
/*	if (PPPtr.x < PP.x)
	    color.b = PP.x - PPPtr.x;
    else
		color.b = PPPtr.x - PP.x;
	
   	color.b*= 4;
*/
	gl_FragColor = color;

	}
//#endif //2
#endif
}


