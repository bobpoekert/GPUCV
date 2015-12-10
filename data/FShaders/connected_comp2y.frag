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



vec2 GetPixelPos(vec4 PixelPtr)
{
	vec2 Pos;
	int offset  = floor(PixelPtr.b*256.*65536.);	// = floor(offset/(65536.0))/256;
		offset += floor(PixelPtr.g*256.*256.);		// = floor((offset-r*65536.0)/256.0)/256;
		offset += floor(PixelPtr.r * 256.);		//mod(offset,256)/256;
		Pos.y = offset/(Parameters[1]*Parameters[0]*256.);
		Pos.x = (offset - (Parameters[1]*Parameters[0]*256.*Pos.y))/Parameters[0];
		
		Pos.x = (floor(Parameters[0] * Pos.x))/Parameters[0];
		Pos.y = (floor(Parameters[1] * Pos.y))/Parameters[1];
	return Pos;
}

vec4 GetPixelPtr(vec2 PixelPos)
{
	vec4 Ptr;
	vec2 PosTemp;
	
	//clean position, cause of using floats
	PosTemp.x = (floor(Parameters[0] * PixelPos.x))/Parameters[0];
	PosTemp.y = (floor(Parameters[1] * PixelPos.y))/Parameters[1];
	
	return GETTEX(BaseImage, PosTemp.xy);
}

void main(void)
{
	int i,j;	
	float sizex = Parameters[0];
	float sizey = Parameters[1];
	
    vec4 color = GETTEX(BaseImage, gl_TexCoord[0].st);
    vec4 colorNext;
    float offset;
	vec2 PP = gl_TexCoord[0].st ;//- vec2(1./sizex,1./sizey);
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
	
	
PPPtr = GetPixelPos(color);
	

if(color.bgr == vec3(0.))
			discard;
#if 1	
	else
	{
//test pointer precision		
		PPPtr = GetPixelPos(color);
/*		offset  = floor(color.b*256.*65536.);	// = floor(offset/(65536.0))/256;
		offset += floor(color.g*256*256);		// = floor((offset-r*65536.0)/256.0)/256;
		offset += floor(color.r * 256);		//mod(offset,256)/256;
		PPPtr.y = offset/(Parameters[1]*Parameters[0]*256.);
		PPPtr.x = (offset - (Parameters[1]*Parameters[0]*256.*PPPtr.y))/Parameters[0];
		//PPPtr.x = (offset - (Parameters[0]*256.*PPPtr.y));
*/		
	//	if ((PP.x == PPPtr.x) && (PP.y == PPPtr.y))
	//		gl_FragColor = color;
	//	else
		{
			if (PPPtr.y > PP.y)
				b= PPPtr.y - PP.y;
			else
				b= PP.y - PPPtr.y;
				
			if (PPPtr.x > PP.x)
				g= PPPtr.x - PP.x;
			else
				g= PP.x - PPPtr.x;	
			
			b/=Parameters[1];
			g/=Parameters[0];
			r/=Parameters[0];
			
		/*	if ((b<1/Parameters[1]))// && (g<1/Parameters[0]))
			{
				b=0;	
				gl_FragColor = color;
			}
			else
			{
		*/
				b*=128.;
				g*=128.;
				r*=128.;
				gl_FragColor = vec4(b,g,r,0);
		//	}
		}
	}
#else
	else if (PPPtr==PP)
	{	
		//look for max arround the pixel	
		for(i=0;i<3;i++)
			for(j=0;j<3;j++)
				color = max(GETTEX(BaseImage, PP+vec2(j/sizex,i/sizey)),color);

		color.a=0;
	}
	else
	{
		PPPtr = GetPixelPos(color);
		
/*		offset = color.b*256.*65536.;	// = floor(offset/(65536.0))/256;
		offset += color.g*256*256;		// = floor((offset-r*65536.0)/256.0)/256;
		offset += color.r * 256;		//mod(offset,256)/256;
		PPPtr.y = floor(offset/Parameters[0]);
		PPPtr.x = floor(offset - Parameters[0]*PPPtr.y);
		
		if (PPPtr!=PP)
			discard;
*/
  
		colorNext = GetPixelPtr(PPPtr);
		
		if(colorNext.bgra == vec4(0.))
		{
			color=vec4(1.);
		}
		else if (PP != PPPtr)
		{
			//pixel is not the first max pixel of the component
			//looking into the PPTR pixel
			
				for(i=0;i<10;i++)
				{
					/*offset = color.b*256.*65536.;	// = floor(offset/(65536.0))/256;
					offset += color.g*256*256;		// = floor((offset-r*65536.0)/256.0)/256;
					offset += color.r * 256;		//mod(offset,256)/256;
					PPPtr.y = floor(offset/Parameters[0]);
					PPPtr.x = floor(offset - Parameters[0]*PPPtr.y);
					*/
					
						//color = max(colorNext, color);
						color = max(colorNext, color);
						PPPtr = GetPixelPos(color);
						colorNext = GetPixelPtr(PPPtr);
						
						//get ptr pixel coordinates
					/*	offset = floor(color.b*256.*65536.);	// = floor(offset/(65536.0))/256;
						offset += floor(color.g*256*256);		// = floor((offset-r*65536.0)/256.0)/256;
						offset += floor(color.r * 256);		//mod(offset,256)/256;
						PPPtr.y = floor(offset/Parameters[0]);
						PPPtr.x = floor(offset - Parameters[0]*PPPtr.y);
						if(color.bgra == vec4(0.))
						{
							discard;//gl_FragColor=vec4(1.);
						}
					*/
				}
				//colorNext = GETTEX(BaseImage, PPPtr);
				//color = max(colorNext,color);	
		}
		else
		{
		 color = vec4(1.,1.,1.,0.);
		}
	
	gl_FragColor = color;
	}
#endif
}


