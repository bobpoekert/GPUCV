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
//---------------------CVG_METASHADEREnd// This shader will mutiply matrix A with matrix B and store result in matrix C.
// A, B and C are all packed Parameter[2] elements per texture pixel.

//CVG_METASHADERBegin
#define GETTEX //function to get texture
#define IMGTYPE //texture type
#define CHANNELS_PER_PIXEL 4

#define DEF_CMPNT_1 1//
#define DEF_CMPNT_2 2//
#define DEF_CMPNT_3 3//
#define DEF_CMPNT_4 4//default is 4, but some images can have 3 channels(RGB/BGR) so we need to process both cases.

//nbr 1 and 2 can be processed by nbr 4

#if DEF_CMPNT_4
	#define SWIZZLE_ORDER bgra
	#define REVERSE_SWIZZLE_ORDER rgba
	#define DEF_CMPNT_NBR DEF_CMPNT_4
	#define DEF_VEC_TYPE vec4
#elif DEF_CMPNT_3
	#define SWIZZLE_ORDER bgr
	#define REVERSE_SWIZZLE_ORDER rgb
	#define DEF_CMPNT_NBR DEF_CMPNT_3
	#define DEF_VEC_TYPE vec3
#elif DEF_CMPNT_2
	#define SWIZZLE_ORDER xy
	#define REVERSE_SWIZZLE_ORDER yx
	#define DEF_CMPNT_NBR DEF_CMPNT_2
	#define DEF_VEC_TYPE vec2
#else
	#define SWIZZLE_ORDER b
	#define REVERSE_SWIZZLE_ORDER b
	#define DEF_CMPNT_NBR DEF_CMPNT_1
	#define DEF_VEC_TYPE float
#endif
//---------------------CVG_METASHADEREnd
IMGTYPE BaseImage; // source image
IMGTYPE Image0;    // second image

#define DEF_ADD 	0//is set to 1, shader also proceed to addition

#if DEF_ADD 
	IMGTYPE Image1;				// image to addition
	uniform float Parameters[6];	//parameter 6 is addition factor
	#define DEF_ADD_OPERATOR + GETTEX(Image1,gl_TexCoord[0].st).REVERSE_SWIZZLE_ORDER*Parameters[5]	//macro to place addition in last operation
#else
	uniform float Parameters[5];	//description of the matrix
	#define DEF_ADD_OPERATOR //nothing...
#endif

/*Parameters
	0 = nb of cols in A  //= nb of rows in B/nb of channels in A and B (Parameters[2] )
	1 = nb of rows in B
	//2 =  nb of channels in A and B, ie number of elements packed in each pixel (for better utilisation, it should be 4)
	3 =  nb of rows in A
	4 = nb of cols in B
	5 = alpha coef value used for multiply
	6 =  optional beta coef value used for addition
*/
void main(void)
{
	vec2 workingSize 	= vec2(Parameters[0], Parameters[1]);	
	// increment by Parameter[2] because we can utilize (Parameter[2]) pixels,  at a time in the second matrix
	
	//default working step.
	vec2 workingStep 	= vec2(1./workingSize.x,  1./workingSize.y);
	
	//starting position has an offset of workingStep.y *3./4.
	//vec2 workingPos 	=  vec2(1., workingStep.y *6./8. );//+ workingStep.y - workingStep.y * 1./4.);
	
	//starting pos.
	vec2 workingPos 	= vec2(workingStep.x*DEF_CMPNT_NBR/2., workingStep.y/2.);//vec2(1., workingStep.y *6./8. );//+ workingStep.y - workingStep.y * 1./4.);
	vec2 pixelPos		= gl_TexCoord[0].st;
	
#if 0	 //debug
		gl_FragColor = vec4(workingSize, workingStep).SWIZZLE_ORDER;	
		//gl_FragColor = GETTEX(BaseImage,	gl_TexCoord[0].st).SWIZZLE_ORDER;
		//gl_FragColor = vec4(Parameters[0],Parameters[1],Parameters[2],Parameters[3]).SWIZZLE_ORDER;
		//gl_FragColor = vec4(Parameters[4],Parameters[5],0., 0.).rgba;	
		//gl_FragColor = vec4(pixelPos, workingPos).SWIZZLE_ORDER;
#else //RELEASE	
	DEF_VEC_TYPE colorC=DEF_VEC_TYPE(0.);// Result of the matrix multiplication
	vec2 posA,posB1,posB2,posB3,posB4;// positions for various pixels in the corresponding texture maps
	DEF_VEC_TYPE colorA,colorB1,colorB2,colorB3,colorB4;// Actual texture values in the texture maps
	
	
#if DEF_CMPNT_2
	float step1 = workingStep.y;
#endif
#if DEF_CMPNT_3
	float step2 = workingStep.y*2;
#endif
#if DEF_CMPNT_4
	float step3 = workingStep.y*3;
#endif
	workingStep *=DEF_CMPNT_NBR;
	
	
	while(workingPos.x<1.0 && workingPos.y<1.0)
	{
		//pixelPos is fixed.
		//posA move from x[0..1]
		//posBx move from y[0..1]
		posA=vec2(workingPos.x,pixelPos.y);					// Position in matrix A of (k*Parameter[0])th pixel in row gl_TexCoord[0].t
		colorA=GETTEX(BaseImage,	posA).SWIZZLE_ORDER;// This gets the texture values from First Matrix 
		
		posB1=vec2(pixelPos.x,	workingPos.y);	// Position in matrix B of (j*Parameter[1])th pixel in coloumn gl_TexCoord[0].s
		colorB1=GETTEX(Image0,		posB1).SWIZZLE_ORDER;// Texture values from the first pixel in the Second Matrix
#if DEF_CMPNT_2
		posB2=vec2(pixelPos.x,	workingPos.y+step1);	// Position in matrix B of [(j*Parameter[1])-1]th pixel in coloumn gl_TexCoord[0].s
		colorB2=GETTEX(Image0,		posB2).SWIZZLE_ORDER;// Texture values from the second pixel, ie the one just below the first pixel in the Second Matrix
#endif
#if DEF_CMPNT_3
		posB3=vec2(pixelPos.x,	workingPos.y+step2);	// Position in matrix B of [(j*Parameter[1])-2]th pixel in coloumn gl_TexCoord[0].s
		colorB3=GETTEX(Image0,		posB3).SWIZZLE_ORDER;// Texture values from the third pixel, ie the one 2 pixels below the first pixel in the Second Matrix
#endif
#if DEF_CMPNT_4
		posB4=vec2(pixelPos.x,	workingPos.y+step3);	// Position in matrix B of [(j*Parameter[1])-3]th pixel in coloumn gl_TexCoord[0].s
		colorB4=GETTEX(Image0,		posB4).SWIZZLE_ORDER;// Texture values from the fourth pixel , ie the one 3 pixels below the first pixel in the Second Matrix
#endif
		
//#if DEF_CMPNT_4
//		colorB4=GETTEX(Image0,		posB4).SWIZZLE_ORDER;// Texture values from the fourth pixel , ie the one 3 pixels below the first pixel in the Second Matrix
//#endif
		// Be careful and choose wisely which of the colorC's to pick
#if 0//DEBUG		
		colorC+=	  colorC*10000.+
					+ colorB1.SWIZZLE_ORDER *1.
					+ colorB2.SWIZZLE_ORDER*10.
					+ colorB3.SWIZZLE_ORDER*100.
					+ colorB4.SWIZZLE_ORDER*1000.;
#else
	#if DEF_CMPNT_4
		colorC+=	  colorA.b*colorB1.SWIZZLE_ORDER
					+ colorA.g*colorB2.SWIZZLE_ORDER
					+ colorA.r*colorB3.SWIZZLE_ORDER
					+ colorA.a*colorB4.SWIZZLE_ORDER;
	#elif DEF_CMPNT_3
		colorC+=	  colorA.b*colorB1.SWIZZLE_ORDER
					+ colorA.g*colorB2.SWIZZLE_ORDER
					+ colorA.r*colorB3.SWIZZLE_ORDER;
	#elif DEF_CMPNT_2
		colorC+=	  colorA.x*colorB1.SWIZZLE_ORDER
					+ colorA.y*colorB2.SWIZZLE_ORDER;
	#else 
		colorC+=	  colorA*colorB1;
	#endif
#endif
		// I chose rgba, but thou ought to know better!!! 
		//colorC+=colorA.bbbb*colorB1.rgba+colorA.gggg*colorB2.rgba+colorA.rrrr*colorB3.rgba+colorA.aaaa*colorB4.rgba;
		//colorC+=colorA.bbbb*colorB1.bgra+colorA.gggg*colorB2.bgra+colorA.rrrr*colorB3.bgra+colorA.aaaa*colorB4.bgra;
		//colorC+=colorA.rrrr*colorB1.bgra+colorA.gggg*colorB2.bgra+colorA.bbbb*colorB3.bgra+colorA.aaaa*colorB4.bgra;
		
		// Update to fetch next column in the same row
		// Update to get next row in the same column,
		workingPos += workingStep;
	}
	#if DEF_CMPNT_4
		gl_FragColor = colorC.SWIZZLE_ORDER*Parameters[4] DEF_ADD_OPERATOR;
	#elif DEF_CMPNT_3
		gl_FragColor = vec4(colorC.SWIZZLE_ORDER*Parameters[4] DEF_ADD_OPERATOR, 0.);
	#elif DEF_CMPNT_2
		gl_FragColor = vec4(colorC.SWIZZLE_ORDER*Parameters[4] DEF_ADD_OPERATOR, 0., 0.);
	#else
		gl_FragColor = vec4(colorC*Parameters[4] DEF_ADD_OPERATOR,0., 0., 0.);
	#endif
#endif
}
