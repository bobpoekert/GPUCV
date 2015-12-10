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
//---------------------CVG_METASHADEREnd// orignal version come from SuperBible book
// sobel.fs
//
// Sobel edge detection


// Sobel Fragment Shader
#define SOBEL_X_ORDER 1
#define SOBEL_Y_ORDER 0
#define SOBEL_APERTURE_SIZE 3
#if SOBEL_APERTURE_SIZE==-1
	#define KERNEL_SIZE 3
#else
	#define KERNEL_SIZE SOBEL_APERTURE_SIZE
#endif


IMGTYPE BaseImage;
uniform float Parameters[KERNEL_SIZE*KERNEL_SIZE*2];
vec4 sample[KERNEL_SIZE*KERNEL_SIZE];


#if SOBEL_APERTURE_SIZE==3
	vec4 CalculateSobel()
	{
	//    -1 -2 -1       1 0 -1 
	// H = 0  0  0   V = 2 0 -2
	//     1  2  1       1 0 -1
	//
	// result = sqrt(H^2 + V^2)
	#if SOBEL_X_ORDER == 1 
		vec4 vertEdge = sample[0] + (2.0*sample[1]) + sample[2] -
						(sample[6] + (2.0*sample[7]) + sample[8]);
	#else
		vec4 vertEdge = vec4(0.);
	#endif
	
	
	#if SOBEL_Y_ORDER == 1
		vec4 horizEdge = sample[2] + (2.0*sample[5]) + sample[8] -
						 (sample[0] + (2.0*sample[3]) + sample[6]);
	#else
		vec4 horizEdge = vec4(0.);
	#endif

		return sqrt((horizEdge * horizEdge) + (vertEdge * vertEdge));
	}
#else if SOBEL_APERTURE_SIZE==-1
	vec4 CalculateSobel()
	{
	//    -3 -10 -3      -3  0 3 
	// H = 0  0  0   V = -10 0 10
	//     3  10 3       -3  0 3
	//
	#if SOBEL_X_ORDER == 1 
		vec4 vertEdge = -(3.*sample[0] + (10.0*sample[1]) + 3.*sample[2])
						+(3.*sample[6] + (10.0*sample[7]) + 3.*sample[8]);
	#else
		vec4 vertEdge = vec4(0.);
	#endif
	
	
	#if SOBEL_Y_ORDER == 1
		vec4 horizEdge =  3.*sample[2] + (10.0*sample[5]) + 3.*sample[8] -
						 (3.*sample[0] + (10.0*sample[3]) + 3.*sample[6]);
	#else
		vec4 horizEdge = vec4(0.);
	#endif
	
		return sqrt((horizEdge * horizEdge) + (vertEdge * vertEdge));
	}
#endif




void main(void)
{
   // vec4 sample[KERNEL_SIZE*KERNEL_SIZE];
	vec2 Offset;

    for (int i = 0; i < 9; i++)
    {
		Offset.x=Parameters[i*2];
		Offset.y=Parameters[i*2+1];
        sample[i] = GETTEX(BaseImage, gl_TexCoord[0].st + Offset);
    }
    
	vec4 Result = CalculateSobel();
	gl_FragColor = vec4(Result.rgb*10.,1.);
}
