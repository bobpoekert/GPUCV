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
//---------------------CVG_METASHADEREnd// laplacian.fs
//
// Laplacian edge detection
#define APERTURE_SIZE_1 0
#define APERTURE_SIZE_3 0
#define APERTURE_SIZE	3
#if (APERTURE_SIZE_1)
	#define KERNEL_SIZE 3
#else
	#define KERNEL_SIZE APERTURE_SIZE
#endif
IMGTYPE BaseImage;
uniform float Parameters[KERNEL_SIZE*KERNEL_SIZE*2];


void main(void)
{
    vec4 sample[9];
	vec2 Offset;

#if APERTURE_SIZE_1
//   0  1  0
//   1 -4  1
//   0  1  0

	sample[1]=texture2D(BaseImage, gl_TexCoord[0].st + vec2(Parameters[1*2], Parameters[1*2+1]));
	sample[3]=texture2D(BaseImage, gl_TexCoord[0].st + vec2(Parameters[3*2], Parameters[3*2+1]));
	sample[4]=texture2D(BaseImage, gl_TexCoord[0].st + vec2(Parameters[4*2], Parameters[4*2+1]));
	sample[5]=texture2D(BaseImage, gl_TexCoord[0].st + vec2(Parameters[5*2], Parameters[5*2+1]));
	sample[7]=texture2D(BaseImage, gl_TexCoord[0].st + vec2(Parameters[7*2], Parameters[7*2+1]));
	gl_FragColor = (sample[1] + sample[3] - 4.*sample[4] + sample[5] + sample[7]);
#else if APERTURE_SIZE_3
//   -1 -1 -1
//   -1  8 -1
//   -1 -1 -1
	for (int i = 0; i < 9; i++)
    {
		Offset.x=Parameters[i*2];
		Offset.y=Parameters[i*2+1];
        sample[i] = texture2D(BaseImage, 
                              gl_TexCoord[0].st + Offset);
    }
    gl_FragColor = (sample[4] * 8.0) - 
                    (sample[0] + sample[1] + sample[2] + 
                     sample[3] + sample[5] + 
                     sample[6] + sample[7] + sample[8]);
#endif
}
