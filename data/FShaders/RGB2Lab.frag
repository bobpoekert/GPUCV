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

float F(const in float t)
{
   /*if (t > .008856)*/ return pow(t,.333333);
   //else return 7.787*t+0.137931;
}

void main(void)
{
/*
|X|   |0.433910  0.376220  0.189860| |R/255|
|Y| = |0.212649  0.715169  0.072182|*|G/255|
|Z|   |0.017756  0.109478  0.872915| |B/255|

L = 116*Y1/3      for Y>0.008856
L = 903.3*Y      for Y<=0.008856

a = 500*(f(X)-f(Y))
b = 200*(f(Y)-f(Z))
where f(t)=t1/3              for t>0.008856
      f(t)=7.787*t+16/116   for t<=0.008856
*/   

    // colors in textures are not set like IplImage ones, B & R are invert

    mat3 matrix = mat3(0.433910, 0.212649, 0.017756,
                       0.376220, 0.715169, 0.109478,
                       0.189860, 0.072182, 0.872915);

    vec3 color = (matrix * GETTEX(BaseImage, gl_TexCoord[0].st).bgr);

    float L, A, B;
    float Y = pow(color.y, .3333333);

    if (color.y > .008856) L = 116.  * Y - 16.;
    else                   L = 903.3 * color.y;

    L/=100.; // luminance : %
    A = (2.083333*(pow(color.x, .3333333)-Y))+.5; // -120 to +120
    B = (.8333333*(Y-pow(color.z, .3333333)))+.5;

    gl_FragColor = vec4(L, A, B, 1.).bgra;
}


