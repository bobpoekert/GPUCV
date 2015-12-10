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

#version 120
#extension ARB_draw_buffers : require
#extension EXT_gpu_shader4 : require
#extension EXT_geometry_shader4 : require

void main() 
{
	//Route to Layer 0
	for (int i = 0; i < gl_VerticesIn; i++) 
	{
		// You will recieve 3 positions since we set the input type to Triangles
		gl_Position = gl_PositionIn[i];	
		gl_Position.y -= 1.;
		gl_FrontColor = vec4(1.0, 1.0, 1.0,1.0);
	//	gl_Layer = 0;
		EmitVertex();
	}
	EndPrimitive();

	//Route to Layer 1
	for (int i = 0; i < gl_VerticesIn; i++) 
	{
		gl_Position = gl_PositionIn[i];
		//Just to see a difference in Layer 1
		gl_Position.xy = -gl_Position.xy;
		gl_FrontColor = vec4(0.0, 1.0, 0.0,1.0);
	//	gl_Layer = 1;
		EmitVertex();
	}
	EndPrimitive();
}
