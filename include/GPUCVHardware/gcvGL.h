#include <stdlib.h>//must be include before GL to avoid EXIT redefinition
#include <definitions.h>
#if defined(_MACOS)
	#include <GL/glew.h>
	#include <GL/glxew.h>
	#include <OpenGL/gl.h>
#if _GPUCV_GL_USE_GLUT
	#include <GLUT/glut.h>
#endif
	//#include <AGL/agl.h>
#elif defined(_LINUX)
	#include <GL/glew.h>
    #include <GL/glxew.h>
	
//glxext generate compile error on Ubuntu 9.10, replaced by glx.h and works fine
//!\todo Check where this error come from and if it works fine and other systems: 
//glx generate errors on Fedora 11	
//	#include <GL/glx.h>
	#include <GL/glxext.h>
	#include <GL/gl.h>
#if _GPUCV_GL_USE_GLUT
	#include <GL/glut.h>
#endif	
#elif defined(_WINDOWS)
	#include <GL/glew.h>
	#include <GL/wglew.h>
#if _GPUCV_GL_USE_GLUT
	#include <GL/glut.h>
#endif
	#include <GL/gl.h>
	#include <GL/glu.h>
	#pragma warning (disable : 4786)
	#pragma warning (disable : 4996)
#else
	#error Unknown operating system, do not know how to include OpenGL headers!
#endif