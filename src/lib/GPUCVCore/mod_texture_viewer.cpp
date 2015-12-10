//CVG_LicenseBegin==============================================================
//
//	Copyright@ Institut TELECOM 2005
//		http://www.institut-telecom.fr/en_accueil.html
//	
//	This software is a GPU accelerated library for computer-vision. It 
//	supports an OPENCV-like extensible interface for easily porting OPENCV 
//	applications.
//	
//	Contacts :
//		patrick.horain@it-sudparis.eu
//		gpucv-developers@picoforge.int-evry.fr
//	
//	Project's Home Page :
//		https://picoforge.int-evry.fr/cgi-bin/twiki/view/Gpucv/Web/WebHome
//	
//	This software is governed by the CeCILL-B license under French law and
//	abiding by the rules of distribution of free software.  You can  use, 
//	modify and/ or redistribute the software under the terms of the CeCILL-B
//	license as circulated by CEA, CNRS and INRIA at the following URL
//	"http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html". 
//	
//================================================================CVG_LicenseEnd

#include "StdAfx.h"
#include <GPUCVCore/GpuTextureManager.h>
#include <GPUCVCore/GpuFilter.h>
#include <GPUCVCore/coretools.h>
bool	Visible = true;

namespace GCV{
#if 1
////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////


#if _GPUCV_GL_USE_GLUT
void ViewerWindowDisplay()
{
	glDrawBuffer(GL_BACK);
	InitGLView(0,GetGpuCVSettings()->GetWindowSize()[0], 0, GetGpuCVSettings()->GetWindowSize()[1]);
	glDisable(GL_LIGHTING);

	DataContainer * CurTex = NULL;

	size_t TextureNbr = GetTextureManager()->GetCount();
	float Pos2D[2]={-1+1./TextureNbr, -1+1./TextureNbr};
	float Scale2D[2] = { 
		1./TextureNbr,
		1./TextureNbr
	};

	glColor4f(1.,1.,1.,1.);
	TextureManager::iterator itTex=TextureManager::GetSingleton()->GetFirstIter();
	for (; itTex!= TextureManager::GetSingleton()->GetLastIter( ) ;itTex++ )
	{
		//get texture
		CurTex = (*itTex).second;
		if(!CurTex)
			continue;

		//draw texture
		CurTex->GetDataDsc<DataDsc_GLTex>()->DrawFullQuad(Pos2D[0], Pos2D[1], Scale2D[0], Scale2D[1], GetGpuCVSettings()->GetWindowSize()[0], GetGpuCVSettings()->GetWindowSize()[1]);

		Pos2D[1] += 1./TextureNbr*2;
	}
	glutSwapBuffers();
}

void ViewerWindowIdle()
{
	if(Visible == true)
		glutPostRedisplay();
}
#endif
////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void
ViewerWindowKeyboard( unsigned char key, int x, int y) {

	switch( key) {
	case( 27) :
		exit( 0);
		break;
	}
}

void ViewerWindowReshape(int w, int h)
{
	GetGpuCVSettings()->SetWindowSize(w, h);
}



void DestroyViewerWindow()
{

}

#if 0
void CreateViewerWindow(std::string _name ,unsigned int _width/* = 512 */, unsigned int _height/* = 512*/)
{
	//glutInitWindowSize(_width, _height);
	ViewerWindowReshape(_width, _height);
	//glutCreateWindow(_name.data());
	glutDisplayFunc(ViewerWindowDisplay);
	glutKeyboardFunc(ViewerWindowKeyboard);
	glutReshapeFunc(ViewerWindowReshape);
	glutIdleFunc(ViewerWindowIdle);
	Visible = true;
	glutMainLoop();
}
#endif
}//namespace GCV
#endif
