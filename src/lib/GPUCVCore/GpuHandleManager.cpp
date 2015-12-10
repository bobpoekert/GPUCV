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
#include <GPUCVCore/GpuHandleManager.h>

namespace GCV{

//=================================================
GpuHandleManager :: CvgHandle :: CvgHandle(GLhandleARB sh, GLhandleARB vh, GLhandleARB fh, ShaderObject * so)
:shader_handle(sh),
vertex_handle(vh),
fragment_handle(fh),
linked_object(so)
{
}
//=================================================
GpuHandleManager :: CvgHandle :: ~CvgHandle()
{
	if(linked_object)
	{
		linked_object->SetQueueIndex(-1);
		linked_object->SetIsLoaded(false);
	}


	if (shader_handle != 0)
	{//Yann.A. 02/11/05 : Error when exiting the application comes from Here...???
		//Check code to find if shader_handle has already been detached...??
		/*   glDetachObjectARB(shader_handle, vertex_handle);
		glDetachObjectARB(shader_handle, fragment_handle);

		//glDeleteShader(fragment_handle);
		//glDeleteShader(shader_handle);

		glDeleteObjectARB(vertex_handle);
		glDeleteObjectARB(fragment_handle);
		glDeleteObjectARB(shader_handle);
		*/
	}
}
//=================================================
GpuHandleManager :: GpuHandleManager():current_pos(-1)
{
	stack = new CvgHandle*[_GPUCV_SHADER_MAX_NB];
	for (int i=_GPUCV_SHADER_MAX_NB; i--; )
	{
		stack[i] = new CvgHandle(0,0, 0, NULL);
	}
}
//=================================================
GpuHandleManager :: ~GpuHandleManager()
{
	for (int i=_GPUCV_SHADER_MAX_NB; i--; )
	{
		stack[i]->~CvgHandle();
	}
}
//================================================
GLhandleARB GpuHandleManager :: GetHandle(int pos)
{
	if (pos >= 0 && pos < _GPUCV_SHADER_MAX_NB)
	{
		return stack[pos]->shader_handle;
	}
	else
	{
		GPUCV_ERROR("Critical : unable to access shader handle ARB, index outbound.\n");
		return 0;
	}
}
//=================================================
void GpuHandleManager :: cvgPush(GLhandleARB sh, GLhandleARB vh, GLhandleARB fh, ShaderObject * so)
{
	current_pos++;
	if (current_pos == _GPUCV_SHADER_MAX_NB) current_pos = 0;
	stack[current_pos]->~CvgHandle();
	stack[current_pos] = new CvgHandle(sh, vh, fh, so);
	so->SetQueueIndex(current_pos);
	so->SetIsLoaded(true);
}
//=================================================
GLhandleARB GpuHandleManager :: SetFirst(int pos)
{
	if (pos < 0 || pos >= _GPUCV_SHADER_MAX_NB)
	{
		GPUCV_WARNING("Warning : can't set handle position : given one is not correct !\n");
		cerr << "Position = " << pos << endl;
		return 0;
	}
	else
	{
		if (pos != current_pos)
		{
			int nb = current_pos - pos;
			if(nb<0) nb = _GPUCV_SHADER_MAX_NB + nb;

			int next;
			for (int i=1; i<=nb; i++)
			{
				CvgHandle * tmp = stack[pos];
				next = (pos+1 == _GPUCV_SHADER_MAX_NB)?0:pos+1;
				stack[pos] = stack[next];
				stack[next] = tmp;
				stack[pos]->linked_object->SetQueueIndex(pos);
				stack[next]->linked_object->SetQueueIndex(next);

				//pos ++;
				//if (pos == _GPUCV_SHADER_MAX_NB) pos = 0;
				pos = next;
			}
			stack[current_pos]->linked_object->SetQueueIndex(current_pos);
		}
		return stack[current_pos]->shader_handle;
	}
}
//=================================================
void GpuHandleManager :: RemoveHandle(int pos)
{
	if (pos < 0 || pos >= _GPUCV_SHADER_MAX_NB)
	{
		GPUCV_WARNING("Warning : can't remove handle : position is not correct !\n");
		cerr << "Position = " << pos << endl;
	}
	else
	{
		stack[pos]->~CvgHandle();
		stack[pos] = new CvgHandle(0,0, 0, NULL);
	}
}
//=================================================

}//namespace GCV

