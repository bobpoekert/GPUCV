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
#include "GPUCVHardware/exception.h"
#include "GPUCVHardware/GlobalSettings.h"

namespace GCV{
//============================================================
Exception::Exception(const std::string& _file, int _line, const std::string& _test,const std::string& _message)
:CAssertException(_file, _line, _message)
,m_FunctionName("")
,m_ObjPtr(NULL)
,m_ExceptionTest(_test)
{
}
//============================================================
Exception::Exception(const std::string& _file, int _line, const std::string& _test,const std::string& _message, const std::string& _functionName, void *_obj/*=NULL*/)
:CAssertException(_file, _line, _message)
,m_FunctionName(_functionName)
,m_ObjPtr(_obj)
,m_ExceptionTest(_test)
{
}
//============================================================
/*virtual*/
Exception::~Exception() throw()
{

}
//============================================================
/*virtual*/
const char* Exception::what() const throw()
{
	return "message not defined yet";
}
//============================================================
//============================================================
//============================================================
//============================================================
ExceptionCompat::ExceptionCompat(const std::string& _file, int _line, const std::string& _test,const std::string& _message)
:Exception(_file, _line, _test, _message)
{
}
//============================================================
ExceptionCompat::ExceptionCompat(const std::string& _file, int _line, const std::string& _test, const std::string& _message, const std::string& _functionName, void *_obj/*=NULL*/)
:Exception(_file, _line, _test, _message,_functionName, _obj)
{
}
//============================================================
/*virtual*/
ExceptionCompat::~ExceptionCompat() throw()
{
}
//============================================================
/*virtual*/
const char* ExceptionCompat::what() const throw()
{
	std::string strMsg= "\nExceptionCompat:>\n";
	strMsg+= "Line :";
	strMsg+= m_Line;
	strMsg+= "\nFile : ";
	strMsg+= m_File;
	strMsg+= "\nMsg: ";
	strMsg+= m_OriginalMsg;
	strMsg+= "\n";
	return strMsg.data();	
}
//============================================================
//============================================================
//============================================================
//============================================================
//============================================================
ExceptionTodo::ExceptionTodo(const std::string& _file, int _line, const std::string& _test,const std::string& _message)
:Exception(_file, _line, _test, _message)
{
}
//============================================================
ExceptionTodo::ExceptionTodo(const std::string& _file, int _line, const std::string& _test,const std::string& _message, const std::string& _functionName, void *_obj/*=NULL*/)
:Exception(_file, _line, _test, _message,_functionName, _obj)
{
}
//============================================================
/*virtual*/
ExceptionTodo::~ExceptionTodo() throw()
{
}
//============================================================
/*virtual*/
const char* ExceptionTodo::what() const throw()
{
	std::string strMsg= "\nExceptionTodo:>\n";
	strMsg+= "Line :";
	strMsg+= m_Line;
	strMsg+= "\nFile : ";
	strMsg+= m_File;
	strMsg+= "\nMsg: ";
	strMsg+= m_OriginalMsg;
	strMsg+= "\n";
	return strMsg.data();	
}
//============================================================
}//namespace GCV
