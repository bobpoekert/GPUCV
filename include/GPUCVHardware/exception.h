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
#ifndef __GPUCVHARDWARE_EXCEPTIONS_H
#define __GPUCVHARDWARE_EXCEPTIONS_H

#include <SugoiTools/exceptions.h>
#include <GPUCVHardware/config.h>

namespace GCV{

class _GPUCV_HARDWARE_EXPORT Exception : public SGE::CAssertException
{
public :

    //----------------------------------------------------------
    // Constructeur
    //----------------------------------------------------------
    Exception(const std::string& _file, int _line, const std::string& _test, const std::string& _message);
	Exception(const std::string& _file, int _line, const std::string& _test, const std::string& _message, const std::string& _functionName, void *_obj=NULL);

    virtual ~Exception() throw();

    virtual const char* what() const throw();
protected :
	//! Name of the function that raised the exception 
	_DECLARE_MEMBER(std::string, FunctionName);
	//! Pointer of the object that raised the exception 
	_DECLARE_MEMBER(void*, ObjPtr);
	//! String describing the operation that raised exception 
	_DECLARE_MEMBER(std::string, ExceptionTest);
};

class _GPUCV_HARDWARE_EXPORT ExceptionCompat : public Exception
{
public :

    //----------------------------------------------------------
    // Constructeur
    //----------------------------------------------------------
    ExceptionCompat(const std::string& _file, int _line, const std::string& _test, const std::string& _message);
	ExceptionCompat(const std::string& _file, int _line, const std::string& _test, const std::string& _message, const std::string& _functionName, void *_obj=NULL);
    virtual ~ExceptionCompat() throw();
	virtual const char* what() const throw();
};

class _GPUCV_HARDWARE_EXPORT ExceptionTodo : public Exception
{
public :

    //----------------------------------------------------------
    // Constructeur
    //----------------------------------------------------------
    ExceptionTodo(const std::string& _file, int _line, const std::string& _test, const std::string& _message);
	ExceptionTodo(const std::string& _file, int _line, const std::string& _test, const std::string& _message, const std::string& _functionName, void *_obj=NULL);
    virtual ~ExceptionTodo() throw();
	virtual const char* what() const throw();
};

}//namespace GCV

#endif // __GPUCVHARDWARE_EXCEPTIONS_H
