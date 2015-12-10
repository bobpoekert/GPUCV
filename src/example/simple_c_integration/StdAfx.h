// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#if 1
#ifndef _WIN32_WINNT		// Allow use of features specific to Windows XP or later.                   
#define _WIN32_WINNT 0x0501	// Change this to the appropriate value to target other versions of Windows.
#endif						

#include <stdio.h>
#ifdef _WINDOWS
	#include <tchar.h>
#endif


// TODO: reference additional headers your program requires here
#include <math.h>
#include <stdio.h>
#include <malloc.h>
#include <fcntl.h>
#include <memory.h>
#include <assert.h>
#include <includecv.h>


#ifdef _WINDOWS
    #include <io.h>
#endif

#ifdef __cplusplus
	#include <typeinfo>
	#include <string>
	#include <iostream>
	#include <vector>
	#include <map>
	#include <sstream>
#endif

#endif