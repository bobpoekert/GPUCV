// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#ifndef _WIN32_WINNT		// Allow use of features specific to Windows XP or later.                   
#define _WIN32_WINNT 0x0501	// Change this to the appropriate value to target other versions of Windows.
#endif						

#include <stdio.h>
#ifdef _WINDOWS
	#include <tchar.h>
#endif


// TODO: reference additional headers your program requires here
#include <SugoiTools/config.h>
#include <SugoiTools/logger_main.h>
#include <SugoiTools/cl_base_obj.h>
#include <SugoiTools/cl_template_manager.h>
#include <SugoiTools/tools.h>
#include <SugoiTools/exceptions.h>
#include <stdlib.h>//must be include before GL to avoid EXIT redefinition
#include <SugoiTools/sg_gl.h>

#include <stdio.h>
#ifdef MACOS
	#include <malloc/malloc.h>
#else
	#include <malloc.h>
#endif
#include <fcntl.h>


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
	#include <cstring>
#endif
