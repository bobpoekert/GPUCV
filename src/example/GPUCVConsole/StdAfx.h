// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#if 1

#include <stdio.h>
#ifdef _WINDOWS
#	ifndef _WIN32_WINNT		// Allow use of features specific to Windows XP or later.                   
#		define _WIN32_WINNT 0x0501	// Change this to the appropriate value to target other versions of Windows.
#	endif						
#	include <tchar.h>
#endif


// TODO: reference additional headers your program requires here
#include <SugoiTools/config.h>
#include <SugoiTools/logger_main.h>
#include <SugoiTools/cl_base_obj.h>
#include <SugoiTools/tools.h>
#include <SugoiTools/cl_pos.h>
#include <SugoiTools/exceptions.h>
#include <SugoiTools/cl_base_obj.h>
#include <SugoiTools/tools.h>
#include <SugoiTools/cl_vector2.h>
#include <SugoiTools/tools.h>
#include <SugoiTools/math.h>
#include <SugoiTools/MultiPlat.h>
#include <SugoiTools/cl_tpl_manager_array.h>
#include <math.h>
#include <stdio.h>
#ifdef _MACOS
#	include <sys/malloc.h>
#else
#	include <malloc.h>
#endif
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