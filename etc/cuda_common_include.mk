
#define GPUCV option depending on architecture
# Debug/release configuration
config?=debug32

ifeq ($(config), release32)
	dbg=0
	ARCH=x32
endif
ifeq ($(config), release64)
	dbg=0
	ARCH=x64
endif
ifeq ($(config), debug32)
	dbg=1
	ARCH=x32
endif
ifeq ($(config), debug64)
	dbg=1
	ARCH=x64
endif


GCV_BIN_SUFFIXE?=64
DEBUG_SUFFIXE=
ifeq ($(dbg),1)
	DEBUG_SUFFIXE=D
endif

#default
export TARGET_ARCH=-m32
 

ifeq ($(ARCH),x64)
 export TARGET_ARCH=-m64
 #GCV_OBJ_DIR=./obj/$(uname)/x64/$(CONFIG)/
 GCV_BIN_SUFFIXE=64$(DEBUG_SUFFIXE)
endif

ifeq ($(ARCH),x32)
 export TARGET_ARCH=-m32
 #GCV_OBJ_DIR=./obj/$(uname)/x32/$(CONFIG)/
 GCV_BIN_SUFFIXE=32$(DEBUG_SUFFIXE)
endif

#export TARGET_ARCH_SUFFIXE=.$(arch)

#CUBIN_ARCH_FLAG:= $(TARGET_ARCH)



GCV_LIB_DIR=./lib/$(uname)/

ROOTDIR ?= ./
SRCDIR ?= ./


#include all source files from the local dir:
CUFILES		?=  $(shell find . -maxdepth 1 -name '*.cu')
CCFILES 	?=  $(shell find . -maxdepth 1 -name '*.cpp' -o -name '*.c')




#defines GPUCV specific includes paths
INCLUDES += -I/usr/local/include\
	-I/usr/include\
	-I$(ROOTDIR)/include\
	-I$(ROOTDIR)/include\
	-I$(ROOTDIR)/src\
	-I$(ROOTDIR)/src/lib\
	-I$(ROOTDIR)/src/plugin\
	-I$(ROOTDIR)/dependencies/include\
	-I$(ROOTDIR)/dependencies/otherlibs/include/gl\
	-I$(ROOTDIR)/dependencies/otherlibs/include/\
	-I$(ROOTDIR)/dependencies/SugoiTools/include/\
	-I$(NVSDKCOMPUTE_ROOT)/C/common/inc/\
	-I$(CUDA_INC_PATH)



#defines GPUCV Specific lib paths
LIB =	-L$(ROOTDIR)/lib/linux-gmake/ \
	-L$(ROOTDIR)/dependencies/otherlibs/linux-gmake/ \
	-L$(ROOTDIR)/dependencies/SugoiTools/lib/linux-gmake/ \
	-L$(NVSDKCOMPUTE_ROOT)/shared/lib/linux/\

ifeq ($(ARCH),x64)
	LIB +=  -L$(NVSDKCOMPUTE_ROOT)/C/common/lib/\
		-L$(NVSDKCOMPUTE_ROOT)/C/common/lib/linux\
		-L$(CUDA_LIB_PATH)/../lib64
	#LIB +=	-lcudpp_x86_64
endif
ifeq ($(ARCH),x32)
	LIB +=  -L$(NVSDKCOMPUTE_ROOT)/C/lib/\
		-L$(NVSDKCOMPUTE_ROOT)/C/common/lib/\
		-L$(CUDA_LIB_PATH) 
	#LIB +=	-lcudpp_i386
endif	

#defines GPUCV Specific lib
LIB += -lGPUCVHardware$(GCV_BIN_SUFFIXE)\
	-lGPUCVTexture$(GCV_BIN_SUFFIXE)\
	-lGPUCVCore$(GCV_BIN_SUFFIXE)\
	-lGPUCV$(GCV_BIN_SUFFIXE)\
	-lcxcoreg$(GCV_BIN_SUFFIXE)\
	-lhighguig$(GCV_BIN_SUFFIXE)

#defines SugoiTools lib
LIB +=	-lSugoiTools$(GCV_BIN_SUFFIXE)\
		-lSugoiTracer$(GCV_BIN_SUFFIXE)



#preprocessor
#[DefineUseCudpp] [DefineUseCufft] [DefineUseOpenGL] [DefineUseDirectX] [DefineUseDoubleImgFormat] [DefineUseAllImgFormat]
#

COMMONFLAGS+= -D_GPUCV_CUDA_SUPPORT_OPENGL -D_SG_TLS_SUPPORT_GL -D_GPUCV_CUDA_SUPPORT_CUFFT 
# -D_GPUCV_CUDA_SUPPORT_CUDPP
#COMMONFLAGS+= -D_GPUCV_CUDA_SUPPORT_DIRECTX
#COMMONFLAGS+= -D_GPUCV_CUDA_SUPPORT_ALL_IMAGE_FORMAT
#COMMONFLAGS+= -D_GPUCV_CUDA_SUPPORT_DOUBLE_IMAGE_FORMAT

#target name
OUTPUT_NAME :=
ifneq ($(SHARED_LIB_TEMP),)
#def SHARED_LIB_TEMP
	SHARED_LIB = lib$(SHARED_LIB_TEMP)$(GCV_BIN_SUFFIXE).so
	OUTPUT_NAME=$(SHARED_LIB_TEMP)
endif
ifneq ($(STATIC_LIB_TEMP),)
	STATIC_LIB = lib$(STATIC_LIB_TEMP)$(GCV_BIN_SUFFIXE).a
	OUTPUT_NAME=$(STATIC_LIB_TEMP)
endif
ifneq ($(EXECUTABLE_TEMP),)
	EXECUTABLE = $(EXECUTABLE_TEMP)$(GCV_BIN_SUFFIXE)
	OUTPUT_NAME=$(EXECUTABLE_TEMP)
endif

#define Specific Compiler options
#deprecated: NVCCFLAGS := --host-compilation=c++
ROOTOBJDIR ?= $(ROOTDIR)/$(CONFIG)/$(arch)/linux/$(OUTPUT_NAME)


