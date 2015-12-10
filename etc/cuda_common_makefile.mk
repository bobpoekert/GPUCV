################################################################################
#
# Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to NVIDIA ownership rights under U.S. and 
# international Copyright laws.  
#
# NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################
#
# Common build script
#
################################################################################

.SUFFIXES : .cu .cu_dbg.o .c_dbg.o .cpp_dbg.o .cu_rel.o .c_rel.o .cpp_rel.o .cubin

# Add new SM Versions here as devices with new Compute Capability are released
SM_VERSIONS := sm_10 sm_11 sm_12 sm_13

CUDA_INSTALL_PATH := /usr/local/cuda

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))



#------------------------------------
#GPUCV UPDATE:
#Todo: Test CUDA SDK PATH??
#-------------------------------------


# Basic directory setup for SDK
# (override directories only if they are not already defined)
@echo ==== $(CONFIG) mode selected ====
SRCDIR     ?= 
ROOTDIR    ?= ..
ifeq ($(dbg),1)
ROOTOBJDIR = $(ROOTDIR)obj/$(OSLOWER)-gmake/$(ARCH)/Debug/$(OUTPUT_NAME)
@echo ==== Debug mode selected ====
else
ROOTOBJDIR = $(ROOTDIR)obj/$(OSLOWER)-gmake/$(ARCH)/Release/$(OUTPUT_NAME)
@echo ==== Release mode selected ====
endif
ROOTBINDIR ?= $(ROOTDIR)bin/
BINDIR     ?= $(ROOTBINDIR)
LIBDIR     := $(ROOTDIR)lib/$(OSLOWER)-gmake/
COMMONDIR  := $(NVSDKCUDA_ROOT)common/

# Compilers
NVCC       := $(CUDA_INSTALL_PATH)/bin/nvcc 
CXX        := g++
CC         := gcc
LINK       := g++ -fPIC

# Includes
INCLUDES  += -I inc/ -I. -I$(CUDA_INSTALL_PATH)/include -I$(COMMONDIR)/inc

# architecture flag for cubin build
#CUBIN_ARCH_FLAG := -m32

# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \

# Compiler-specific flags
NVCCFLAGS := -Xcompiler -fPIC 
CXXFLAGS  := $(CXXWARN_FLAGS) -Xcompiler -fPIC  --shared 
CFLAGS    := $(CWARN_FLAGS) -Xcompiler -fPIC --shared 

# Common flags
COMMONFLAGS += $(INCLUDES) -DUNIX -D__LINUX__ -DLINUX -DNVCC




# Debug/release configuration
ifeq ($(dbg),1)
	COMMONFLAGS += -g 
	NVCCFLAGS   += -D_DEBUG
	BINSUBDIR   := 
	#debug
	LIBSUFFIX   := D
else 
	COMMONFLAGS += -O3 
	BINSUBDIR   := 
	#release
	LIBSUFFIX   :=
	NVCCFLAGS   += --compiler-options -fno-strict-aliasing  --host-compilation=C++
	CXXFLAGS    += -fno-strict-aliasing
	CFLAGS      += -fno-strict-aliasing
endif

# append optional arch/SM version flags (such as -arch sm_11)
NVCCFLAGS += $(SMVERSIONFLAGS)

# architecture flag for cubin build
CUBIN_ARCH_FLAG = -m32
#$(TARGET_ARCH)
#-m32

# detect if 32 bit or 64 bit system
#HP_64 =	$(shell uname -m | grep 64)
ifeq ($(ARCH), )
	HP_64=$(shell uname -m | grep 64)
endif
ifeq ($(ARCH), x64)
	HP_64=x64
endif
ifeq ($(ARCH), x32)
	HP_64=
endif

CUBIN_ARCH_FLAG=-m32 

ifeq "$(strip $(HP_64))" ""
	NVCCFLAGS += -D__x86_32__
	CUBIN_ARCH_FLAG=-m32
else
	NVCCFLAGS += -D__x86_64__
	CUBIN_ARCH_FLAG=-m64 
endif




# OpenGL is used or not (if it is used, then it is necessary to include GLEW)
ifeq ($(USEGLLIB),1)

	ifneq ($(DARWIN),)
		OPENGLLIB := -L/System/Library/Frameworks/OpenGL.framework/Libraries -lGL -lGLU $(COMMONDIR)/lib/$(OSLOWER)/libGLEW.a
	else
		OPENGLLIB := -lGL -lGLU -lX11 -lXi -lXmu

		ifeq "$(strip $(HP_64))" ""
			OPENGLLIB += -lGLEW -L/usr/lib32 -L/usr/lib
			#CUBIN_ARCH_FLAG := -m32
		else
			OPENGLLIB += -lGLEW -L/usr/lib64
			#CUBIN_ARCH_FLAG := -m64
		endif
	endif

endif


ifeq ($(USEGLUT),1)
	ifneq ($(DARWIN),)
		OPENGLLIB += -framework GLUT
	else
		OPENGLLIB += -lglut
	endif
endif


ifeq ($(USEPARAMGL),1)
	PARAMGLLIB := -lparamgl$(LIBSUFFIX)
endif

ifeq ($(USERENDERCHECKGL),1)
	RENDERCHECKGLLIB := -lrendercheckgl$(LIBSUFFIX)
endif

ifeq ($(USECUDPP), 1)
	ifeq "$(strip $(HP_64))" ""
		CUDPPLIB := -lcudpp
	else
		CUDPPLIB := -lcudpp64
	endif

	CUDPPLIB := $(CUDPPLIB)$(LIBSUFFIX)

	ifeq ($(emu), 1)
		CUDPPLIB := $(CUDPPLIB)_emu
	endif
endif


# Libs
#LIB       += -L$(CUDA_INSTALL_PATH)/lib -L$(LIBDIR) -L$(COMMONDIR)/lib -lcuda -lcudart ${OPENGLLIB}

LIB    += -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_INSTALL_PATH)/lib64 -L$(LIBDIR) -L$(COMMONDIR)/lib/$(OSLOWER) -L$(NVSDKCUDA_ROOT)/lib



# If dynamically linking to CUDA and CUDART, we exclude the libraries from the LIB
ifeq ($(USECUDADYNLIB),1)
     LIB += ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB) ${LIB} 
else
# static linking, we will statically link against CUDA and CUDART
  ifeq ($(USEDRVAPI),1)
     LIB += -lcuda   ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB)
  else
     LIB += -lcudart ${OPENGLLIB} $(PARAMGLLIB) $(RENDERCHECKGLLIB) $(CUDPPLIB)
  endif
endif

ifeq ($(USECUFFT),1)
  ifeq ($(emu),1)
    LIB += -lcufftemu
  else
    LIB += -lcufft
  endif
endif

ifeq ($(USECUBLAS),1)
  ifeq ($(emu),1)
    LIB += -lcublasemu
  else
    LIB += -lcublas
  endif
endif


# Lib/exe configuration
ifneq ($(STATIC_LIB),)
	TARGETDIR := $(LIBDIR)
	TARGET   := $(subst .a,$(LIBSUFFIX).a,$(LIBDIR)$(STATIC_LIB))
	LINKLINE  = ar rucv $(CUBIN_ARCH_FLAG) $(TARGET) $(OBJS) 
else
	#do we really need libCutil??? from SDK
	#LIB += -lcutil$(LIBSUFFIX)
	# Device emulation configuration
	ifeq ($(emu), 1)
		NVCCFLAGS   += -deviceemu
		CUDACCFLAGS += 
		BINSUBDIR   := emu$(BINSUBDIR)
		# consistency, makes developing easier
		CXXFLAGS		+= -D__DEVICE_EMULATION__
		CFLAGS			+= -D__DEVICE_EMULATION__
	endif
	ifneq ($(SHARED_LIB),)
		TARGETDIR := $(LIBDIR)
		#TARGET   := $(subst .so,$(LIBSUFFIX).so,$(LIBDIR)/$(SHARED_LIB))
		TARGET   := $(LIBDIR)$(SHARED_LIB)
		LINKLINE  = $(LINK) $(CUBIN_ARCH_FLAG) -o $(TARGET) $(OBJS) $(LIB) --shared -fPIC 
	else
		TARGETDIR := $(BINDIR)$(BINSUBDIR)
		TARGET    := $(TARGETDIR)$(EXECUTABLE)
		ifeq ($(EXECUTABLE),)#GPUCV, if no executable, no link
			LINKLINE  = 
		else
			LINKLINE  = $(LINK) $(CUBIN_ARCH_FLAG) -o $(TARGET) $(OBJS) $(LIB) --shared -fPIC
		endif
	endif
endif


#check if we compile + link or compile only
NO_LINKING?=1
ifeq ($(NO_LINKING), 0)
	LINKLINE =
endif 
# check if verbose 
ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := @
endif


################################################################################
# Check for input flags and set compiler flags appropriately
################################################################################
ifeq ($(fastmath), 1)
	NVCCFLAGS += -use_fast_math
endif


ifeq ($(keep), 1)
	NVCCFLAGS += -keep
	NVCC_KEEP_CLEAN := *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx
endif

ifdef maxregisters
	NVCCFLAGS += -maxrregcount $(maxregisters)
endif

# Add cudacc flags
NVCCFLAGS += $(CUDACCFLAGS) -DNVCC
#-DNVCC is from thread http://forums.nvidia.com/index.php?showtopic=90555 about having  "error: operand of "*" must be a pointer" error message

# workaround for mac os x cuda 1.1 compiler issues
ifneq ($(DARWIN),)
	NVCCFLAGS += --host-compilation=C
endif


# Add common flags
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)

ifeq ($(nvcc_warn_verbose),1)
	NVCCFLAGS += $(addprefix --compiler-options ,$(CXXWARN_FLAGS)) 
	NVCCFLAGS += --compiler-options -fno-strict-aliasing
endif


################################################################################
# Set up object files
################################################################################
#CU_FILENAMES = $(notdir $(CUFILES))

OBJDIR := $(ROOTOBJDIR)/$(BINSUBDIR)
#OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(notdir $(CCFILES)))
#OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c.o,$(notdir $(CFILES)))
#OBJS +=  $(patsubst %.cu,$(OBJDIR)/%.cu.o,$(notdir $(CUFILES)))
 
#))
ifeq ($(NO_LINKING), 2)
	OBJS +=  $(shell find  $(OBJDIR) -name "*.o" -printf '%p ')
else
	OBJS +=  $(patsubst %.cu,$(OBJDIR)/%.cu.o,$(CUFILES))
	OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.o,$(CCFILES))#set to reuse precompiled CPP files
	OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c.o,$(CFILES))
endif 


################################################################################
# Set up cubin files
################################################################################
CUBINDIR := $(OBJDIR)data
CUBINS +=  $(patsubst %.cu,$(CUBINDIR)/%.cubin,$(notdir $(CUBINFILES)))

################################################################################
# Rules
################################################################################
$(OBJDIR)/%.c.o : $(SRCDIR)%.c $(C_DEPS)
	#$(VERBOSE)$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/%.cpp.o : $(SRCDIR)%.cpp $(C_DEPS)
	#$(VERBOSE)$(CXX) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/%.cu.o : $(SRCDIR)%.cu $(CU_DEPS)
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -c $<

$(CUBINDIR)/%.cubin : $(SRCDIR)%.cu cubindirectory
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -cubin $<

#
# The following definition is a template that gets instantiated for each SM
# version (sm_10, sm_13, etc.) stored in SMVERSIONS.  It does 2 things:
# 1. It adds to OBJS a .cu_sm_XX.o for each .cu file it finds in CUFILES_sm_XX.
# 2. It generates a rule for building .cu_sm_XX.o files from the corresponding 
#    .cu file.
#
# The intended use for this is to allow Makefiles that use common.mk to compile
# files to different Compute Capability targets (aka SM arch version).  To do
# so, in the Makefile, list files for each SM arch separately, like so:
#
# CUFILES_sm_10 := mycudakernel_sm10.cu app.cu
# CUFILES_sm_12 := anothercudakernel_sm12.cu
#
define SMVERSION_template
OBJS += $(patsubst %.cu,$(OBJDIR)/%.cu_$(1).o,$(notdir $(CUFILES_$(1))))
$(OBJDIR)/%.cu_$(1).o : $(SRCDIR)%.cu $(CU_DEPS)
	$(VERBOSE)$(NVCC) -o $$@ -c $$< $(NVCCFLAGS) -arch $(1)
endef

# This line invokes the above template for each arch version stored in
# SM_VERSIONS.  The call funtion invokes the template, and the eval
# function interprets it as make commands.
$(foreach smver,$(SM_VERSIONS),$(eval $(call SMVERSION_template,$(smver))))

$(TARGET): makedirectories $(OBJS) $(CUBINS) Makefile
	$(VERBOSE)$(LINKLINE)

#@echo "Linking files($(NO_LINKING)) to $(TARGET): $(OBJS)"

cubindirectory:
	$(VERBOSE)mkdir -p $(CUBINDIR)

makedirectories:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)
	$(VERBOSE)mkdir -p $(TARGETDIR)


tidy :
	$(VERBOSE)find . | egrep "#" | xargs rm -f
	$(VERBOSE)find . | egrep "\~" | xargs rm -f

clean : tidy
	$(VERBOSE)rm -f $(OBJS)
	$(VERBOSE)rm -f $(CUBINS)
	$(VERBOSE)rm -f $(TARGET)
	$(VERBOSE)rm -f $(NVCC_KEEP_CLEAN)
	echo "Obj files:  $(OBJS)"
	echo "CU files:  $(CUFILES)"
	
clobber : clean
	$(VERBOSE)rm -rf $(ROOTOBJDIR)
	$(VERBOSE)echo "Obj files:  $(OBJS)"
	$(VERBOSE)echo "CU files:  $(CUFILES)"

