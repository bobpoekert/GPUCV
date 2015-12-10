
###############################################################################
# SOURCE VARS
CUFILES		=  	gcu_runtime_api_wrapper.cu\
				cuda_wrapper_api.cu
#			cvgcu_custom.filter.cu

#CCFILES 	:= cvgcu.cpp

# CUDA source files (compiled with cudacc)
#CUFILES		:= template.cu
# CUDA dependency files
#CU_DEPS		:= \
#	template_kernel.cu \

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= 	cuda_misc.cpp\
				DataDsc_CUDA_Array.cpp\
				DataDsc_CUDA_Base.cpp\
				DataDsc_CUDA_Buffer.cpp\
				gpucv_wrapper_c.cpp\
				GPU_NVIDIA_CUDA.cpp\
				StdAfx.cpp


SHARED_LIB_TEMP	:= GPUCVCuda

export SRCDIR = ./
export ROOTDIR = ../../../
export GCU_LIB_NAME = GPUCVCuda
NO_LINKING=1
include $(ROOTDIR)etc/cuda_common_include.mk
include $(ROOTDIR)etc/cuda_common_makefile.mk
#NO_LINKING=1
#include $(ROOTDIR)etc/cuda_common_include.mk
#include $(ROOTDIR)etc/cuda_common_makefile.mk