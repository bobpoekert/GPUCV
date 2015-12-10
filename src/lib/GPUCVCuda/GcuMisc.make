
###############################################################################
# SOURCE VARS
CUFILES		=   cuda_wrapper_api.cu\
		cvgcu_custom.filter.cu\
		gcu_runtime_api_wrapper.cu

SRCDIR = ./
ROOTDIR = ../../../
INCLUDES = -I../../../include\
	-I$(ROOTDIR)/include\
	-I$(ROOTDIR)/src\
	-I$(ROOTDIR)/src/lib\
	-I$(ROOTDIR)/../resources/include\
	-I$(ROOTDIR)/../resources/include/gl\
	-I$(ROOTDIR)/../resources/include/cv/include\
	-I$(ROOTDIR)/../resources/include/cxcore/include\
	-I$(ROOTDIR)/../resources/include/otherlibs/highgui\
	-I$(ROOTDIR)/../resources/include/otherlibs/cvcam/include\
	-I$(ROOTDIR)/../resources/include/cvaux/include\
	-I$(ROOTDIR)/../resources/include/\
	-I/home/allusse/NVIDIA_CUDA_SDK/common/inc/
LIB += -L"/home/allusse/NVIDIA_CUDA_SDK/lib/" -L"/home/allusse/NVIDIA_CUDA_SDK/common/lib/"

include $(ROOTDIR)/etc/cuda_common_makefile.mk


