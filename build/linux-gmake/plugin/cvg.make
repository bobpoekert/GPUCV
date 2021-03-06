# GNU Make project makefile autogenerated by Premake
ifndef config
  config=debug32
endif

ifndef verbose
  SILENT = @
endif

ifndef CC
  CC = gcc
endif

ifndef CXX
  CXX = g++
endif

ifndef AR
  AR = ar
endif

ifeq ($(config),debug32)
  OBJDIR     = ../../../obj/linux-gmake/x32/Debug/cvg
  TARGETDIR  = ../../../lib/linux-gmake/x32/Debug
  TARGET     = $(TARGETDIR)/libcvg32D.so
  DEFINES   += -D__LINUX__ -D_LINUX -D_USRDLL -D_GPUCV_CVG_DLL -D_SG_TLS_SUPPORT_GL -D__x86_32__ -DDEBUG -D_DEBUG
  INCLUDES  += -I../../../src/plugin/cvg -I../../../src/lib/cvg -I../../../dependencies/SugoiTools/include -I../../../dependencies/SugoiTools/dependencies/include -I/usr/include -I/usr/local/include -I/usr/local/include/opencv -I/usr/include/opencv -I../../../include -I../../../dependencies/include -I../../../src/lib -I../../../src/plugin
  CPPFLAGS  += -MMD -MP $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -g -m32 -fPIC
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -shared -m32 -L/usr/lib32 -L/usr/local/lib -L/usr/lib -L../../../dependencies/SugoiTools/lib/linux-gmake/x32/Debug -L../../../dependencies/SugoiTools/lib/linux-gmake -L../../../lib/linux-gmake -L../../../lib/linux-gmake/x32/Debug
  LIBS      += -lSugoiTools32D -lSugoiPThread32D -lSugoiTracer32D -lGL -lglut -lGLEW -lGLU -lpthread -lGPUCVHardware32D -lGPUCVTexture32D -lGPUCVCore32D -lGPUCV32D -lcv -lcxcore -lcvaux -lhighgui -lcxcoreg32D -lhighguig32D
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LDDEPS    += ../../../lib/linux-gmake/x32/Debug/libGPUCVHardware32D.so ../../../lib/linux-gmake/x32/Debug/libGPUCVTexture32D.so ../../../lib/linux-gmake/x32/Debug/libGPUCVCore32D.so ../../../lib/linux-gmake/x32/Debug/libGPUCV32D.so ../../../lib/linux-gmake/x32/Debug/libcxcoreg32D.so ../../../lib/linux-gmake/x32/Debug/libhighguig32D.so
  LINKCMD    = $(CXX) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(RESOURCES) $(ARCH) $(LIBS)
  define PREBUILDCMDS
	@echo Running pre-build commands
	export ARCH=32
	export GCV_BIN_SUFFIXE=32D
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

ifeq ($(config),release32)
  OBJDIR     = ../../../obj/linux-gmake/x32/Release/cvg
  TARGETDIR  = ../../../lib/linux-gmake/x32/Release
  TARGET     = $(TARGETDIR)/libcvg32.so
  DEFINES   += -D__LINUX__ -D_LINUX -D_USRDLL -D_GPUCV_CVG_DLL -D_SG_TLS_SUPPORT_GL -D__x86_32__ -DNDEBUG
  INCLUDES  += -I../../../src/plugin/cvg -I../../../src/lib/cvg -I../../../dependencies/SugoiTools/include -I../../../dependencies/SugoiTools/dependencies/include -I/usr/include -I/usr/local/include -I/usr/local/include/opencv -I/usr/include/opencv -I../../../include -I../../../dependencies/include -I../../../src/lib -I../../../src/plugin
  CPPFLAGS  += -MMD -MP $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -O2 -m32 -fPIC
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -s -shared -m32 -L/usr/lib32 -L/usr/local/lib -L/usr/lib -L../../../dependencies/SugoiTools/lib/linux-gmake/x32/Release -L../../../dependencies/SugoiTools/lib/linux-gmake -L../../../lib/linux-gmake -L../../../lib/linux-gmake/x32/Release
  LIBS      += -lSugoiTools32 -lSugoiPThread32 -lSugoiTracer32 -lGL -lglut -lGLEW -lGLU -lpthread -lGPUCVHardware32 -lGPUCVTexture32 -lGPUCVCore32 -lGPUCV32 -lcv -lcxcore -lcvaux -lhighgui -lcxcoreg32 -lhighguig32
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LDDEPS    += ../../../lib/linux-gmake/x32/Release/libGPUCVHardware32.so ../../../lib/linux-gmake/x32/Release/libGPUCVTexture32.so ../../../lib/linux-gmake/x32/Release/libGPUCVCore32.so ../../../lib/linux-gmake/x32/Release/libGPUCV32.so ../../../lib/linux-gmake/x32/Release/libcxcoreg32.so ../../../lib/linux-gmake/x32/Release/libhighguig32.so
  LINKCMD    = $(CXX) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(RESOURCES) $(ARCH) $(LIBS)
  define PREBUILDCMDS
	@echo Running pre-build commands
	export ARCH=32
	export GCV_BIN_SUFFIXE=32
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

ifeq ($(config),debug64)
  OBJDIR     = ../../../obj/linux-gmake/x64/Debug/cvg
  TARGETDIR  = ../../../lib/linux-gmake/x64/Debug
  TARGET     = $(TARGETDIR)/libcvg64D.so
  DEFINES   += -D__LINUX__ -D_LINUX -D_USRDLL -D_GPUCV_CVG_DLL -D_SG_TLS_SUPPORT_GL -D__x86_64__ -DDEBUG -D_DEBUG
  INCLUDES  += -I../../../src/plugin/cvg -I../../../src/lib/cvg -I../../../dependencies/SugoiTools/include -I../../../dependencies/SugoiTools/dependencies/include -I/usr/include -I/usr/local/include -I/usr/local/include/opencv -I/usr/include/opencv -I../../../include -I../../../dependencies/include -I../../../src/lib -I../../../src/plugin
  CPPFLAGS  += -MMD -MP $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -g -m64 -fPIC
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -shared -m64 -L/usr/lib64 -L/usr/local/lib -L/usr/lib64 -L../../../dependencies/SugoiTools/lib/linux-gmake/x64/Debug -L../../../dependencies/SugoiTools/lib/linux-gmake -L../../../lib/linux-gmake -L../../../lib/linux-gmake/x64/Debug
  LIBS      += -lSugoiTools64D -lSugoiPThread64D -lSugoiTracer64D -lGL -lglut -lGLEW -lGLU -lpthread -lGPUCVHardware64D -lGPUCVTexture64D -lGPUCVCore64D -lGPUCV64D -lcv -lcxcore -lcvaux -lhighgui -lcxcoreg64D -lhighguig64D
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LDDEPS    += ../../../lib/linux-gmake/x64/Debug/libGPUCVHardware64D.so ../../../lib/linux-gmake/x64/Debug/libGPUCVTexture64D.so ../../../lib/linux-gmake/x64/Debug/libGPUCVCore64D.so ../../../lib/linux-gmake/x64/Debug/libGPUCV64D.so ../../../lib/linux-gmake/x64/Debug/libcxcoreg64D.so ../../../lib/linux-gmake/x64/Debug/libhighguig64D.so
  LINKCMD    = $(CXX) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(RESOURCES) $(ARCH) $(LIBS)
  define PREBUILDCMDS
	@echo Running pre-build commands
	export ARCH=64
	export GCV_BIN_SUFFIXE=64D
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

ifeq ($(config),release64)
  OBJDIR     = ../../../obj/linux-gmake/x64/Release/cvg
  TARGETDIR  = ../../../lib/linux-gmake/x64/Release
  TARGET     = $(TARGETDIR)/libcvg64.so
  DEFINES   += -D__LINUX__ -D_LINUX -D_USRDLL -D_GPUCV_CVG_DLL -D_SG_TLS_SUPPORT_GL -D__x86_64__ -DNDEBUG
  INCLUDES  += -I../../../src/plugin/cvg -I../../../src/lib/cvg -I../../../dependencies/SugoiTools/include -I../../../dependencies/SugoiTools/dependencies/include -I/usr/include -I/usr/local/include -I/usr/local/include/opencv -I/usr/include/opencv -I../../../include -I../../../dependencies/include -I../../../src/lib -I../../../src/plugin
  CPPFLAGS  += -MMD -MP $(DEFINES) $(INCLUDES)
  CFLAGS    += $(CPPFLAGS) $(ARCH) -O2 -m64 -fPIC
  CXXFLAGS  += $(CFLAGS) 
  LDFLAGS   += -s -shared -m64 -L/usr/lib64 -L/usr/local/lib -L/usr/lib64 -L../../../dependencies/SugoiTools/lib/linux-gmake/x64/Release -L../../../dependencies/SugoiTools/lib/linux-gmake -L../../../lib/linux-gmake -L../../../lib/linux-gmake/x64/Release
  LIBS      += -lSugoiTools64 -lSugoiPThread64 -lSugoiTracer64 -lGL -lglut -lGLEW -lGLU -lpthread -lGPUCVHardware64 -lGPUCVTexture64 -lGPUCVCore64 -lGPUCV64 -lcv -lcxcore -lcvaux -lhighgui -lcxcoreg64 -lhighguig64
  RESFLAGS  += $(DEFINES) $(INCLUDES) 
  LDDEPS    += ../../../lib/linux-gmake/x64/Release/libGPUCVHardware64.so ../../../lib/linux-gmake/x64/Release/libGPUCVTexture64.so ../../../lib/linux-gmake/x64/Release/libGPUCVCore64.so ../../../lib/linux-gmake/x64/Release/libGPUCV64.so ../../../lib/linux-gmake/x64/Release/libcxcoreg64.so ../../../lib/linux-gmake/x64/Release/libhighguig64.so
  LINKCMD    = $(CXX) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(RESOURCES) $(ARCH) $(LIBS)
  define PREBUILDCMDS
	@echo Running pre-build commands
	export ARCH=64
	export GCV_BIN_SUFFIXE=64
  endef
  define PRELINKCMDS
  endef
  define POSTBUILDCMDS
  endef
endif

OBJECTS := \
	$(OBJDIR)/cvg_dist_transform.o \
	$(OBJDIR)/cvg_sat.o \
	$(OBJDIR)/cvg.o \
	$(OBJDIR)/cvg_imgproc_histo.o \

RESOURCES := \

SHELLTYPE := msdos
ifeq (,$(ComSpec)$(COMSPEC))
  SHELLTYPE := posix
endif
ifeq (/bin,$(findstring /bin,$(SHELL)))
  SHELLTYPE := posix
endif

.PHONY: clean prebuild prelink

all: $(TARGETDIR) $(OBJDIR) prebuild prelink $(TARGET)

$(TARGET): $(GCH) $(OBJECTS) $(LDDEPS) $(RESOURCES)
	@echo Linking cvg
	$(SILENT) $(LINKCMD)
	$(POSTBUILDCMDS)

$(TARGETDIR):
	@echo Creating $(TARGETDIR)
ifeq (posix,$(SHELLTYPE))
	$(SILENT) mkdir -p $(TARGETDIR)
else
	$(SILENT) mkdir $(subst /,\\,$(TARGETDIR))
endif

$(OBJDIR):
	@echo Creating $(OBJDIR)
ifeq (posix,$(SHELLTYPE))
	$(SILENT) mkdir -p $(OBJDIR)
else
	$(SILENT) mkdir $(subst /,\\,$(OBJDIR))
endif

clean:
	@echo Cleaning cvg
ifeq (posix,$(SHELLTYPE))
	$(SILENT) rm -f  $(TARGET)
	$(SILENT) rm -rf $(OBJDIR)
else
	$(SILENT) if exist $(subst /,\\,$(TARGET)) del $(subst /,\\,$(TARGET))
	$(SILENT) if exist $(subst /,\\,$(OBJDIR)) rmdir /s /q $(subst /,\\,$(OBJDIR))
endif

prebuild:
	$(PREBUILDCMDS)

prelink:
	$(PRELINKCMDS)

ifneq (,$(PCH))
$(GCH): $(PCH)
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o $@ -c $<
endif

$(OBJDIR)/cvg_dist_transform.o: ../../../src/plugin/cvg/cvg_dist_transform.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o $@ -c $<
$(OBJDIR)/cvg_sat.o: ../../../src/plugin/cvg/cvg_sat.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o $@ -c $<
$(OBJDIR)/cvg.o: ../../../src/plugin/cvg/cvg.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o $@ -c $<
$(OBJDIR)/cvg_imgproc_histo.o: ../../../src/plugin/cvg/cvg_imgproc_histo.cpp
	@echo $(notdir $<)
	$(SILENT) $(CXX) $(CXXFLAGS) -o $@ -c $<

-include $(OBJECTS:%.o=%.d)
