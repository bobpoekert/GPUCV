#parse all the sub directory given by arg ALL_SUB_DIRS and call the MakeFile


#you might not change the end of the file
ifndef CONFIG
  CONFIG=Debug
endif

export CONFIG
export MAKE_NODIR_PRINT = --no-print-directory

.PHONY: all clean local local_clean sub_dir sub_dir_clean

MAKE_FILE_CU_NAME = MakefileCUDA
HAVE_LOCAL_MAKEFILE_CU = $(wildcard MakefileCU)
MAKEFILE_CU_CMD =
 
ifeq ($(strip $(HAVE_LOCAL_MAKEFILE_CU)),) 
	MAKEFILE_CU_CMD =@make -f $(MAKE_FILE_CU_NAME)
	MAKEFILE_CU_CMD_CLEAN =@make -f $(MAKE_FILE_CU_NAME) clean
else
	MAKEFILE_CU_CMD =
	MAKEFILE_CU_CMD_CLEAN =
endif

all:
	@for t in $(ALL_SUB_DIRS); do \
	echo ==== Building  ./$(SHARED_LIB_TMP)/$$t ====;\
	export SHARED_LIB;\
	export SHARED_LIB_TEMP;\
	$(MAKE) $(MAKE_NODIR_PRINT) -C $$t $@ || exit; \
	done
	@echo ==== Building  ./$(SHARED_LIB_TMP) ====;
	@$(MAKEFILE_CU_CMD)
	
clean:
	@for t in $(ALL_SUB_DIRS); do \
	echo ==== Cleaning ./$(SHARED_LIB_TMP)/$$t ====;\
	$(MAKE) $(MAKE_NODIR_PRINT) -C $$t clean|| exit; \
	done
	echo ==== Cleaning  ./$(SHARED_LIB_TMP) ====;
	@$(MAKEFILE_CU_CMD_CLEAN)

	

###############################################################################
