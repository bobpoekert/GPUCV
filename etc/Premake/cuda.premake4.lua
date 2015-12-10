Debug("Include cuda.premake4.lua")


cudpp_enable=0

--check the cuda plugin options, cuda_enable is > 0 when env variables have been detected...
if cuda_enable > 0 then
	if _OPTIONS["plug-cuda-off"] then
		cuda_enable =0
		printf("NVIDIA CUDA toolkit found but disabled.")
	else
		--printf("NVIDIA CUDA toolkit found -> GpuCV CUDA plugin enabled (Can be turned off using --plug-cuda-off option).")
	end
end	
--check CUDA path===========================
if _ACTION =="gmake" then --there is a bug in premake that miss with env variable for gmake target
	cudaToolKitDir_inc = os.getenv("CUDA_INC_PATH")
	cudaToolKitDir_lib = os.getenv("CUDA_LIB_PATH")
 	cudaToolKitDir_bin = os.getenv("CUDA_BIN_PATH")

	cudaSDKDir = os.getenv("NVSDKCOMPUTE_ROOT") --since CUDA 3.0
	cudaNPPSDKDir = os.getenv("NPP_SDK_PATH")
else
  	cudaToolKitDir_inc = "$(CUDA_INC_PATH)"
 	cudaToolKitDir_lib = "$(CUDA_LIB_PATH)"
    cudaToolKitDir_bin = "$(CUDA_BIN_PATH)"

        cudaSDKDir = "$(NVSDKCOMPUTE_ROOT)" --since CUDA 3.0
	cudaNPPSDKDir = "$(NPP_SDK_PATH)"
end
--==============================================
if _ACTION  and not (_ACTION=="checkdep") then
if (cuda_enable > 0) then
printf("Using CUDA dependencies")
	
	configuration "*"
		includedirs{ 	
			cudaSDKDir..'/C/common/inc'
			,cudaToolKitDir_inc
			}
		libdirs{ 	
				cudaSDKDir..'/C/lib'
				,cudaSDKDir..'/C/common/lib'
				,cudaSDKDir..'/lib'
				}

	configuration {"x32"}
		libdirs{cudaToolKitDir_lib}
	configuration {"x64"}
		libdirs{cudaToolKitDir_lib}
		libdirs{cudaToolKitDir_lib..'/../lib64'} --the lib path may be set to lib/ and not lib64...
		
	configuration "*"
		links{ 	
				"cudart"
				,"cuda" --libcuda.so is not in cudaToolkit 3.0 under linux but in devdriver(/usr/lib)
				,"cublas"
				,"cufft"
				}
				
	configuration {"linux"}
		libdirs (cudaSDKDir..'/C/common/lib/linux')
	--configuration {"macosx"}
	--	links "cutil"
		
	
		
		
	
	--configuration {"windows", "x32", "Debug"}
	--	links("cutil32D")
	--configuration {"windows", "x32", "Release"}
	--	links("cutil32")
	--configuration {"windows", "x64", "Debug"}
	--	links("cutil64D")
	--configuration {"windows", "x64", "Release"}
	--	links("cutil64")

	configuration "*"
		defines ("_GPUCV_SUPPORT_CUDA")
	--CUFFT
		links ("cufft")
		defines ("_GPUCV_CUDA_SUPPORT_CUFFT")	
	--CUDPP
	if cudpp_enable ==1 then
	--configuration {"plug-cudpp"}
		defines("_GPUCV_CUDA_SUPPORT_CUDPP")
	
--	configuration {"windows"}
		--includedirs{"$(CUDA_INC_PATH)/../cudpp/cudpp/include/"}
		--libdirs{"$(CUDA_INC_PATH)/../cudpp/cudpp/lib/"}
		
--	configuration {"linux"} --to test
--		includedirs{"$(NVSDKCUDA_ROOT)/shared/inc/cudpp/"}
--		libdirs{"$(NVSDKCUDA_ROOT)/shared/lib/linux"}
		
	if _OPTIONS["os"]=="windows" then
		configuration "*"
			--includedirs{"c:/CUDA/cudpp/cudpp/include/"} --really here and not in the SDK?
			--libdirs{"c:/CUDA/cudpp/cudpp/lib/"}
		configuration {"x32"}
			links ("cudpp32")
		configuration {"x64"}
			links ("cudpp64")		
	else
		configuration "*"
			--includedirs{"/usr/local/CUDA/cudpp/cudpp/include/"}
			libdirs{"/usr/local/CUDA/cudpp/cudpp/lib/"}
		configuration {"x32"}--, "Debug"}
			links ("cudpp_i386")
		configuration {"x64"}
			links ("cudpp_x86_64")			
	end
	end--cudpp

	--not tested yet
	if(cuda_npp_enable==1) then
		defines("_GPUCV_SUPPORT_NPP")
		includedirs{cudaNPPSDKDir.."/common/npp/include"}
		libdirs{cudaNPPSDKDir.."/common/lib/"}
		
	configuration {"windows", "x32"}
		links ("libnpp-mt")
	configuration {"windows", "x64"}
		links ("libnpp-mt-x64")
	--	links ("UtilNPP-mt")
	--add linux, X32 and 64 supports
	end
--====================



--post build command
	--windows use Visual Studio Custom build files....
	if _OPTIONS["os"] == "windows" then
		if (use_nsight==1) then
		Debug("Using Nsight custom build rules")
			configuration {  "Debug_Nsight" }
				project()["custom_build_tools_name"]="Cudart Build Rule"
				project()["custom_build_tools_path"]="NsightCudaRuntimeApi.v30.rules"
			configuration {  "Release_Nsight" }
				project()["custom_build_tools_name"]="Cudart Build Rule"
				project()["custom_build_tools_path"]="NsightCudaRuntimeApi.v30.rules"
		--		targetdir = targetdir .. "-nsight"
		end
		configuration {  "x32", "x64" }
			project()["custom_build_tools_name"]="GpuCVCUDA_VS2008"
			project()["custom_build_tools_path"]="../../../etc/vs2008_cuda.rules"
		
	else
	--use custom build rules MakeFiles for .cu files		
		if os.isfile("../../src/"..project()["category"].."/"..project()["name"].."/Makefile") then
			configuration "linux"
				prebuildcommands { "rm -f $(TARGET)" }
				postbuildcommands { "rm -f $(TARGET)" }
				postbuildcommands { "$(MAKE) $(MAKE_OPT) -C ../../../src/"..project()["category"].."/"..project()["name"].." -f Makefile" }
				--prelinkcommands "rm -f $(TARGET)"
				--linkoptions { " $(OBJDIR)/*.cu.o"} --link all object in the path (for CUDA *.cu.o)
			configuration "macosx"
				prebuildcommands { "rm -f $(TARGET)" }
				postbuildcommands { "rm -f $(TARGET)" }
				postbuildcommands { "$(MAKE) $(MAKE_OPT) -C ../../../src/"..project()["category"].."/"..project()["name"].." -f Makefile" }
				--prelinkcommands "rm -f $(TARGET)"
				--linkoptions { " $(OBJDIR)/*.cu.o"} --link all object in the path (for CUDA *.cu.o)
		end
	end
--
--printf("Using CUDA dependencies...done")
end--CUDA
end--_ACTION
Debug("Include cuda.premake4.lua...done")
