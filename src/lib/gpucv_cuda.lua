Debug("Include gpucv_cuda.lua")
if _ACTION  and not (_ACTION=="checkdep") then

	if (cuda_enable>0) then
		--GPUCVCuda
		CreateProject("../../", "GPUCVCuda", "SharedLib", "lib", "_GPUCV_CUDA_DLL",
			{gpucv_core_list, "cuda", "cxcoreg","highguig"}) --cxcoreg is used for based cvgShowImage/etc/...
			files{"../../../include/GPUCVCuda/*.cu"}
			
			package.guid = "8661697D-D3D1-5840-85A8-CC1D00940750"--random package id
			
		--	if((options["os"]=="linux) or (options["os"]=="macosx"")) then
		--		package.prelinkcommands = {"make --no-print-directory -C "..default_rootpath.."src/lib/GPUCVCuda -f Makefile all"}
		--	-	package.config["Release"].postbuildcommands = { "g++ -fPIC -i "..default_rootpath.."Debug/gnu/GPUCVCuda/*.*o -o "..default_rootpath.."lib/gnu/libGPUCVCudad.so -shared -Wl -L"..default_rootpath.."lib/gnu/ -L"..default_rootpath.."../resources/lib/gnu/ -L/usr/local/cuda/lib -L/home/allusse/NVIDIA_CUDA_SDK/common/lib/ -L/home/allusse/NVIDIA_CUDA_SDK/lib/ -lGL -lglut -lGLEW -lGLU -lGPUCVHardware -lGPUCVTexture -lGPUCVCore -lGPUCV -lSugoiTools -lSugoiTracer -lcv -lcxcore -lcvaux -lhighgui -lcudart -lcutil" }
		--		package.config["Debug"].postbuildcommands = { "g++ -fPIC -i "..default_rootpath.."Debug/gnu/GPUCVCuda/*.*o -o "..default_rootpath.."lib/gnu/libGPUCVCudad.so -shared -Wl -L"..default_rootpath.."lib/gnu/ -L"..default_rootpath.."../resources/lib/gnu/ -L/usr/local/cuda/lib -L/home/allusse/NVIDIA_CUDA_SDK/common/lib/ -L/home/allusse/NVIDIA_CUDA_SDK/lib/ -lGL -lglut -lGLEW -lGLU -lGPUCVHardwared -lGPUCVTextured -lGPUCVCored -lGPUCVd -lSugoiToolsd -lSugoiTracerd -lcv -lcxcore -lcvaux -lhighgui -lcudart -lcutil" }
		--	end
			
		gpucv_cuda_libs = {"cuda"}
	else
		gpucv_cuda_libs = {""}
	end
end--_ACTION
Debug("Include gpucv_cuda.lua...done")