Debug("Include gpucv_cuda-plug.lua")
if _ACTION  and not (_ACTION=="checkdep") then

	if (cuda_enable>0) then
		--GPUCVCuda - cxcoregcu
		CreateProject("../../", "cxcoregcu", 	"SharedLib", "plugin", "_GPUCV_CXCOREGCU_DLL",
			{gpucv_core_list,gpucv_opengl_list,"GPUCVCuda", "cuda"})
			--files {"../../../include/cxcoregcu/*.cu"}
			package.guid = "8661697D-D3D1-5840-85A8-CC1D00940751"--random package id
			
		--GPUCVCuda - cvgcu
		CreateProject("../../", "cvgcu", 	"SharedLib", "plugin", "_GPUCV_CVGCU_DLL",
			{gpucv_core_list,gpucv_opengl_list,"GPUCVCuda", "cuda", "cxcoregcu"})
			--files {"../../../include/cvgcu/*.cu"}
			package.guid = "8661697D-D3D1-5840-85A8-CC1D00940752"--random package id
			
		table.insert(gpucv_cuda_libs,{"GPUCVCuda","cxcoregcu","cvgcu"})
		
		if (cuda_npp_enable==1) then
			--GPUCVCuda - gcvnpp
			CreateProject("../../", "gcvnpp", 	"SharedLib", "plugin", "_GPUCV_GCVNPP_DLL",
				{gpucv_core_list,gpucv_opengl_list,gpucv_cuda_libs})
		
			table.insert(gpucv_cuda_libs,{"gcvnpp"})
		end
		
	end
end--_ACTION
Debug("Include gpucv_cuda-plug.lua...done")