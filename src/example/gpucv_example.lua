--AddOption("use-qt", "GpuCV: Compile the QT example application.")
--------------------------------------------------------------------------------------------------
---				GPUCV examples	 					--
--------------------------------------------------------------------------------------------------
if _ACTION  and not (_ACTION=="checkdep") then

	--GPUCVConsole
	CreateProject("../../", "GPUCVConsole", "ConsoleApp" , "example", SWITCH_DEFINES,
			{gpucv_core_list, gpucv_opengl_list, gpucv_switch_lib_list})--gpucv_cuda_libs,
			package.guid = "C1B52701-AD09-424C-BA66-CAD6BEE3BB18"--random package id
		
	--GPUCVSimpleApp
	CreateProject("../../", "GPUCVSimpleApp", "ConsoleApp" , "example", SWITCH_DEFINES, 
		{gpucv_core_list, gpucv_opengl_list, gpucv_switch_lib_list})--,gpucv_cuda_libs
			
	--GPUCVCamDemo
	CreateProject("../../", "GPUCVCamDemo", "ConsoleApp" , "example", SWITCH_DEFINES,
			{gpucv_core_list, gpucv_opengl_list, gpucv_switch_lib_list})--, gpucv_cuda_libs
					
	--simple_c_integration
	CreateProject("../../", "simple_c_integration", "ConsoleApp" , "example", SWITCH_DEFINES,
			{gpucv_core_list, gpucv_opengl_list, gpucv_switch_lib_list})--,gpucv_cuda_libs

	if _OPTIONS["use-switch"] then
		--switch wrapper srouce code generator
		CreateProject("../../", "genSwitch", "ConsoleApp" , "example", "",
			{gpucv_core_list, gpucv_opengl_list, "GPUCVSwitch"})

	end

end --_ACTION
