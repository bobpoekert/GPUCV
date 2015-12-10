Debug("Include gpucv_switch.lua")

if _ACTION  and not (_ACTION=="checkdep") then
	--GPUCVSwitch
	CreateProject("../../", "GPUCVSwitch", "SharedLib", "lib", "_GPUCV_SWITCH_DLL",
		{gpucv_core_list})

	if not _OPTIONS["use-switch-off"] or _OPTIONS["use-switch"] then
		SWITCH_DEFINES="_GPUCV_SUPPORT_SWITCH"
		--CV_switch
		CreateProject("../../", "cv_switch", "SharedLib" , "lib", "_CV_SWITCH_DLL",
			{gpucv_core_list, "GPUCVSwitch"})
		
		--CXCORE_switch
		CreateProject("../../", "cxcore_switch", "SharedLib" , "lib", "_CXCORE_SWITCH_DLL",
			{gpucv_core_list, "GPUCVSwitch"})
						
		--highgui_switch
		CreateProject("../../", "highgui_switch", "SharedLib" , "lib", "_HIGHGUI_SWITCH_DLL",
			{gpucv_core_list, "GPUCVSwitch"})

 		--CVAUX...

		gpucv_switch_lib_list = {"GPUCVSwitch", "cxcore_switch", "cv_switch", "highgui_switch"}
	else
		gpucv_switch_lib_list = {"GPUCVSwitch"}
	end
end--_ACTION
Debug("Include gpucv_switch.lua...done")
