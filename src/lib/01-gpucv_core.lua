Debug("Include 01-gpucv_core.lua")
------------------------------------------------------------------------------------------------------
---				GPUCV main libraries	 							--
--- NOTE: these libs are independent from OpenCV and do not contains any operators  --
------------------------------------------------------------------------------------------------------
if _ACTION  and not (_ACTION=="checkdep") then

	--GPUCVHardware
	CreateProject("../../", "GPUCVHardware", "SharedLib", "lib", "_GPUCV_HARDWARE_DLL", 
	{"main", "sugoitools", "opengl", "pthread"})
	--	package.guid = "4D6A9DAD-76AD-414A-A585-941267BFDE2F"--random package id
	--
	--GPUCVTexture
	CreateProject("../../", "GPUCVTexture", "SharedLib", "lib", {"_GPUCV_TEXTURE_DLL","GLUT_BUILDING_LIB"},
		{"main", "sugoitools", "opengl", "GPUCVHardware", "pthread"})
	--	package.guid = "B788A2F1-9183-1C4D-A6CB-4A6E9FC9D401"--random package id

	--GPUCVCore
	CreateProject("../../", "GPUCVCore", 	"SharedLib", "lib", "_GPUCV_CORE_DLL",
		{"main", "sugoitools", "opengl", "pthread", "GPUCVHardware", "GPUCVTexture"})
	--	package.guid = "C5920ADC-52C0-7C42-9F2B-082F101AB591"--random package id
	--

	--GPUCV: integration with opencv and Misc functions
	CreateProject("../../", "GPUCV", 	"SharedLib", "lib", "_GPUCV_DLL",
		{"main", "sugoitools", "opengl", "pthread", "GPUCVHardware", "GPUCVTexture", "GPUCVCore", "opencv"})
		package.guid = "BFB29282-C75E-8F48-ADF8-3F12AC24BC68"--random package id
		files {
				"./bin/FShaders/*.frag",
				"./bin/VShaders/*.vert",
				"./bin/GShaders/*.geo"
			}

	gpucv_core_list= {"sugoitools","opengl", "pthread", "GPUCVHardware", "GPUCVTexture", "GPUCVCore", "GPUCV","opencv","main"}
end --_ACTION
Debug("Include 01-gpucv_core.lua...done")