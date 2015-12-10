Debug("Include 02-gpucv_opengl.lua")
---------------------------------------------------------------------------------------------------------
---				GPUCV - Integration with opencv and GPU-accelerated operators --
---tag: TUTO_CREATE_PLUGIN_TAG__STP1__CREATE_PROJECT
---------------------------------------------------------------------------------------------------------
if _ACTION  and not (_ACTION=="checkdep") then

	--GPUCV-cxcoreg
	CreateProject("../../", "cxcoreg",	"SharedLib", "plugin", "_GPUCV_CXCOREG_DLL", gpucv_core_list)
		package.guid = "BFB29282-C75E-8F48-ADF9-3F12AC24BC68"--random package id
		files {
				"./bin/FShaders/*.frag",
				"./bin/VShaders/*.vert",
				"./bin/GShaders/*.geo"
			}

			
	--GPUCV-Highguig
	CreateProject("../../", "highguig",	"SharedLib", "plugin", "_GPUCV_HIGHGUIG_DLL",
		{gpucv_core_list, "cxcoreg"})	
		package.guid = "BFB29282-C75E-8F48-ADF6-3F12AC24BC68"--random package id
					

	--GPUCV-cvg
	CreateProject("../../", "cvg",	"SharedLib", "plugin", "_GPUCV_CVG_DLL",
		{gpucv_core_list, "cxcoreg", "highguig"})
		package.guid = "BFB29282-C75E-8F48-ADF7-3F12AC24BC68"--random package id
		files {
				"./bin/FShaders/*.frag",
				"./bin/VShaders/*.vert",
				"./bin/GShaders/*.geo"
			}

	--GPUCV-cvauxg
	CreateProject("../../", "cvauxg",	"SharedLib", "plugin", "_GPUCV_CVAUXG_DLL",
		{gpucv_core_list, "cxcoreg", "highguig", "cvg"})
		package.guid = "BFB29282-C75E-8F48-ADF5-3F12AC24BC68"--random package id
		files {
				"./bin/FShaders/*.frag",
				"./bin/VShaders/*.vert",
				"./bin/GShaders/*.geo"
			}

	gpucv_opengl_list= { "cxcoreg", "cvg", "cvauxg", "highguig"}

end--_ACTION
Debug("Include 02-gpucv_opengl.lua...done")