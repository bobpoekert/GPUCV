Debug("Include sugoitools.premake4.lua")

AddOption("path-sugoitools", "GpuCV: Specify the path of SugoiTools libraries. Default assumes SugoiTools is in ./gpucv/../resources/")

if _ACTION  and not (_ACTION=="checkdep") then

	configuration {"*"}
		
		
	sugoiToolsSuffixes = {}
	sugoiToolsSuffixes["x32"] = "32"
	sugoiToolsSuffixes["x64"] = "64"
	sugoiToolsSuffixes["Debug"] = "D"
	sugoiToolsSuffixes["Release"] = ""
	sugoiToolsLocalPath = "../../dependencies/SugoiTools"
	
	
	
--for all set of configurations and platforms:
	Debug("Parse all configurations and platforms:")			
	for iConf, strConf in ipairs(project().solution.configurations) do
		for iPlat, strPlat in ipairs(project().solution.platforms) do
				Debug("Current Configuration: " .. strConf .. "/" .. strPlat)
				configuration {strConf, strPlat}
				
					defines ("_SG_TLS_SUPPORT_GL")--enable openGL support in sugoitools
				
				--include sugoitools include/libs
					libdirs {sugoiToolsLocalPath.."/lib/".._OPTIONS["os"].."-".._ACTION.."/"..strPlat.."/"..strConf.."/"}
					includedirs {sugoiToolsLocalPath.."/include"}
					links { "SugoiTools"..sugoiToolsSuffixes[strPlat]..sugoiToolsSuffixes[strConf],
							"SugoiPThread"..sugoiToolsSuffixes[strPlat]..sugoiToolsSuffixes[strConf],
							"SugoiTracer"..sugoiToolsSuffixes[strPlat]..sugoiToolsSuffixes[strConf]}

					
				--include sugoitools/dependencies include/libs
					includedirs {sugoiToolsLocalPath.."/dependencies/include"}
					
				--windows specific
				configuration {strPlat, "windows"}
					libdirs {sugoiToolsLocalPath.."/dependencies/lib/"..strPlat.."/"}
					
				
				--linux specific
				configuration {strConf, strPlat, "linux"}
					prebuildcommands {"export ARCH=".. defaultSuffixes.arch[strPlat],
										"export GCV_BIN_SUFFIXE=".. defaultSuffixes.arch[strPlat]..defaultSuffixes.arch[strConf]
									}	
				--object dir
		end		
	end
	
	

end --_ACTION

Debug("Include sugoitools.premake4.lua...done")