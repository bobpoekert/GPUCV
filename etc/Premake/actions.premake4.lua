Debug("Include action.premake4.lua")

function CompileDepencies(COMPILER_CMD)
	--compile sugoitools
	print ""
	print ""
	print("_________________________________________________________________________________________")
	print ("Compiling dependencies:")
	print("_________________________________________________________________________________________")
		
			if _OPTIONS["os"]=="windows" then
				dep_compile_cmd = "cd dependencies\\SugoiTools\\ & premake4.exe "
			else
				dep_compile_cmd = "cd dependencies/SugoiTools/ ; ./premake4.linux "
			end
			
			if _OPTIONS["verbose"] then
				dep_compile_cmd = dep_compile_cmd.." --verbose "
			end
			if _OPTIONS["debug"] then
				dep_compile_cmd = dep_compile_cmd.." --debug "
			end
		
			
			--generate projects
			if _ACTION=="compile_vs2008" then
				dep_current_cmd = dep_compile_cmd.. " vs2008"
			end
			
			if _ACTION=="compile_vs2005" then
				dep_current_cmd = dep_compile_cmd.. " vs2005"
			end
			
			if _ACTION=="compile_gmake" then
				dep_current_cmd = dep_compile_cmd.. " gmake"
			end
			Debug("Generating dependencies projects:")
			Debug("cmd: "..dep_current_cmd)
			print("Generating dependencies...")
			if not (os.execute(dep_current_cmd)==0) then
				error "could not generate dependencies/SugoiTools: "--..testapp_list[i]
			end
			-------------------
			
			--compile it
			if _OPTIONS["platform"] then
				dep_compile_cmd = dep_compile_cmd.." --platform=".._OPTIONS["platform"]
			end
			
			dep_current_cmd=dep_compile_cmd.." "..COMPILER_CMD
			Debug("Compiling dependencies:")
			Debug("cmd: "..dep_current_cmd)
			print("Compiling dependencies...")
			if not (os.execute(dep_current_cmd)==0) then
				error "could not compile dependencies/SugoiTools: "--..testapp_list[i]
			end
			print("Compiling dependencies done")
	
	print("=========================================================================================")
	print("Compiling dependencies done")
	print("=========================================================================================")
	print ""
	print ""	

end

--return host CPU architecture x32/x64, still in test
function GetHostArchitecture()
	if _OPTIONS["os"]=="windows" then
		win_arch = os.getenv("PROCESSOR_ARCHITECTURE")
		if win_arch then
			if win_arch=="x86" then
				return "x32"
			else
				return {"x32", "x64"}
			end
		end
	elseif _OPTIONS["os"]=="linux" then
	
	elseif _OPTIONS["os"]=="macosx" then
	
	end
end
--============================================================
-- ACTION: compile_vs2008
--============================================================
newaction {
   trigger     = "compile_vs2008",
   description = "GpuCV: Launch compiling with VS 2008(compile SugoiTools dependencies as well)",
   execute     = function ()
		if _OPTIONS["os"] == "windows" then
			--compile only X32, x64 or all
			if _OPTIONS["platform"]=="x32" then
				arch_list={"x86"}
				Debug("Platform x32 only")
			elseif _OPTIONS["platform"]=="x64" then
				arch_list = {"amd64"}
				Debug("Platform x64 only")
			else
				arch_list = {"x86", "amd64"}
				Debug("Platform x32/x64")
			end 
			mode_list = {"debug", "release"}
			--compile sugoitools
			CompileDepencies(_ACTION)
			--compile gpucv
			for j,arch in ipairs(arch_list) do
				for i,mode in ipairs(mode_list) do
					compileVS2008("GpuCV", "build\\windows-vs2008\\GpuCV.sln", mode, arch)
				end
			end
		end
	end
}
--============================================================

--============================================================
-- ACTION: compile_vs2005
--============================================================
newaction {
   trigger     = "compile_vs2005",
   description = "GpuCV: Launch compiling with VS 2005",
   execute     = function ()
		if _OPTIONS["os"] == "windows" then
			--compile only X32, x64 or all
			if _OPTIONS["platform"]=="x32" then
				arch_list={"x86"}
				Debug("Platform x32 only")
			elseif _OPTIONS["platform"]=="x64" then
				arch_list = {"amd64"}
				Debug("Platform x64 only")
			else
				arch_list = {"x86", "amd64"}
				Debug("Platform x32/x64")
			end 
			mode_list = {"debug", "release"}
			
			--compile sugoitools
			CompileDepencies(_ACTION)
			--compile gpucv
			for j,arch in ipairs(arch_list) do
				for i,mode in ipairs(mode_list) do
					compileVS2005("GpuCV", "build\\windows-vs2005\\GpuCV.sln", mode, arch)
				end
			end
		end

	end
}
--============================================================

--============================================================
-- ACTION: compile_gmake
--============================================================
newaction {
   trigger     = "compile_gmake",
   description = "GpuCV: Launch compiling with gmake",
   execute     = function ()
		if not (_OPTIONS["os"] == "windows") then
			--check dir
			if not os.isdir("tmp") then
				os.execute("mkdir tmp")
			else
				os.execute('rm tmp/build_*')
			end
			
			if os.isdir("lib") then
				os.execute('rm lib/'.._OPTIONS["os"]..'-gmake/*')
			end
			--check config
			config_list = {"debug32", "release32", "debug64", "release64"}
            testapp_list = {"GPUCVConsole.x86_32d", "GPUCVConsole.x86_32", "GPUCVConsole.x86_64d", "GPUCVConsole.x86_64"}

			if _OPTIONS["platform"]=="x32" then
				config_list={"debug32","release32"}
				testapp_list= {"GPUCVConsole.x86_32d", "GPUCVConsole.x86_32"}
			end
			if  _OPTIONS["platform"]=="x64" then
				config_list = {"debug64", "release64"}
				testapp_list = {"GPUCVConsole.x86_64", "GPUCVConsole.x86_64"}
			end 
			
			--compile sugoitools
			CompileDepencies(_ACTION)
			--compile gpucv
			for i,config in ipairs(config_list) do
				compileGmake("GpuCV", "build//".._OPTIONS["os"].."-gmake//", config)
				--on LINUX, makefile does not stop on the first error...no way to stop the process...
				--cmd = "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:lib/linux-gmake/;lib/linux-gmake/"..testapp_list[i].." -q"
				--print("Test command:\n"..cmd)
				--if not (os.execute(cmd)==0) then
				--	error "could not execute test application: "--..testapp_list[i]
				--end	
			end
		end
	end
}
--============================================================
Debug("Include action.premake4.lua...done")
