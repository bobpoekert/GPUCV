if _OPTIONS["verbose"] then
	verbose=1
else
	verbose=0
end
if _OPTIONS["debug"] then
	verbose=2
end
--import premake function from sugoitools
dofile("./dependencies/SugoiTools/etc/Premake/compiling.premake4.lua")
dofile("./dependencies/SugoiTools/etc/Premake/functions.premake4.lua")


--============================================================
--Definition of global options, plugins can introduce new options:
--============================================================
AddOption("path-inc", "GpuCV: Additionnal include paths.")
AddOption("path-lib", "GpuCV: Additionnal lib paths.")
AddOption("path-obj", "GpuCV: Destination temporary objects directory, useful to accelerate compiling with with Ramdisk drives!")
AddOption("use-vtune", "GpuCV: Add custom flags to profile the application with VTUNE.")
--AddOption("libversion", "GpuCV: set the GpuCV version that will be used in filenames such as the zip files.")
AddOption("verbose", "GpuCV: disable logging compilation message to file and show additionnal verbose output.")
AddOption("debug", "GpuCV: show additionnal debug information from the premake scrips.")
--AddOption("zip-name", "GpuCV: give zip file name used by action 'zip', default is 'gpucv.zip'.")
--cuda
AddOption("plug-cuda", "GpuCV: Enable nVidia CUDA plugin.")
AddOption("plug-cuda-off", "GpuCV: Disable nVidia CUDA plugin.")
--AddOption("plug-cudpp", "GpuCV: Use CUDPP library (require option plug-cuda)")
--cudpp is disabled for now, compiling issues under linux: relocation R_X86_64_32S against `vtable for CUDPPPlan'
AddOption("plug-npp", "GpuCV: Use NVIDIA NPP library (require option plug-cuda)")
AddOption("use-switch", "GpuCV: Enable generation of switch plugin mechanism (Default)")
AddOption("use-switch-off", "GpuCV: Disable generation of switch plugin mechanism")
SWITCH_DEFINES=""


--check os, OS can be forced to a different target.
printf("Checking Environement:")
if not _OPTIONS["os"] then
	_OPTIONS["os"]=GetHostOS()
end

--check environement variables
--opencv. OPENCV_PATH is checked at project creation time, not when running batch commands

if _OPTIONS["os"]=="windows" then
	if not os.getenv("OPENCV_PATH") then
		print "Warning: OPENCV_PATH: environement variable should be defined."
	else
		printf("OPENCV_PATH: OK.")
	end
	if not os.getenv("OPENCV_VERSION") then
		print "Warning: OPENCV_VERSION  environement variable should be defined => '210' for v2.1.0. or '110' for v1.1.0, he version is used as library suffixe."
	else
		printf("OPENCV_VERSION: OK.")
	end
end
--cuda
cuda_enable=0
cuda_npp_enable=0
if not os.getenv("CUDA_INC_PATH") then
	printf("CUDA_INC_PATH: environement variable not found.")
else
	cuda_enable=cuda_enable+1
	printf("CUDA_INC_PATH: OK.")
end
if not os.getenv("CUDA_BIN_PATH") then
	printf("CUDA_BIN_PATH: environement variable not found.")
else
	cuda_enable=cuda_enable+1
	printf("CUDA_BIN_PATH: OK.")
end
if not os.getenv("CUDA_LIB_PATH") then
	printf("CUDA_LIB_PATH: environement variable not found.")
else
	cuda_enable=cuda_enable+1
	printf("CUDA_LIB_PATH: OK.")
end
if not os.getenv("CUDA_LIB64_PATH") then
	printf("Warning: CUDA_LIB64_PATH: environement variable not found. Please set it on X64 system.")
else
	printf("CUDA_LIB64_PATH: OK.")
end
if not os.getenv("NVSDKCOMPUTE_ROOT") then
	printf("NVSDKCOMPUTE_ROOT: environement variable not found..")
else
	cuda_enable=cuda_enable+1
	printf("NVSDKCOMPUTE_ROOT: OK.")
end
if not os.getenv("NPP_SDK_PATH") then
	printf("NPP_SDK_PATH: environement variable not found, disabling NPP plugin.")
else
	cuda_npp_enable=1
	printf("NPP_SDK_PATH: OK.")
end
if not os.getenv("NPP_SDK_PATH") then
	printf("NPP_SDK_PATH: environement variable not found, disabling NPP plugin.")
else
	cuda_npp_enable=1
	printf("NPP_SDK_PATH: OK.")
end
if not (cuda_enable>=3)then
	cuda_enable=0
	printf "WARNING:\t NVIDIA CUDA toolkit not setup on this computer. GpuCV CUDA plugin disabled. Force CUDA plugin generation using --plug-cuda option."
end
if _OPTIONS["plug-cuda-off"]then
	cuda_enable=0
	printf "WARNING:\t GpuCV CUDA plugin disabled."
end
--check for NVIDIA Parallel NSIGHT tool
local nsight_path
use_nsight=0
if _OPTIONS["os"] == "windows" then
	--windows only yet...
	nsight_path = checkSysPath("NVIDIA Parallel Nsight", {"C:\\Program Files\\NVIDIA Parallel Nsight 1.0"
											,"C:\\Program Files\\NVIDIA Parallel Nsight 1.0"
		} )
	if(nsight_path) then
		if not (nsight_path=="") then
			printf("NVIDIA Parallel Sight found in %s", nsight_path)
			printf("-> Enabling compilation for NVIDIA Parallel Sight")
			use_nsight = 1
		end
	end
end
	-- cuda done




    ---------------Properties-----------
	solutiontProperties = {
						configurations = {"Debug", "Release" }
						,platforms={  "x32", "x64" }
						,name="GpuCV"
					}
	if nsight_path then
		table.insert(solutiontProperties.configurations, {  "Debug_Nsight", "Release_Nsight" })
	end
		
	---------------SUFFIXES--------------
	defaultSuffixes = {
						dir=""
						,arch={}
						,version=""
						}
	defaultSuffixes.arch["x32"] = "32"
	defaultSuffixes.arch["x64"] = "64"
	defaultSuffixes.arch["Debug"] = "D"
	defaultSuffixes.arch["Release"] = ""
	
	if _ACTION then
		defaultSuffixes.dir=_OPTIONS["os"].."-".._ACTION
	else
		defaultSuffixes.dir=_OPTIONS["os"]
	end

	if _OPTIONS["os"]=="macosx" then
		defaultSuffixes.version="10"
	end
	
	Verbose("---------------SUFFIXES--------------");
	Verbose("dir:\t"..defaultSuffixes.dir);
	--Verbose("ARCH:\t"..defaultSuffixes.arch);
	Verbose("VERSION:\t"..defaultSuffixes.version);
	Verbose("---------------SUFFIXES--------------\n");
	---------------SUFFIXES--------------


	---------------PATHS--------------	
	defaultProjectPaths = {
					projects="build/".. defaultSuffixes.dir
					,bin="bin/".. defaultSuffixes.dir
					,obj="obj/".. defaultSuffixes.dir
					,lib="lib/".. defaultSuffixes.dir
					,lib_src="src/lib"
					,lib_header="include"
					,app_src="src/example"
					,plug_src="src/plugin"
					,test_src="src/test"
					,doc="doc"
					,dependencies="dependencies/"
					}
	--custom option
	if _OPTIONS["path-obj"] then
		defaultProjectPaths.obj = _OPTIONS["path-obj"].."/"..defaultSuffixes.dir.."/"
	end
	if _OPTIONS["path-inc"] then
		table.insert(defaultProjectPaths.lib_header, _OPTIONS["path-inc"])
	end
	if _OPTIONS["path-lib"] then
		defaultProjectPaths.lib= _OPTIONS["path-lib"]
	end
	
	Verbose("---------------PATHS--------------");
	Verbose("projects:\t"..defaultProjectPaths.projects);
	Verbose("bin:\t"..defaultProjectPaths.bin);
	Verbose("obj:\t"..defaultProjectPaths.obj);
	Verbose("lib:\t"..defaultProjectPaths.lib);
	Verbose("lib_src:\t"..defaultProjectPaths.lib_src);
	Verbose("lib_header:\t"..defaultProjectPaths.lib_header);
	Verbose("app_src:\t"..defaultProjectPaths.app_src);
	Verbose("test_src:\t"..defaultProjectPaths.test_src);
	Verbose("doc:\t"..defaultProjectPaths.doc);
	Verbose("doc:\t"..defaultProjectPaths.dependencies);
	Verbose("---------------PATHS--------------\n");	
	---------------PATHS--------------


---------------denpendencies--------------
--used in files dependency files...
---------------denpendencies--------------
if not _ACTION or (_ACTION=="checkdep") then -- we parse all files but do not build anything
	lua_files_matches = os.matchfiles("**.lua")     -- recursive match
	Verbose("Parsing all LUA files for options...")
	for i,lua_file in ipairs(lua_files_matches) do
		if not string.startswith(lua_file, "dependencies", 0) then
			if not string.endswith(lua_file, "~", 0) then
				if not (lua_file=="Premake.lua") then --exclude previous premake version script
					if not (lua_file=="premake4.lua") then --exclude this file
						Verbose("\tParsing %s", lua_file)
						dofile(lua_file)
					end
				end
			end
		end
	end
	Verbose("Parsing done!")
end
dofile("./etc/Premake/packaging.premake4.lua")
dofile("./etc/Premake/actions.premake4.lua")
--============================================================





if _ACTION  and ((_ACTION=="vs2008") or (_ACTION=="vs2005") or (_ACTION=="gmake") or (_ACTION=="xcode3")) then
	--check thqt opencv path is set
	if _OPTIONS["os"]=="windows" then
		if not os.getenv("OPENCV_PATH") then
			error "OPENCV_PATH: environement variable should be defined."
		else
			printf("OPENCV_PATH: OK.")
		end
		if not os.getenv("OPENCV_VERSION") then
			error "OPENCV_VERSION  environement variable should be defined => '210' for v2.1.0. or '110' for v1.1.0, he version is used as library suffixe."
		else
			printf("OPENCV_VERSION: OK.")
		end
	end

	-- A solution contains projects, and defines the available configurations
	solution (solutiontProperties.name)
		configurations 	(solutiontProperties.configurations)
		location 		(defaultProjectPaths.projects)
		platforms 		(solutiontProperties.platforms)
		objdir			(defaultProjectPaths.obj)
		
		--update target dir for Nsight configurations
		configuration "*"
		targetdir ( defaultProjectPaths.lib)
		if nsight_path then
			configuration {  "*_Nsight"}
				targetdir (defaultProjectPaths.lib.."-Nsight")
		end
	
			
		
	function CreateProject(projectPath, projectName, projectType, projectCategory, optionalDefine, projectDependencies)
		printf("Creating project '%s'.", projectName) 
   		-- A project defines one build target
			CreateProjectHeader(projectName, projectType, "C++", projectCategory)
	
			if projectPath == "" then
				location (defaultProjectPaths.projects.."/"..projectCategory)
			else
				location (projectPath..defaultProjectPaths.projects.."/"..projectCategory)
			end
			
	   		if projectType =="SharedLib" then
	   		--	targetdir = defaultProjectPaths.lib
				if(projectCategory == "lib") then
					--file included
					  files {
						projectPath..defaultProjectPaths.lib_src.."/"..(projectName).."/**.cpp"
						,projectPath..defaultProjectPaths.lib_src.."/"..(projectName).."/**.c"
						,projectPath..defaultProjectPaths.lib_src.."/"..(projectName).."/**.cu"
						,projectPath..defaultProjectPaths.lib_src.."/"..(projectName).."/**.h"
						,projectPath..defaultProjectPaths.lib_header.."/"..(projectName).."/**.h"
						,projectPath..defaultProjectPaths.lib_header.."/"..(projectName).."/**.hpp"
						,projectPath..defaultProjectPaths.lib_header.."/"..(projectName).."/**.cu"
						,projectPath..defaultProjectPaths.doc.."/"..(projectName).."/**.dox"
						,projectPath..defaultProjectPaths.doc.."/"..(projectName).."/**.dox"
						--test APP
						,projectPath..defaultProjectPaths.test_src.."/"..(projectName).."/**.*"
						}
						includedirs ("../../"..defaultProjectPaths.lib_header.."/"..(projectName))
				elseif  projectCategory =="plugin" then
				--	targetdir = defaultProjectPaths.lib
					Debug("Including project " ..projectName .." files from path:" ..projectPath..defaultProjectPaths.plug_src)
					--file included
					  files {
						projectPath..defaultProjectPaths.plug_src.."/"..(projectName).."/**.cpp"
						,projectPath..defaultProjectPaths.plug_src.."/"..(projectName).."/**.c"
						,projectPath..defaultProjectPaths.plug_src.."/"..(projectName).."/**.cu"
						,projectPath..defaultProjectPaths.plug_src.."/"..(projectName).."/**.h"
						,projectPath..defaultProjectPaths.plug_src.."/"..(projectName).."/**.hpp"
						,projectPath..defaultProjectPaths.doc.."/"..(projectName).."/**.dox"
						,projectPath..defaultProjectPaths.plug_src.."/"..(projectName).."/**.dox"
						}
					
					includedirs ("../../"..defaultProjectPaths.plug_src.."/"..(projectName))
				end
				includedirs (projectPath..defaultProjectPaths.lib_src.."/"..(projectName))
			else--Application
	  			--flags { "NoMain" }
			--	targetdir = defaultProjectPaths.bin
	  			Debug("Application source path: " ..projectPath..defaultProjectPaths.app_src.."/"..projectName.."/*.*")
	  			files {projectPath..defaultProjectPaths.app_src.."/"..projectName.."/*.*"}  
				includedirs (projectPath..defaultProjectPaths.app_src.."/"..projectName)
	  		end
			excludes { "**~" }
				
			--Open OS specific files
				--dofile("../../etc/Premake/".._OPTIONS["os"]..".premake4.lua")
				dofile("../../dependencies/SugoiTools/etc/Premake/".._OPTIONS["os"]..".premake4.lua")
				
				
			--target dir:
			for iConf, strConf in ipairs(project().solution.configurations) do
				for iPlat, strPlat in ipairs(project().solution.platforms) do
					configuration {strConf, strPlat}
						targetdir ("../../"..defaultProjectPaths.lib.."/"..strPlat.."/"..strConf.."/")
				end
			end
					
			--BINARY TYPE
			configuration "SharedLib"
				defines "_USRDLL"
			configuration "ConsoleApp"
				defines "_EXE"
			
		    --DEFINTITIONS
			configuration "*"
				defines (optionalDefine)
				
			configuration "use-switch"
				defines "_GPUCV_SUPPORT_SWITCH"
			--LOAD DEPENDENCIES
				--generate list of dependencies
				premakeDependenciesPath = {projectPath.."dependencies/SugoiTools/etc/Premake",
											projectPath.."etc/Premake"}
				--load them
				LoadDependencies(projectDependencies, premakeDependenciesPath)

			--DEBUG & RELEASE setting are set int ./etc/Premake/main.premake.lua
			
			
		configuration "*"
			includedirs {
					"../../"..defaultProjectPaths.lib_header
					,"../../"..defaultProjectPaths.dependencies.."/include/"
					,"../../"..defaultProjectPaths.lib_src
					,"../../"..defaultProjectPaths.lib_src.."/../plugin/"
				}
			libdirs ("../../"..defaultProjectPaths.lib)
				
		configuration "windows"
			includedirs	{
					"../../"..defaultProjectPaths.dependencies.."/include/pthread"
				}	
	end


	--TODO: now the script with parse following folders:
	---			./src/lib
	---			./src/plugins
	---			./src/example

	Debug("Parsing project files")
	if false then--automatic parsing
		lua_files_matches = os.matchfiles("src/**.lua")     -- recursive match
		printf("Parsing all LUA files for projects...")
		for i,lua_file in ipairs(lua_files_matches) do
			if not string.endswith(lua_file, "~", 0) then
				if not (lua_file=="Premake.lua") then --exclude previous premake version script
					if not (lua_file=="Premake4.lua") then --exclude this file
						printf("\tParsing %s", lua_file)
						dofile(lua_file)
					end
				end
			end
		end
		printf("Parsing done!")	
	else
		dofile("./src/lib/01-gpucv_core.lua")		
		dofile("./src/plugin/02-gpucv_opengl-plugin.lua")
		dofile("./src/lib/gpucv_cuda.lua")
		dofile("./src/plugin/03-gpucv_cuda-plugin.lua")
		dofile("./src/lib/gpucv_switch.lua")
		dofile("./src/example/gpucv_example.lua")
	end
end
