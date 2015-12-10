Debug("Include packaging.premake4.lua")
--============================================================
-- ACTION: setenv
--============================================================
newaction {
   trigger     = "getpath",
   description = "GpuCV(deprecated): do not use any more.", 
   --print the command to set the right environnement variables (LD_LIBRARY_PATH/PATH).",
   execute     = function ()
		error "'getpath' action is not used any more, refer to local documentation or online manual at https://picoforge.int-evry.fr/projects/svn/gpucv//gpucv_doc/1.0.0/EXAMPLE__LIST__PAGE.html."
		
		
		printf("Run one of the following PATH command to export the required library path to run GpuCV applications:")
		
		if( _OPTIONS["os"] =="linux") then
			printf(" * LINUX & MacOSX: \n\t export LD_LIBRARY_PATH=$(pwd)/lib/linux-gmake/:$(pwd)/dependencies/SugoiTools/lib/linux-gmake/:$CUDA_LIB_PATH:$CUDA_LIB_PATH/../lib64/:/usr/local/lib/:$LD_LIBRARY_PATH")
		end
		
		
		if( _OPTIONS["os"] =="windows") then
		
			-- %% is equivalent to % in LUA
			windows_PATH = "%%CD%%\\dependencies\\otherlibs\\bin\\;"
			windows_release_PATH=""
			
			windows_path_x32 = "%%CD%%\\dependencies\\otherlibs\\bin\\x32;%%CD%%\\dependencies\\SugoiTools\\dependencies\\bin\\x32;"
			windows_path_x64 = "%%CD%%\\dependencies\\otherlibs\\bin\\x64;%%CD%%\\dependencies\\SugoiTools\\dependencies\\bin\\x64;"
			if os.getenv("OPENCV_PATH") then
				windows_path_x32 = windows_path_x32.."%%OPENCV_PATH%%\\bin\\;"
				windows_path_x64 = windows_path_x64.."%%OPENCV_PATH%%\\bin64\\;"
			end
			if cuda_enable==1 then
				windows_path_x32 = windows_path_x32.."%%CUDA_BIN_PATH%%;%%NVSDKCOMPUTE_ROOT%%\\C\\bin\\win32\\Release\\;"
				windows_path_x64 = windows_path_x64.."%%CUDA_BIN_PATH%%;%%NVSDKCOMPUTE_ROOT%%\\C\\bin\\win64\\Release\\;"
			end
			
			
			if (cuda_npp_enable==1) then
					windows_PATH = windows_PATH .."%%NPP_SDK_PATH%%\\common\\lib\\;" 
			end


			printf("\n * MS Windows + Visual Studio 2005:")
			printf("\tRelease x32:\nset PATH=%%PATH%%;"..windows_PATH..";"..windows_release_PATH..";"..windows_path_x32..";%%CD%%\\lib\\windows-vs2005\\;%%CD%%\\dependencies\\SugoiTools\\lib\\windows-vs2005\\;")
			printf("\n\tRelease x64:\nset PATH=%%PATH%%;"..windows_PATH..";"..windows_release_PATH..";"..windows_path_x64..";%%CD%%\\lib\\windows-vs2005\\;%%CD%%\\dependencies\\SugoiTools\\lib\\windows-vs2005\\;")
			
			printf("\n\n * MS Windows + Visual Studio 2008:")
			printf("\tRelease x32:\nset PATH=%%PATH%%;"..windows_PATH..";"..windows_release_PATH..";"..windows_path_x32..";%%CD%%\\lib\\windows-vs2008\\;%%CD%%\\dependencies\\SugoiTools\\lib\\windows-vs2008\\;")
			printf("\n\tRelease x64:\nset PATH=%%PATH%%;"..windows_PATH..";"..windows_release_PATH..";"..windows_path_x64..";%%CD%%\\lib\\windows-vs2008\\;%%CD%%\\dependencies\\SugoiTools\\lib\\windows-vs2008\\;")
			
			if nsight_path then
				printf("\n\tDebug with NVIDIA NSIGHT:\nset PATH=%%PATH%%;"..windows_PATH..";"..windows_release_PATH..";"..windows_path_x64..";"..windows_path_x32..";%%CD%%\\lib\\windows-vs2008-nsight\\;%%CD%%\\dependencies\\SugoiTools\\lib\\windows-vs2008\\;")
			end
		end
	end
}
--============================================================
-- ACTION: makedoc -> generate doxygen documentation
--============================================================
newaction {
   trigger     = "makedoc",
   description = "GpuCV: Generate doxygen documentation (doxygen must be installed)",
   execute     = function ()
		cmd= ""
		if _OPTIONS["os"] == "windows" then
			cmd= "cd doc & doxygen GpuCv.doxy & cd .."
		else
			cmd="cd doc; doxygen GpuCv.doxy; cd .."
		end
		Debug("makedoc -> "..cmd)
		if (os.execute(cmd)) then
			error "Error while making documentation"
		end
	end
}
--============================================================

--============================================================
-- ACTION: clean -> 
--============================================================
newaction {
   trigger     = "clean",
   description = "GpuCV: clean source code and temp files",
   execute     = function ()
				printf("cleaning project")
				--check that we are in correct folder
				lua_file_matches = os.matchfiles("premake4.lua")
				if not os.matchfiles("premake4.lua") then
					printf("Wrong path")
					return
				end
				--remove obj folder
				--remove lib folder
				--remove html folder
				if _OPTIONS["os"] == "windows" then
					os.execute("del obj\\* /S /F /Q")
					os.execute("rmdir obj\\* /S /Q")
					os.execute("del lib	/S /F /Q")
					os.execute("rmdir lib /S /Q")
					os.execute("del doc\\html /S /F /Q")
					--clean sugoitools folders
					os.execute("del dependencies\\SugoiTools\\obj\\* /S /F /Q")
					os.execute("rmdir dependencies\\SugoiTools\\obj\\* /S /Q")
				else
					os.execute("rm obj -R -f")
					os.execute("rm lib -R -f")
					os.execute("rm doc/html/* -R -f")
					--clean sugoitools folders
					os.execute("rm dependencies/SugoiTools/obj -R -f")
				end
	end
}
--============================================================

--============================================================
-- ACTION: ZIP -> 
--============================================================
AddOption("zip-name", "Custom: set zip path and filename, library version can be added to the name with option --zip-version. Used with action ZIP")
AddOption("zip-source", "Custom: set source path to zip, ex: '--zip-source=../ProjectName'. Used with action ZIP.") 


newaction {
   trigger     = "zip",
   description = "Custom: create a zip archive of current project. Arguments zip-name/zip-version/zip-source must be passed.",
   execute     = function ()
				printf("zipping project")
				--check that we are in correct folder
				lua_file_matches = os.matchfiles("premake4.lua")
				if not os.matchfiles("premake4.lua") then
					error "Wrong path"
				end
				
				if not _OPTIONS["zip-source"] then
					error "Option 'zip-source' must be passed as argument"
				end
				
				if not _OPTIONS["zip-name"] then
					error "Option 'zip-name' must be passed as argument"
				end
				
				
				filename=""
				if _OPTIONS["zip-name"] then
					filename = _OPTIONS["zip-name"] 	 
				end
				--filename = filename ..".zip", not used anymore, file type depend on the OS
				
				return zip(_OPTIONS["zip-source"], filename)
	end
}--============================================================
-- Add a new action to check which depenencies have been found in the system/
newaction {
   trigger     = "checkdep",
   description = "GpuCV: (EXPERIMENTAL) Check for dependencies paths on the system",
   --execute     = function ()
   --   os.execute("bin/debug/myprogram", "/usr/local/bin/myprogram")
   --end
}

-- Add a new action to import external DLLs to local lib dir, so we do not need to set the path to execute and test applications (under windows)
newaction {
   trigger     = "importlibs",
   description = "GpuCV(Windows only): Import dependencies' libraries (OpenCV/CUDA...) into the './gpucv/lib/$TARGET/$ARCH/$CONFIG/' paths, so GpuCV can run on any computer without additional packages.",
   execute     = function ()
   
		Debug("Parse all configurations and platforms:")			
		for iConf, strConf in ipairs(solutiontProperties.configurations) do
			--for iPlat, strPlat in ipairs(solutiontProperties.platforms) do
					Debug("Current Configuration: " .. strConf)

					dstFolder32 = "%CD%\\lib\\windows-vs2008\\x32\\"..strConf
					dstFolder64 = "%CD%\\lib\\windows-vs2008\\x64\\"..strConf
					
					Verbose("Copy OpenCV files x32")
					Verbose("Copy cmd, from:" .. "%%OPENCV_PATH%%\\bin\\*" .. " \nto\n" .. dstFolder32)
					cmd = "copy \"%OPENCV_PATH%\\bin\\*.dll\" \""..dstFolder32.."\""
					os.execute("echo "..cmd)
					os.execute(cmd)
					Verbose("Copy OpenCV files x64")
					os.execute("copy \"%OPENCV_PATH%\\bin64\\*.dll\" \""..dstFolder64.."\"")

					
					Verbose("Copy CUDA files x32")
					os.execute("copy \"%CUDA_BIN_PATH%\\..\\bin\\*.dll\"  \""..dstFolder32.."\"")
					Verbose("Copy CUDA files x64")
					os.execute("copy \"%CUDA_BIN_PATH%\\..\\bin\\*.dll\"  \""..dstFolder64.."\"")
		
		
					Verbose("Copy NVSDKCompute files x32")
					os.execute("copy \"%NVSDKCOMPUTE_ROOT%\\C\\bin\\win32\\Release\\*.dll\"  \""..dstFolder32.."\"")
					Verbose("Copy NVSDKCompute files x64")
					os.execute("copy \"%NVSDKCOMPUTE_ROOT%\\C\\bin\\win64\\Release\\*.dll\"  \""..dstFolder64.."\"")
					
					
					if (cuda_npp_enable==1) then
						Verbose("Copy NPP files x32/64")
						os.execute("copy \"%NPP_SDK_PATH%\\common\\lib\\npp32*.dll\"  \""..dstFolder32.."\"")
						os.execute("copy \"%NPP_SDK_PATH%\\common\\lib\\npp64*.dll\"  \""..dstFolder64.."\"")
					end
		
		
					--sugoitools & dependencies must be copied after NVIDIA files
					Verbose("Copy SugoiTools files")
					os.execute("copy \"dependencies\\SugoiTools\\lib\\windows-vs2008\\x32\\"..strConf.. "\\*.dll\" \""..dstFolder32.."\"")
					os.execute("copy \"dependencies\\SugoiTools\\lib\\windows-vs2008\\x64\\"..strConf.. "\\*.dll\" \""..dstFolder64.."\"")
					
					Verbose("Copy SugoiTools dependencies")
					os.execute("copy \"dependencies\\SugoiTools\\dependencies\\bin\\x32\\*.dll\" "..dstFolder32.."\"")
					os.execute("copy \"dependencies\\SugoiTools\\dependencies\\bin\\x64\\*.dll\" "..dstFolder64.."\"")
					
					
					Verbose("Removing unecessary files")
					os.execute("del \""..dstFolder32.."\\cublas64*\"")
					os.execute("del \""..dstFolder64.."\\cublas32*\"")
					os.execute("del \""..dstFolder32.."\\cufft64*\"")
					os.execute("del \""..dstFolder64.."\\cufft32*\"")
					os.execute("del \""..dstFolder32.."\\cudart64*\"")
					os.execute("del \""..dstFolder64.."\\cudart32*\"")
					--os.execute("del \""..dstFolder32.."\\cufft*\"")
					--os.execute("del \""..dstFolder64.."\\cufft*\"")
		end
		
   end
}




-- Add a new action to import external DLLs to local lib dir, so we do not need to set the path to execute and test applications (under windows)
newaction {
   trigger     = "winbin_zip",
   description = "GpuCV: create the windows binary zip file containing the libraries/sample applications, the data files and documentation. '--platform' option must be specified",
   execute     = function ()
		if not _OPTIONS["platform"] then
			error "'--platform=[x32|x64]' option must be specified."
		end
		if not ( _OPTIONS["os"] =="windows") then
			error "'winbin_zip' action is a windows only command."
		end
		if not _OPTIONS["zip-name"] then
			error "Option 'zip-name' must be passed as argument"
		end
		
		dstFolder_root = "%CD%\\tmp\\".._OPTIONS["zip-name"].."\\"
		
		--clean destination folders
		dstFolder = dstFolder_root
		Debug("Clean export folder"..dstFolder)	
		os.execute("rmdir /S /Q "..dstFolder)
		
		--create destination folders
		dstFolder = dstFolder_root
		Debug("Create tmp folder: "..dstFolder)	
		os.execute("mkdir "..dstFolder)
		
		dstFolder = dstFolder_root
		Debug("Create export folder"..dstFolder)	
		os.execute("mkdir "..dstFolder)

		if 1 then
			--copy bin files
			dstFolder = dstFolder_root.."\\bin\\"
			Debug("Copy bin files: "..dstFolder)
			os.execute("xcopy /S %CD%\\lib\\windows-vs2008\\".._OPTIONS["platform"].."\\Release\\*.dll "..dstFolder)
			os.execute("xcopy /S %CD%\\lib\\windows-vs2008\\".._OPTIONS["platform"].."\\Release\\*.exe "..dstFolder)
			
			--copy lib files
			dstFolder = dstFolder_root.."\\lib\\"
			Debug("Copy lib files: "..dstFolder)
			os.execute("xcopy /S %CD%\\lib\\windows-vs2008\\".._OPTIONS["platform"].."\\Release\\*.lib "..dstFolder)
			
			
			--copy include files
			dstFolder = dstFolder_root.."\\include\\"
			Debug("Copy include files: "..dstFolder)
			os.execute("echo xcopy /S %CD%\\include\* "..dstFolder)
				
			--copy doc files
			dstFolder = dstFolder_root.."\\doc\\"
			Debug("Create export folder"..dstFolder)
			os.execute("xcopy /S %CD%\\doc\\html\\* "..dstFolder)
			
		end 
		
		--copy data files
		dstFolder = dstFolder_root.."\\data\\"
		Debug("Copy data files: "..dstFolder)
		os.execute("echo xcopy /S %CD%\\data\\ "..dstFolder)
		os.execute("xcopy /S %CD%\\data\\* "..dstFolder)
		
		--copy licence files
		dstFolder = dstFolder_root.."\\"
		Debug("Copy license files: "..dstFolder)
		os.execute("echo xcopy /S %CD%\\License*.* "..dstFolder)
		
		--zip file
		printf("zipping windows binary project")
		--check that we are in correct folder
				
		filename=""
		if _OPTIONS["zip-name"] then
			filename = _OPTIONS["zip-name"]
		end
		--filename = filename ..".zip", not used anymore, file type depend on the OS
		dstFolder = dstFolder_root
		return zip(dstFolder, dstFolder_root.."..\\"..filename)
   end
}

Debug("Include packaging.premake4.lua...done")
