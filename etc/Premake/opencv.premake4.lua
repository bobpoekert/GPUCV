Debug("Include opencv.premake4.lua")

--default values
opencv_version = "$(OPENCV_VERSION)"
opencv_path = "$(OPENCV_PATH)"

if verbose == 1 then
	printf("OpenCV version: %s", opencv_version)
end
		
if _ACTION  and not (_ACTION=="checkdep") then

--windows ONLY
	if _OPTIONS["os"]=="windows" then

	configuration "windows"
		if tonumber(os.getenv("OPENCV_VERSION")) >= 200 then
				includedirs {
						opencv_path.."/include/opencv"
						}
		else		
			includedirs{ 	
				opencv_path.."/cv/include"
				,opencv_path.."/cxcore/include"
				,opencv_path.."/otherlibs/highgui"
				,opencv_path.."/cvaux/include"
				,opencv_path.."/otherlibs/cvcam/include"
			}
		end
	configuration {"windows", "x32"}
		libdirs{opencv_path.."/lib"}
	configuration {"windows", "x64"}
		libdirs{opencv_path.."/lib64"}
	end
	--on linux, we prefer to use system includes		
		configuration "linux"		
			includedirs{ 	
				"/usr/local/include/opencv/"--useful when using manual setup of opencv, ex: 2.x
				,"/usr/include/opencv/"
				}
		--what about the libs?
			libdirs{"/usr/local/lib/"--useful when using manual setup of opencv, ex: 2.x
				}
		--what about MACOSX?				

	
	--opencv 2.x lib name, TODO test under linux
	configuration {"linux"}
		links{"cv", "cxcore", "cvaux", "highgui"}

	configuration {"macosx"}
		links{"cv", "cxcore", "cvaux", "highgui"}

	
	configuration {"windows"}
		links{"cv"..opencv_version, "cxcore"..opencv_version, "cvaux"..opencv_version, "highgui"..opencv_version}

end--_ACTION
Debug("Include opencv.premake4.lua...done")