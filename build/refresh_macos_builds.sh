echo removing temp files
rm $find(../ -name '*.*~')

echo Generate makefiles
../premake4 --os=macos --plug-cuda gmake
../premake4 --os=macos --plug-cuda xcode3

