echo removing temp files
rm $(find ../ -name '*.*~')

echo Generate makefiles
cd ..
./premake4 --os=linux --plug-cuda gmake
cd build

