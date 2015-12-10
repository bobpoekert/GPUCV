echo off
echo !!==================================================!!
echo Add SVN version to GpuSetting.cpp file
rem http://tortoisesvn.net/docs/release/TortoiseSVN_fr/tsvn-subwcrev.html
echo !!==================================================!!
"C:\Program Files\TortoiseSVN\bin\SubWCRev.exe" ..\ ..\etc\revision_base.h ..\include\GPUCVHardware\revision.h -nm
pause