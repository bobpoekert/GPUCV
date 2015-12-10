echo removing temp files
del /s ..\*.*~

echo Generate solutions
cd ..
dir
rem vs2010: not supported yet
premake4.exe --use-switch vs2008
premake4.exe --use-switch vs2005
rem no cuda support under VS2003

pause


