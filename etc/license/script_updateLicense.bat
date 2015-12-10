echo off
rem =====================================
:source
FOR /r %%D IN (.) DO (
	FOR /F "tokens=*" %%F IN ('DIR /B %%D\*.cpp %%D\*.h %%D\*.cu') DO (
		echo %%D\%%F
		sed -f etc\license\script_RemoveLicense.seq "%%D\%%F" > "%%D\%%F.new"
		sed -f etc\license\script_AddLicense.seq "%%D\%%F.new" > "%%D\%%F"
		del "%%D\%%F.new"
	)
)
:shader
FOR /r %%D IN (.) DO (
	FOR /F "tokens=*" %%F IN ('DIR /B %%D\*.frag %%D\*.vert') DO (
		IF NOT %%F=="Fichier introuvable" (
			echo %%D\%%F
			sed -f etc\license\script_RemoveShaderLicense.seq "%%D\%%F" > "%%D\%%F.new"
			sed -f etc\license\script_RemoveLicense.seq "%%D\%%F.new" > "%%D\%%F.new1"
			sed -f etc\license\script_AddShaderLicense.seq "%%D\%%F.new1" > "%%D\%%F"
			del "%%D\%%F.new"
			del "%%D\%%F.new1"
			)
	)
)