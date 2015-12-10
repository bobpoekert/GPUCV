rem remove starting and ending tabs and space
s/[ ^]*$//
s/^[ ^]*//
s/^[ \t]*$//
s/[ \t]*$//

s/^[ \t]*$//
s/[ \t]*$//



rem remove starting and ending spaces/double space
rem s/[ ]*$//
rem s/^[ ]*//
rem s/[  ]*$//
rem s/^[  ]*//

rem get function definition in one line
s/^,\\\n*/, /


rem concat lines that ends with ',\n'
 :a 
 /,$/N; s/,\n/, /; ta

 rem concat lines that ends with ',\n'
 :d 
 /,$/N; s/,\n/, /; td

 
rem concat lines that ends with '\'
:b
/\\$/N; s/\\\n//; tb 

rem concat lines that ends with '('
:c
/($/N; s/(\n/(/; tc 

rem empty lines
/^$/ d

rem remove double space
s/( )\{2,10\}/ /
s/( )\{4,\}/ /
s/[  ]/ /
s/[  ]/ /
