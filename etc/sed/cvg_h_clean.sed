rem sed -e 's/#.*//' -e 's/[ ^I]*$//' -e '/^$/ d' 
rem /^\// d

rem remove starting and ending tabs
s/[ ^I]*$//
s/^[ ^I]*//

rem remove starting and ending spaces
s/[ ]*$//
s/^[ ]*//
rem remove starting and ending spaces
s/[ ]*$//
s/^[ ]*//
rem remove starting and ending spaces
s/[ ]*$//
s/^[ ]*//

rem remove # starting and ending lines
/#/ d
rem /^#/ d



rem /,./N; s/\\\n/ /; ta
rem /\\$/N; s/\\\n//; ta
rem /,$/N; s/,$/,join/;
$!N;s/\n/ /
s/^double/DOUBLE/



rem extern"C" tag
/^extern/ d

rem empty lines
/^$/ d
/^\\\n$/ d

rem comments
/[/][*]/ !b 
: a
/[*][/]/ !{
  N
  b a
}  
s/[/][*].*[*][/]//


rem //
/^[/\][/\]/ d



rem { code }
rem /[{]/ !b 
rem : a
rem /[}]/ !{
  rem N
  rem b a
rem }  
rem s/[{].*[}]//






rem Concat functions lines

rem empty lines
/^$/ d
