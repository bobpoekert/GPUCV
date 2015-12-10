rem sed -e 's/#.*//' -e 's/[ ^I]*$//' -e '/^$/ d' 
rem /^\// d

rem remove starting and ending tabs
s/[ ^]*$//
s/^[ ^]*//

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
rem $!N;s/\n/ /



rem extern"C" tag
/^extern/ d


rem { code }
rem /[{]/ !b 
rem : a
rem /[}]/ !{
rem  N
rem  b a
rem }
rem s/[{].*[}]//





rem empty lines
/^$/ d

/^\\\n$/ d

rem empty lines
/^$/ d
