s/^[  ]*$//

s/^[  ]*//


rem empty lines
/^$/ d

/^\\\n$/ d


rem //
/^[/\][/\]/ d

