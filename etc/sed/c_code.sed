rem extern"C" tag
/^extern/ d

/typedef / d

/return / d

/else / d


rem get function definition in one line
s/^,\\\n*/, /
rem empty lines
/^$/ d
