#!/bin/zsh

result_folder=$1
cd cache/$result_folder


# check whether there is any auto-login errors
errors=$(grep -l "Creating an account has many benefits: check out faster" *.html | sort -u | grep -o '[0-9]\+')
c=$(echo $errors | wc -l)
echo "Shopping total errors: $c"
echo $errors | tr '\n' ','
echo '\n\n'


errors=$(grep -l "Welcome, please sign in" *.html | sort -u | grep -o '[0-9]\+')
c=$(echo $errors | wc -l)
echo "Admin total errors: $c"
echo $errors | tr '\n' ','
echo '\n\n'



errors=$(grep -l "Username or email" *.html | sort -u | grep -o '[0-9]\+')
c=$(echo $errors | wc -l)
echo "Gitlab errors: $c"
echo $errors | tr '\n' ','
echo '\n\n'


errors=$(grep -l "Keep me logged in" *.html | sort -u | grep -o '[0-9]\+')
c=$(echo $errors | wc -l)
echo "Reddit errors: $c"
echo $errors | tr '\n' ','
