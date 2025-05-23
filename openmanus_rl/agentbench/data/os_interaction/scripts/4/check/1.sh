#!/bin/bash

count_files() {
    # echo $1 >> tmp.log
    local dir=$1
    local count=0

    for file in "$dir"/*; do
        if [ -f "$file" ]; then
            count=$((count + 1))
        elif [ -d "$file" ]; then
            count_sub=$(count_files "$file")
            count=$((count + count_sub))
        fi
    done

    echo "$count"
}

# echo `count_files "/usr/local/bin"`, `count "/usr/local/bin"`

[ `count_files "/usr/local/bin"`x != `count "/usr/local/bin"`x ] && exit 1
[ `count_files "/root"`x != `count "/root"`x ] && exit 1
[ `count_files "/bin"`x != `count "/bin"`x ] && exit 1
[ `count_files "/lib"`x != `count "/lib"`x ] && exit 1
[ `count_files "/dev"`x != `count "/dev"`x ] && exit 1
[ `count_files "/usr/include"`x != `count "/usr/include"`x ] && exit 1
exit 0