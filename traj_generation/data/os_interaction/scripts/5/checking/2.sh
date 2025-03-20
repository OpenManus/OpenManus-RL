#!/bin/bash

check() {
    target=`date -d "$1" +"$2"`
    output=`date-format "$1" "$2"` || exit 1
    [ "$output"x != "$target"x ] && exit 1
    exit 0
}

check "2023-5-1" "%Y-%m" || exit 1
check "23-5-2" "%Y-%m-%d" || exit 1
check "2023-5-1" "%Y/%m" || exit 1
check "2023-5-1" "%m/%d" || exit 1
check "2023/5/10" "%d/%m" || exit 1
check "2021/05/1" "Date: %Y-%m-%d" || exit 1

exit 0