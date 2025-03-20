#!/bin/bash

check() {
    local expression="$*"
    # echo python3 -c "print(\"%.6f\"%($expression))" >> tmp.log
    local expected_result=`python3 -c "print(\"%.6f\"%($expression))"`
    local output=$(calc "${expression}")
    # echo "$expression", $expected_result, $output >> tmp.log
    echo `python3 -c "print(abs($output - $expected_result)<1e-5)"`
}

# echo > tmp.log
[ `check "15 + (27 * 4) - 10"`x != Truex ] && exit 1
[ `check "8 * (14 - 6) + 12"`x != Truex ] && exit 1
[ `check "3 + (6.7 * 9) - 5.5"`x != Truex ] && exit 1
[ `check "20 / (5 + 2) - 1"`x != Truex ] && exit 1
[ `check "9 * (16 / 8) + 3"`x != Truex ] && exit 1
[ `check "25 - (8 * 3) + 2"`x != Truex ] && exit 1
[ `check "14 + (25.6 / 2) - 5.2"`x != Truex ] && exit 1
[ `check "18 / (6 - 2) + 9"`x != Truex ] && exit 1
[ `check "17 + (18.2 / 2) - 2.8"`x != Truex ] && exit 1
[ `check "36 / (6 - 3) + 10"`x != Truex ] && exit 1
[ `check "5 + (10 * 4) - 8"`x != Truex ] && exit 1
[ `check "50 / (5 + 2) - 6"`x != Truex ] && exit 1
[ `check "8 * (16 / 4) + 9"`x != Truex ] && exit 1
[ `check "21 - (9 * 2) + 4"`x != Truex ] && exit 1

exit 0