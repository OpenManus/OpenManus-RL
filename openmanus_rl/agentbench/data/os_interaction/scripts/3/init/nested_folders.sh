#!/bin/bash

create_nested_folders() {
    local depth=$1  # Current depth
    local max_depth=$2  # Max depth


    mkdir "folder$depth"
    cd "folder$depth"

    touch "echo-love"

    if [ $depth -eq 5 ]; then
        echo "echo \"I love myself.\"" > "echo-love"
        chmod +x "echo-love"
    fi

    if [ $depth -lt $max_depth ]; then
        create_nested_folders $((depth + 1)) $max_depth
    fi

    cd ..
}

max_depth=10

create_nested_folders 1 $max_depth
