echo '#!/bin/bash

count_files() {
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

directory="$1"
total_count=$(count_files "$directory")
echo "$total_count"' > /usr/local/bin/count
chmod +x /usr/local/bin/count