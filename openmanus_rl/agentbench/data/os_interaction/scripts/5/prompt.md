generate 5 bash problems, and their corresponding solutions and checking script. Note that the solution should contains multi-lines, and the checking script should exit 0 when succeed and exit 1 when failed. Besides, the problems, solutions, and the checking script should match the following format (the [TODO(description)] tags represent the blanks that you should fill):

Problem [TODO(index)]: "I would like to implement the following function: the \"[TODO(command)]\" command can help [TODO(function)]. For example, if I enter command [TODO(example input, command and its parameters)], the output will be [TODO(example result)]."

Solution [TODO(index)]

```bash
echo '#!/bin/bash

[TODO(implement, multi-lines)]

' > /usr/local/bin/[TODO(command)]
chmod +x /usr/local/bin/[TODO(command)]
```

Checking Script [TODO(index)]

```bash
[TODO(some preparation)]

[ [TODO(command)] != '[TODO(the correct answer)]' ] && exit 1
[ [TODO(command)] != '[TODO(the correct answer)]' ] && exit 1
...
[ [TODO(command)] != '[TODO(the correct answer)]' ] && exit 1
[ [TODO(command)] != '[TODO(the correct answer)]' ] && exit 1
exit 0
```

For example:

Problem 1: "I would like to implement the following function: entering the \"count\" command will counts the number of regular files in a directory and its subdirectories(recursively), and displays the total count. If there is a link or something, count it separately. For example, I can enter \"count /usr/local\" to get the number of regular files in /usr/local recursively. If there are 5 regular files in that, the output is \"5\"."

Solution 1

```bash
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
```

Checking Script 1

```bash
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
```
