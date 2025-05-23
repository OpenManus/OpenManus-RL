#!/bin/bash

# Define the possible values for each field
names=("Alice" "Bob")
actions=("Purchase" "Sell")

# Generate 400 random lines
for ((i=1; i<=401; i++))
do
    # Randomly select values for each field
    name=${names[$RANDOM % ${#names[@]}]}
    action=${actions[$RANDOM % ${#actions[@]}]}
    stock_index=$((RANDOM % 100))
    count=$((RANDOM % 1000))
    
    # Write the line to the file
    echo "$name | $action | $stock_index | $count" >> /usr/stock.log
done
