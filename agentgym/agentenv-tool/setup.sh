#!/bin/bash

pip install -r requirements.txt
cd ./Toolusage

pip install -r requirements.txt

cd toolusage
pip install -e .
cd ..
cd ..
pip install --upgrade openai
pip install -e .

current_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PROJECT_PATH="$current_dir/Toolusage"
export MOVIE_KEY=""
export TODO_KEY=""
export SHEET_EMAIL=""