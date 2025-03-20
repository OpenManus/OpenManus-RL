#!/bin/bash
cd ./webarena
pip install -r requirements.txt
playwright install-deps
playwright install
pip install -e .

export SHOPPING="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
export SHOPPING_ADMIN="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
export REDDIT="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
export GITLAB="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023"
export MAP="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
export WIKIPEDIA="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:4399"

python scripts/generate_test_data.py
mkdir -p ./.auth
python browser_env/auto_login.py
python agent/prompts/to_json.py

cd ..

pip install -e .
