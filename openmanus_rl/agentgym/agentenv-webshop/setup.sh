pip install -U "Werkzeug>=2,<3" "mkl>=2021,<2022"

cd ./webshop
bash ./setup.sh -d small
pip install -U "typing_extensions<4.6.0" "gym==0.23.1"
python -m spacy download en_core_web_lg
cd ..

pip install -U python-Levenshtein
pip install -r requirements.txt

pip install -e .

pip uninstall numpy -y
pip install numpy
