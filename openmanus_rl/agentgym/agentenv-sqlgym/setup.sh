# https://github.com/KYLN24/sqlgym/blob/master/README.md

mkdir bird
cd bird

# Download BIRD-SQL Dataset
wget -c https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip
unzip train.zip
cd train
unzip train_databases.zip
cd ..

wget -c https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip
unzip dev.zip
cd dev
unzip dev_databases.zip
cd ..

pwd
