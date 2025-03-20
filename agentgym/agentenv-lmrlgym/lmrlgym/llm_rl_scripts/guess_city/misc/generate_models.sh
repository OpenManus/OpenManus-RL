export GOOGLE_APPLICATION_CREDENTIALS="INCLUDE CREDENTIALS"
export GCLOUD_PROJECT="INCLUDE GCLOUD PROJECT"
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
conda init bash
conda activate LLM_RL

# Generate GPT-3 Data 

# Clean GPT-3 Data 

# Training Oracle in JaxSEQ
python examples_jaxseq/T5/T5_train.py HF google/flan-t5-xl google/flan-t5-xl /nfs/nfs1/users/marwa/datasets_final/city/guess_my_city_train.json /nfs/nfs1/users/marwa/datasets_final/city/guess_my_city_eval.json --outputs-path=gs://rail-tpus-marwa/guess_city/ --max-input-length=124 --max-output-length=4 --lr=0.00001 --epochs=4 --train-bsize=32 --eval-loss-bsize=32 --grad-accum-steps=1 --log-every=64 --eval-every-steps=64 --save-every-epochs=1 --use-wandb --wandb-project guess_city

# Training Answerer in JaxSEQ
python examples_jaxseq/T5/T5_train.py HF google/flan-t5-xl google/flan-t5-xl /nfs/nfs1/users/marwa/datasets_final/city/guess_my_city_train.json /nfs/nfs1/users/marwa/datasets_final/city/guess_my_city_eval.json --outputs-path=gs://rail-tpus-marwa/guess_city/ --max-input-length=124 --max-output-length=4 --lr=0.00001 --epochs=4 --train-bsize=32 --eval-loss-bsize=32 --grad-accum-steps=1 --log-every=64 --eval-every-steps=64 --save-every-epochs=1 --use-wandb --wandb-project guess_city

# Generate BC Data in LLM_RL 

# Clean BC Data in LLM_RL 

