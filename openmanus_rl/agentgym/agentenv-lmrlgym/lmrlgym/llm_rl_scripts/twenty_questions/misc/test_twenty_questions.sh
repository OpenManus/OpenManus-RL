# Configs 
export GOOGLE_APPLICATION_CREDENTIALS="INCLUDE CREDENTIALS"
export GCLOUD_PROJECT="INCLUDE GCLOUD PROJECT"
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
conda init bash
conda activate LLM_RL

# Training Oracle in JaxSEQ
python examples_jaxseq/T5/T5_train.py HF T5 google/flan-t5-xl /nfs/nfs1/users/marwa/datasets_gpt/oracle_flan-t5-xxl_train.json /nfs/nfs1/users/marwa/datasets_gpt/oracle_flan-t5-xxl_eval.json --outputs-path=gs://rail-tpus-marwa/twenty_questions/

# Training Guesser in JaxSEQ
python examples_jaxseq/T5/T5_train.py HF google/flan-t5-xl google/flan-t5-xl /nfs/nfs1/users/marwa/datasets_final/city/guess_my_city_train.json /nfs/nfs1/users/marwa/datasets_final/city/guess_my_city_eval.json --outputs-path=gs://rail-tpus-marwa/guess_city/ --max-input-length=124 --max-output-length=4 --lr=0.00001 --epochs=4 --train-bsize=32 --eval-loss-bsize=32 --grad-accum-steps=1 --log-every=64 --eval-every-steps=64 --save-every-epochs=1 --use-wandb --wandb-project=guess_city

# Training BC in LLM_RL
python -m llm_rl_scripts.twenty_questions.bc.train_bc HF gpt2-medium gcs://rl-llm-bench-dataset-internal/twenty-questions/train.json gcs://rl-llm-bench-dataset-internal/twenty-questions/eval.json gcs://rl-llm-bench-dataset-internal/twenty-questions/oracle --outputs_path=gcs://rail-tpus-isadora/test-twenty-questions/bc/ --data-mesh-shape 4 --model-mesh-shape 2 --epochs 1 --use-wandb --wandb-project twenty_questions
python -m llm_rl_scripts.twenty_questions.bc.train_bc HF gpt2 gcs://rl-llm-bench-dataset-internal/twenty-questions/train.json gcs://rl-llm-bench-dataset-internal/twenty-questions/eval.json gcs://rl-llm-bench-dataset-internal/twenty-questions/oracle --data-mesh-shape 4 --model-mesh-shape 2 --epochs 1 --use-wandb --wandb-project twenty_questions

# Training Filtered BC in LLM_RL

# Training MC in LLM_RL
python -m llm_rl_scripts.twenty_questions.mc.train_mc_returns HF gpt2-medium gcs://rl-llm-bench-dataset-internal/twenty-questions/train.json gcs://rl-llm-bench-dataset-internal/twenty-questions/eval.json gcs://rl-llm-bench-dataset-internal/twenty-questions/oracle --outputs_path=gcs://rail-tpus-isadora/twenty-questions/ilql/ --eval-every-steps 512 --data-mesh-shape 4 --model-mesh-shape 1 --eval-at-beginning --epochs 1 --use-wandb --wandb-project twenty_questions

# Training ILQL in LLM_RL
python -m llm_rl_scripts.twenty_questions.ilql.train_ilql HF gpt2-medium gcs://rl-llm-bench-dataset-internal/twenty-questions/train.json gcs://rl-llm-bench-dataset-internal/twenty-questions/eval.json gcs://rl-llm-bench-dataset-internal/twenty-questions/oracle --outputs_path=gcs://rail-tpus-isadora/twenty-questions/ilql/ --eval-every-steps 512 --data-mesh-shape 4 --model-mesh-shape 1 --eval-at-beginning --epochs 1 --use-wandb --wandb-project twenty_questions

# Training PPO in LLM_RL
python -m llm_rl_scripts.twenty_questions.ppo.train_ppo HF gpt2-medium gcs://rl-llm-bench-dataset-internal/twenty-questions/train.json gcs://rl-llm-bench-dataset-internal/twenty-questions/eval.json PARAMS gcs://rl-llm-bench-dataset-internal/twenty-questions/oracle --outputs_path=gcs://rail-tpus-isadora/twenty-questions/PPO/ --eval-every-rounds 1 --data-mesh-shape 4 --model-mesh-shape 1 --eval-at-beginning --n-rounds 100 --use-wandb --wandb-project twenty_questions