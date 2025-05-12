export GOOGLE_APPLICATION_CREDENTIALS="INCLUDE CREDENTIALS"
export GCLOUD_PROJECT="INCLUDE GCLOUD PROJECT"
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
conda init bash
conda activate LLM_RL

# Training BC in LLM_RL
python3 -m llm_rl_scripts.guess_city.bc.train_bc HF gpt2-medium /nfs/nfs1/users/marwa/generation-benchmarks/cities_data/guess_my_city_train_bc.json /nfs/nfs1/users/marwa/generation-benchmarks/cities_data/guess_my_city_eval_bc.json  /nfs/nfs1/users/marwa/lmrl-gym-final/JaxSEQ/gs:/rail-tpus-marwa/guess_city/exp.2024-01-16-22-14-31.343.9f6185f4b4bc11ee8b6d872abd62eb10/best --outputs_path=gcs://rail-tpus-marwa/guess_city/bc/ --epochs 1     --train-bsize 128     --grad-accum-steps 1     --eval-loss-bsize 32     --eval-loss-batches 256     --policy-max-output-length 128     --log-every 256     --eval-every-steps 1024     --save-every-steps 1024     --save-at-end     --no-save-train-state     --data-mesh-shape -1     --fsdp-mesh-shape 1     --model-mesh-shape 1     --gradient-checkpointing

# Training MC in LLM_RL
python3 -m llm_rl_scripts.guess_city.mc.train_mc_returns HF gpt2-medium /nfs/nfs1/users/marwa/generation-benchmarks/cities_data/guess_my_city_train_bc.json /nfs/nfs1/users/marwa/generation-benchmarks/cities_data/guess_my_city_eval_bc.json  /nfs/nfs1/users/marwa/lmrl-gym-final/JaxSEQ/gs:/rail-tpus-marwa/guess_city/exp.2024-01-16-22-14-31.343.9f6185f4b4bc11ee8b6d872abd62eb10/best --outputs_path=gcs://rail-tpus-marwa/guess_city/mc/ --epochs 1     --train-bsize 128     --grad-accum-steps 1     --eval-loss-bsize 32     --eval-loss-batches 256     --policy-max-output-length 128     --log-every 256     --eval-every-steps 1024     --save-every-steps 1024     --save-at-end     --no-save-train-state     --data-mesh-shape -1     --fsdp-mesh-shape 1     --model-mesh-shape 1     --gradient-checkpointing

# Training ILQL in LLM_RL
python3 -m llm_rl_scripts.guess_city.ilql.train_ilql HF gpt2-medium /nfs/nfs1/users/marwa/generation-benchmarks/cities_data/guess_my_city_train_bc.json /nfs/nfs1/users/marwa/generation-benchmarks/cities_data/guess_my_city_eval_bc.json  /nfs/nfs1/users/marwa/lmrl-gym-final/JaxSEQ/gs:/rail-tpus-marwa/guess_city/exp.2024-01-16-22-14-31.343.9f6185f4b4bc11ee8b6d872abd62eb10/best --outputs_path=gcs://rail-tpus-marwa/guess_city/ilql/ --epochs 1     --train-bsize 128     --grad-accum-steps 1     --eval-loss-bsize 32     --eval-loss-batches 256     --policy-max-output-length 128     --log-every 256     --eval-every-steps 1024     --save-every-steps 1024     --save-at-end     --no-save-train-state     --data-mesh-shape -1     --fsdp-mesh-shape 1     --model-mesh-shape 1     --gradient-checkpointing

# Training PPO in LLM_RL
python -m llm_rl_scripts.guess_city.ppo.train_ppo PARAMS gcs://rail-tpus-charles-3/ILQL5/outputs/twenty_questions/bc_gpt2med_test8_converted/model  /nfs/nfs1/users/marwa/generation-benchmarks/cities_data/guess_my_city_train_bc.json  /nfs/nfs1/users/marwa/lmrl-gym-final/JaxSEQ/gs:/rail-tpus-marwa/guess_city/exp.2024-01-16-22-14-31.343.9f6185f4b4bc11ee8b6d872abd62eb10/best --outputs_path=rail-tpus-marwa/guess_city/ppo/ --n-rollouts 512 --train-bsize 8 --grad-accum-steps 4 --rollout-bsize 64 --ppo-data-bsize 64 --n-rounds 1000 --epochs 4 --log-every 32 --weight-decay 1e-6 --lr 3e-5 --init-kl-coef 0.001 --kl-target 0.1  --kl-horizon 10000 --value-loss-coef 1.0 --data-mesh-shape 1 --fsdp-mesh-shape -1  --model-mesh-shape 1  --use-wandb --wandb-project guess_city  --env-deterministic

# Human Eval in LLM_RL
python -m llm_rl_scripts/guess_city/misc/guess_city_human_eval.py