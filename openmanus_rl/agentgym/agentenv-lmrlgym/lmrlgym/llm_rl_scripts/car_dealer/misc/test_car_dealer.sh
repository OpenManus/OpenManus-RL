export GOOGLE_APPLICATION_CREDENTIALS="INCLUDE CREDENTIALS"
export GCLOUD_PROJECT="INCLUDE GCLOUD PROJECT"
export TOKENIZERS_PARALLELISM=false
sudo rm -r /tmp/*tpu*
conda init bash
conda activate LLM_RL

# Training BC in LLM_RL
python -m llm_rl_scripts.car_dealer.bc.train_bc HF gpt2-xl buyer_model/ --outputs-path=car_dealer/outputs/ --data-path=car_dealer/data/ --epochs=18 --train-bsize=16 --grad-accum-steps=8 --inference-bsize=32 --num-logs-per-epoch=4 --num-evals-per-epoch=4 --save-best --save-last --model-p-shape=4 --use-wandb

# Training MC in LLM_RL
python3 -m llm_rl_scripts.car_dealer.mc.train_mc_returns HF gpt2-xl --outputs_path=gcs://rail-tpus-marwa/car_dealer/mc/ --epochs 1    --train-bsize 128     --grad-accum-steps 1     --eval-batches 256      --log-every 256     --eval-every 1024     --save-every 1024   --data_p_shape -1   --model_p_shape 1     --gradient_checkpoint

# Training ILQL in LLM_RL
python3 -m llm_rl_scripts.car_dealer.ilql.train_ilql HF gpt2-xl train.json eval.json  model/best  --epochs 1  --train-bsize 128     --grad-accum-steps 1     --eval-loss-bsize 32     --eval-loss-batches 256       --log-every 256     --eval-every 1024     --save-every 1024   --data-p-shape -1     --model-p-shape 1     --gradient-checkpoint

# Training PPO in LLM_RL
python -m llm_rl_scripts.car_dealer.ppo.train_ppo PARAMS seller_bc_gpt2xl_test4_converted/model outputs/car_dealer/buyer_bc_gpt2xl_test4_converted/model  --exp-name ppo_revenue_gpt2xl_test1  --outputs-path gcs://rail-tpus-marwa/car_dealer/  --train-bsize 4   --grad-accum-steps 1000 --log-every 1000  --n-rounds 1000  --epochs 4 --n-rollouts 4000  --gamma 0.99 --rollout-bsize 4 --ppo_data_bsize 4    --eval-every-rounds 1 --weight-decay 0.0 --lr 5e-6 --save-every-rounds 50 --init-kl-coef 0.01   --cliprange-value 0.2 --cliprange 0.2   --value-loss-coef 1.0  --wandb-project car_dealer-ppo  --use-wandb 

# Human Eval in LLM_RL
python -m llm_rl_scripts/guess_city/misc/car_dealer_human_eval.py

# Evaluation any model in LLM_RL
python -m llm_rl_scripts/guess_city/misc/car_dealer_model_eval.py

