```
nohup ./run_qwen_05_sp2.sh 4 /data1/models/openmanus_rl/Qwen/Qwen3-3b-sft \
    data.truncation=right \
    trainer.total_training_steps=1000 \
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    ++critic.model.fsdp_config.model_dtype=bfloat16 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="OpenManus-rl" \
    > training_run.log 2>&1 &
```

You need to clone a new verl codebase, and use verl conda environment to run this multi-turn sft script.

You should copy openmanus-rl/scripts/run_sft.sh to verl/examples/sft/multiturn/
then run the script


```
./run_qwen_05_sp2.sh 4 /data1/models/openmanus_rl/Qwen/Qwen3-3b-sft     data.truncation=right     trainer.total_training_steps=30    trainer.logger="['console','wandb']"     trainer.project_name="OpenManus-rl"
```