# SFT GUIDE


```
nohup ./run_sft.sh 4 /data1/models/openmanus_rl/Qwen/Qwen3-3b-sft \
    data.truncation=right \
    trainer.total_training_steps=1000 \
    trainer.logger="['console','wandb']" \
    trainer.project_name="OpenManus-rl" \
    > training_run.log 2>&1 &
```


```
./run_sft.sh 4 /data1/models/openmanus_rl/Qwen/Qwen3-3b-sft     data.truncation=right     trainer.total_training_steps=30    trainer.logger="['console','wandb']"     trainer.project_name="OpenManus-rl"
```
