export TOKENIZERS_PARALLELISM=false
export GOOGLE_APPLICATION_CREDENTIALS="/nfs/nfs1/users/isadoracw/rail-tpus.json"
export GCLOUD_PROJECT="rail-tpus"
# sudo chmod -R 777 /nfs/nfs1/users/isadoracw/LLM_RL
git config --global --add safe.directory /nfs/nfs1/users/isadoracw/LLM_RL
sudo rm -r /tmp/*tpu*
export maze_name="double_t_maze"
export checkpoint_path="gcs://rail-tpus-isadora/maze/maze_double_t_maze/llm_rl_ilql_submazes_double_t_maze/llm_rl_ilql_submazes_double_t_maze.2023-09-19-22-11-54.348.8a806abe573911ee9749e351425a3ca0/last"
python -m examples_jaxseq.misc.export_checkpoint my_checkpoint_path