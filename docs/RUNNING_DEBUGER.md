bash OpenManus-RL/scripts/docker_setup_debug.sh

echo "1. Enter the container: docker exec -it openmanus-debugger bash"
echo "2. Activate the environment: source /opt/openmanus-venv/bin/activate"

bash scripts/rollout/test_unified_debugger.sh

Results save under the experiments folder

Fix errors in OpenManus-RL/scripts/rollout/openmanus_rollout_debugger.py

Error1: json not as expected in Advance Debugger, Line 381
Error2: Get back to the critical steps and append the suggestion after the obs prompt
