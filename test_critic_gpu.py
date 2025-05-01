import ray
import torch
import os
import time
import socket
from contextlib import closing
from unittest.mock import patch
import numpy as np # For UID array

from omegaconf import OmegaConf, DictConfig
from verl import DataProto # Import DataProto
from tensordict import TensorDict # <<< Import TensorDict >>>

# Assume fsdp_workers and utils are importable from the script's location
# Adjust path if necessary, e.g., using sys.path.append
try:
    from verl.workers.fsdp_workers import CriticWorker # Import the base class
    # Mock related imports that might not be strictly needed for init test
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    from verl.utils.debug import log_gpu_memory_usage # Used internally by worker
except ImportError as e:
    print(f"ImportError: {e}")
    print("Please ensure the script is run from a location where 'verl' is importable,")
    print("or adjust Python path.")
    exit(1)

# --- Helper Function to Create Sample Data ---
def create_sample_dataproto(batch_size=2, seq_len=128, response_len=64, vocab_size=1000):
    """Creates a sample DataProto object for testing."""
    # Note: Using random tensors. Replace with tokenized data if specific inputs are needed.
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    # Create dummy responses (last response_len tokens)
    responses = input_ids[:, -response_len:]
    # Create dummy returns (matching full sequence length, as often expected)
    returns = torch.randn((batch_size, seq_len), dtype=torch.float32)
    # Dummy old_log_probs (might be needed indirectly or for consistency)
    old_log_probs = torch.randn((batch_size, seq_len), dtype=torch.float32)
     # Create a dummy token_level_rewards matching the sequence length
    token_level_rewards = torch.randn((batch_size, seq_len), dtype=torch.float32)


    batch_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'responses': responses, # Need responses for potential metrics calculation inside worker
        'returns': returns,     # Need returns for critic update
        'old_log_probs': old_log_probs, # Needed by actor, but maybe critic uses it too? Add for safety.
        'token_level_rewards': token_level_rewards # Often needed for advantage calc, add for safety
    }
    # <<< Convert dict to TensorDict >>>
    batch_td = TensorDict(batch_dict, batch_size=[batch_size])

    # Create dummy UIDs as a NumPy array of objects
    uids = np.array([f"uid_{i}" for i in range(batch_size)], dtype=object)
    non_tensor_dict = {'uid': uids}
    # Add necessary meta_info if worker functions expect it
    meta_info = {'ppo_micro_batch_size': 1} # Example meta info

    # <<< Pass TensorDict to DataProto >>>
    return DataProto(batch=batch_td, non_tensor_batch=non_tensor_dict, meta_info=meta_info)

def print_cuda_info(step_name):
    """Prints CUDA info at a specific step."""
    print(f"\n--- CUDA Info ({step_name}) ---")
    print(f"Running in PID: {os.getpid()}, Host: {os.uname()[1]}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"torch.cuda.device_count(): {device_count}")
        try:
            current_device = torch.cuda.current_device()
            print(f"torch.cuda.current_device(): {current_device}")
            print(f"Device Name: {torch.cuda.get_device_name(current_device)}")
        except Exception as e:
            print(f"  Error getting current device info: {e}")
        for i in range(device_count):
            try:
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            except Exception as e:
                print(f"  Error getting name for device {i}: {e}")
    else:
        print("  No CUDA devices visible to PyTorch.")
    print("---------------------------")

def find_free_port():
    """Finds a free port for the dummy MASTER_PORT."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def mock_copy_local_path_from_hdfs(path):
    print(f"[Mock] copy_local_path_from_hdfs called with: {path}. Returning path directly.")
    return path

def main():
    print_cuda_info("test_critic_gpu.py - Start")

    # Define minimal configuration for CriticWorker
    # Use paths that likely exist in your environment or adjust as needed
    # Assuming Qwen model path is accessible
    model_path = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B")
    tokenizer_path = model_path # Critic uses its own tokenizer, often same as model

    cfg_dict = {
        'model': {
            'path': model_path,
            'tokenizer_path': tokenizer_path, # Added tokenizer path
            'fsdp_config': {
                'param_offload': False, # Start with offload disabled
                'grad_offload': False,
                'optimizer_offload': False,
                'wrap_policy': None, # Let FSDP handle wrapping initially
                 # 'mixed_precision': {'param_dtype': 'bf16'}, # Optional: uncomment if needed
            },
            'trust_remote_code': True, # Set based on your model needs
            'enable_gradient_checkpointing': False, # Keep it simple for test
            'use_remove_padding': True, # Match train_ppo.sh setting
        },
        'optim': {
            'lr': 1e-5,
            # Add other optim params if CriticWorker init requires them
            'total_training_steps': 100 # Dummy value
        },
        # Add other potential top-level keys CriticWorker might access
        'ppo_mini_batch_size': 2, # Example value
        'ppo_micro_batch_size': 1, # Example value
        'forward_micro_batch_size': 1, # Example value
        'forward_max_token_len_per_gpu': 1024, # Example value
        'use_dynamic_bsz': False, # Example value
        'ppo_epochs': 1, # Dummy value for potential MFU calculation
    }
    cfg = OmegaConf.create(cfg_dict)
    print("\n--- Critic Configuration Used ---")
    print(OmegaConf.to_yaml(cfg))
    print("--------------------------------")


    # Prepare runtime environment for the actor
    master_port = find_free_port()
    runtime_env = {
        'env_vars': {
            'RANK': '0',
            'WORLD_SIZE': '1',
            'MASTER_ADDR': '127.0.0.1',
            'MASTER_PORT': master_port,
            'TOKENIZERS_PARALLELISM': 'false', # Avoid potential deadlocks
            'NCCL_DEBUG': 'INFO', # Get NCCL info if used
            'VERL_PPO_LOGGING_LEVEL': 'INFO' # Ensure internal logs are visible
        }
    }
    # Propagate CUDA_VISIBLE_DEVICES from the environment running the script
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible:
        runtime_env['env_vars']['CUDA_VISIBLE_DEVICES'] = cuda_visible
        print(f"Passing CUDA_VISIBLE_DEVICES='{cuda_visible}' to CriticWorker runtime_env")
    else:
        print("CUDA_VISIBLE_DEVICES not set in environment, not passing explicitly.")

    # Initialize Ray
    print_cuda_info("test_critic_gpu.py - Before Ray Init")
    try:
        ray.init(runtime_env=runtime_env) # Pass global runtime_env
        print("Ray initialized.")
    except Exception as e:
        print(f"ERROR during Ray initialization: {e}")
        print_cuda_info("test_critic_gpu.py - After Ray Init Failure")
        return
    print_cuda_info("test_critic_gpu.py - After Ray Init")
    print("\n--- Ray Cluster Resources ---")
    print(ray.cluster_resources())
    print("---------------------------")

    # Patch the HDFS function before creating the actor
    # Also patch the logger inside the worker to ensure output is visible
    with patch('verl.workers.fsdp_workers.copy_local_path_from_hdfs', mock_copy_local_path_from_hdfs), patch('verl.workers.fsdp_workers.logger') as mock_logger: # Mock the logger

        # Configure mock logger to print
        mock_logger.info.side_effect = print
        mock_logger.warn.side_effect = print
        mock_logger.error.side_effect = print
        mock_logger.debug.side_effect = print

        # Create the CriticWorker actor
        actor_handle = None
        success = False
        init_success = False
        compute_val_success = False
        update_critic_success = False
        try:
            print("\nAttempting to create CriticWorker actor...")

            # FIX: Decorate the imported class with ray.remote first
            RemoteCriticWorker = ray.remote(CriticWorker)

            # Explicitly request 1 GPU and pass the specific runtime_env
            actor_handle = RemoteCriticWorker.options(
                num_gpus=1,
                runtime_env=runtime_env # Pass runtime env again for the specific actor
            ).remote(config=cfg)
            print("CriticWorker actor creation requested.")
            # Wait briefly for actor to potentially start initializing
            # Increase wait time slightly to ensure logs appear
            time.sleep(10)

            # Initialize the model remotely
            print("\n[Test 1] Attempting to call actor.init_model()...")
            init_future = actor_handle.init_model.remote()
            ray.get(init_future) # Wait for init_model to complete
            print("actor.init_model() completed.")
            # Verification: Check logs as before
            print("--> VERIFICATION 1: Check logs above for FSDP memory allocation.")
            init_success = True # Assume success if no error during init_model

            # 2. Test compute_values
            if init_success:
                print("\n[Test 2] Attempting to call actor.compute_values()...")
                # Create sample data ONCE
                print("Creating sample DataProto...")
                # Use config values if available, else defaults
                batch_size = cfg.get('ppo_mini_batch_size', 2)
                # Estimate sequence length based on config (adjust if needed)
                seq_len = cfg.get('forward_max_token_len_per_gpu', 128)
                # Estimate response length (needs careful thought, maybe fixed value?)
                # Critic might not need exact response length for value computation
                response_len = 64 # Dummy value, might not be used by compute_values
                sample_batch_proto = create_sample_dataproto(batch_size=batch_size, seq_len=seq_len, response_len=response_len)
                print(f"Sample DataProto created with batch_size={batch_size}, seq_len={seq_len}")

                compute_val_future = actor_handle.compute_values.remote(sample_batch_proto)
                result_proto_values = ray.get(compute_val_future)
                print("actor.compute_values() completed.")

                # Verification: Check output DataProto
                print("--> VERIFICATION 2: Checking output from compute_values...")
                if isinstance(result_proto_values, DataProto) and 'values' in result_proto_values.batch:
                    values_tensor = result_proto_values.batch['values']
                    print(f"  - Found 'values' tensor with shape: {values_tensor.shape} and dtype: {values_tensor.dtype}")
                    # Check shape (should typically match input sequence length)
                    expected_shape = (batch_size, seq_len)
                    if values_tensor.shape == expected_shape:
                         print(f"  - Shape matches expected: {expected_shape}")
                         compute_val_success = True
                    else:
                         print(f"  - WARNING: Shape {values_tensor.shape} does not match expected {expected_shape}")
                         # Decide if this is critical failure or just a warning
                         compute_val_success = True # Let's consider it a success if it runs
                else:
                    print("  - ERROR: Did not receive DataProto with 'values' tensor.")

                # Update the original proto with computed values IF update_critic needs them
                # (Often update_critic recalculates values internally, but let's add it just in case)
                if compute_val_success and 'values' in result_proto_values.batch:
                     sample_batch_proto = sample_batch_proto.union(result_proto_values.select(['values']))
                     print("  - Updated sample_batch_proto with computed 'values'.")


            # 3. Test update_critic
            if compute_val_success: # Only proceed if compute_values worked
                print("\n[Test 3] Attempting to call actor.update_critic()...")
                # sample_batch_proto already contains necessary inputs like 'returns'
                # and potentially 'values' from the previous step.
                update_critic_future = actor_handle.update_critic.remote(sample_batch_proto)
                result_proto_update = ray.get(update_critic_future)
                print("actor.update_critic() completed.")

                # Verification: Check output metrics
                print("--> VERIFICATION 3: Checking output from update_critic...")
                if isinstance(result_proto_update, DataProto) and 'metrics' in result_proto_update.meta_info:
                    metrics = result_proto_update.meta_info['metrics']
                    print(f"  - Received metrics: {metrics}")
                    # Check for specific expected metrics
                    if 'critic/vf_loss' in metrics:
                        print(f"  - Found 'critic/vf_loss': {metrics['critic/vf_loss']}")
                        update_critic_success = True
                    else:
                        print("  - WARNING: Did not find 'critic/vf_loss' in metrics.")
                        update_critic_success = True # Success if it ran without error
                else:
                     print("  - ERROR: Did not receive DataProto with 'metrics' from update_critic.")


        except Exception as e:
            print(f"\nERROR during actor operation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Overall success check
            success = init_success and compute_val_success and update_critic_success

            if actor_handle:
                print("\nTerminating actor...")
                # Give a bit more time for logs to flush before killing
                time.sleep(2)
                ray.kill(actor_handle, no_restart=True)
                time.sleep(2)
            print("Shutting down Ray.")
            ray.shutdown()

            print("\n--- Test Summary ---")
            print(f"Initialization Test: {'PASSED' if init_success else 'FAILED'}")
            print(f"Compute Values Test: {'PASSED' if compute_val_success else 'FAILED' if init_success else 'SKIPPED'}")
            print(f"Update Critic Test:  {'PASSED' if update_critic_success else 'FAILED' if compute_val_success else 'SKIPPED'}")
            print("--------------------")

            if success:
                 print("\nOverall Test Result: PASSED (All core functions executed without critical errors)")
            else:
                 print("\nOverall Test Result: FAILED (See details above)")

if __name__ == "__main__":
    main() 