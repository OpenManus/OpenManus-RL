import ray
import torch
import os
import time

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

@ray.remote(num_gpus=1) # Request 1 GPU for this actor
class GPUTestActor:
    def __init__(self):
        print_cuda_info("GPUTestActor.__init__")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"GPUTestActor using device: {self.device}")
        # Small tensor to test allocation
        try:
            self.tensor = torch.randn(10, 10).to(self.device)
            print("GPUTestActor successfully allocated tensor on device.")
        except Exception as e:
            print(f"GPUTestActor ERROR allocating tensor: {e}")


    def check_device(self):
        print_cuda_info("GPUTestActor.check_device()")
        return str(self.device), os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')

def main():
    print_cuda_info("test_ray_gpu.py - Before Ray Init")
    # Basic Ray initialization, letting Ray detect resources
    try:
        # Ensure CUDA_VISIBLE_DEVICES from env is passed if set
        runtime_env = {}
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_visible:
             runtime_env['env_vars'] = {'CUDA_VISIBLE_DEVICES': cuda_visible}
             print(f"Passing CUDA_VISIBLE_DEVICES='{cuda_visible}' to ray.init")
        ray.init(runtime_env=runtime_env) # Let Ray detect num_gpus automatically
        print("Ray initialized.")
    except Exception as e:
        print(f"ERROR during Ray initialization: {e}")
        print_cuda_info("test_ray_gpu.py - After Ray Init Failure")
        return

    print_cuda_info("test_ray_gpu.py - After Ray Init")
    print("\n--- Ray Cluster Resources ---")
    print(ray.cluster_resources())
    print("---------------------------")

    # Create the GPU actor
    try:
        print("\nAttempting to create GPUTestActor...")
        actor = GPUTestActor.remote()
        # Wait for actor init to potentially finish
        time.sleep(2)
        print("GPUTestActor created (or creation initiated).")
    except Exception as e:
        print(f"ERROR creating GPUTestActor: {e}")
        ray.shutdown()
        return

    # Check the actor's device
    try:
        print("\nAttempting to check actor device...")
        device_info = ray.get(actor.check_device.remote())
        print(f"Actor reported device: {device_info[0]}, CUDA_VISIBLE_DEVICES: {device_info[1]}")
    except Exception as e:
        print(f"ERROR checking actor device: {e}")

    print("\nShutting down Ray.")
    ray.shutdown()

if __name__ == "__main__":
    main()