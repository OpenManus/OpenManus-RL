#!/usr/bin/env python
import os
import sys
import subprocess

if __name__ == "__main__":
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Get current directory
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run tests using pytest
    try:
        # Run server tests first
        print("Running server tests...")
        subprocess.run(["pytest", os.path.join(test_dir, "test_server.py"), "-v"], check=True)
        
        # Run trajectory tests 
        print("\nRunning trajectory tests...")
        subprocess.run(["pytest", os.path.join(test_dir, "test_trajectories.py"), "-v"], check=True)
        
        # Optionally run all other tests
        print("\nRunning any remaining tests...")
        subprocess.run(["pytest", test_dir, "-v", 
                      "--ignore=test_server.py", 
                      "--ignore=test_trajectories.py"], check=True)
        
        print("\nAll tests completed successfully!")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\nTest execution failed with error code: {e.returncode}")
        sys.exit(e.returncode) 