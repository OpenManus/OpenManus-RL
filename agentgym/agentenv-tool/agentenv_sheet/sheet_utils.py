import os

debug_flg = bool(os.environ.get("AGENTENV_DEBUG", False))

if debug_flg:
    print("Debug mode")
