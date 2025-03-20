"""
Entrypoint for the webarena agent environment.
"""

from gunicorn.app.base import BaseApplication
import argparse

from .utils import debug_flg
from .server import app


class CustomGunicornApp(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            if key in self.cfg.settings and value is not None:
                self.cfg.set(key.lower(), value)

    def load(self):
        return self.application


def launch():
    """entrypoint for `webarena` commond"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    options = {
        "bind": "{}:{}".format(args.host, args.port),
        "workers": args.workers,
        "timeout": 120,  # first creation takes long time
        "reload": True,
    }

    CustomGunicornApp(app, options).run()
