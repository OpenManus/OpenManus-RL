"""Search selection for unconfigured GAIA/tool-use environments."""

import os
import sys


def default_search_tool():
    # Match the existing Google tool's .env support, without overriding the
    # environment. Explicit tool lists bypass this default entirely.
    from dotenv import load_dotenv

    load_dotenv()
    if (
        sys.version_info < (3, 10)
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_CX")
    ):
        return "google_search"
    return "parallel_search"
