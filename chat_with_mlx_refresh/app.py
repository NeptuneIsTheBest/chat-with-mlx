from __future__ import annotations

import argparse
import atexit
import logging
import threading
from typing import Optional

import gradio as gr

from .context import AppContext
from .ui import create_app


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_runtime: tuple[Optional[AppContext], Optional[gr.Blocks]] = (None, None)
_exit_registered = False
_runtime_lock = threading.Lock()


def _close_runtime() -> None:
    context, _ = _runtime
    if context is not None:
        context.close()


def get_runtime() -> tuple[AppContext, gr.Blocks]:
    global _runtime, _exit_registered
    context, app = _runtime
    if context is None or app is None:
        with _runtime_lock:
            context, app = _runtime
            if context is None or app is None:
                context = AppContext()
                app = create_app(context)
                _runtime = (context, app)
                if not _exit_registered:
                    atexit.register(_close_runtime)
                    _exit_registered = True
    return context, app


def start(port: int, share: bool = False, in_browser: bool = True) -> None:
    _, app = get_runtime()
    logger.info("Starting the app on port %s with share=%s and in_browser=%s", port, share, in_browser)
    app.launch(server_port=port, inbrowser=in_browser, share=share)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with MLX")
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="The port number to run the application on (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable sharing the application link externally",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the application in the default web browser",
    )
    args = parser.parse_args()
    start(port=args.port, share=args.share, in_browser=not args.no_browser)


if __name__ == "__main__":
    main()
