from __future__ import annotations

import functools
import logging
from typing import Callable, NoReturn, ParamSpec, TypeVar

import gradio as gr


P = ParamSpec("P")
T = TypeVar("T")


def get_exception_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__


def log_service_exception(
    logger: logging.Logger,
    action: str,
    exc: BaseException,
    *,
    level: int = logging.ERROR,
    include_traceback: bool | None = None,
) -> None:
    if include_traceback is None:
        include_traceback = level >= logging.ERROR

    message = get_exception_message(exc)
    if include_traceback:
        logger.log(level, "Failed to %s: %s", action, message, exc_info=exc)
        return

    logger.log(level, "Failed to %s: %s", action, message)


def raise_gradio_error(exc: Exception, *, logger: logging.Logger, action: str) -> NoReturn:
    if isinstance(exc, gr.Error):
        raise exc

    log_service_exception(logger, action, exc, level=logging.ERROR, include_traceback=True)
    raise gr.Error(get_exception_message(exc)) from exc


def gradio_error_boundary(action: str, logger: logging.Logger) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                raise_gradio_error(exc, logger=logger, action=action)

        return wrapper

    return decorator
