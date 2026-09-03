"""Serialization for Docker builds and local storage cleanup."""

from __future__ import annotations

import fcntl
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

import config

_F = TypeVar("_F", bound=Callable[..., Any])


@contextmanager
def builder_storage_lock(*, blocking: bool, path: Path | None = None) -> Iterator[bool]:
    """Hold the host-wide builder storage lock.

    Non-blocking callers receive ``False`` when a build or cleanup already
    holds the lock. Blocking callers wait and always receive ``True``.
    """
    lock_path = path or config.BUILDER_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def serialized_build_storage(func: _F) -> _F:
    """Run one mutating Docker builder operation at a time."""

    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with builder_storage_lock(blocking=True) as acquired:
            assert acquired
            return func(*args, **kwargs)

    return cast(_F, wrapped)
