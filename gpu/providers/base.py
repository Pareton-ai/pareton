"""Provider protocol for GPU cloud control planes."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from gpu.types import Offer, Pod, PodSpec


@runtime_checkable
class Provider(Protocol):
    name: str

    def search(self, spec: PodSpec) -> list[Offer]: ...

    def provision(self, offer: Offer, *, name: str, ssh_public_key: str) -> Pod: ...

    def destroy(self, pod: Pod) -> None: ...

    def list_pods(self) -> list[Pod]: ...

    def list_volumes(self) -> list[dict[str, Any]]: ...
