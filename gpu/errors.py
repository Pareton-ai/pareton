"""GPU package errors."""

from __future__ import annotations


class GpuError(Exception):
    """Base error for the gpu package."""


class ProvisionError(GpuError):
    """Pod rent / wait-ready failure."""


class DestroyError(GpuError):
    """Workload or volume teardown failure (retain registry destroy_failed)."""
