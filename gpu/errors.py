"""GPU package errors."""

from __future__ import annotations


class GpuError(Exception):
    """Base error for the gpu package."""


class ProvisionError(GpuError):
    """Pod rent / wait-ready failure."""


class NoCapacityError(ProvisionError):
    """No provider had an offer matching the spec.

    Nothing was rented, so there is no partial state to clean up and the same
    request can succeed unchanged once the market has stock. Callers treat this
    as "wait", not "fail": see `worker.round_job` where it defers the round
    instead of voiding it.
    """


class DestroyError(GpuError):
    """Workload or volume teardown failure (retain registry destroy_failed)."""
