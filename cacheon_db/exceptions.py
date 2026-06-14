"""Database errors for API read paths."""


class DatabaseNotConfigured(Exception):
    """Raised when CACHEON_DATABASE_URL is missing."""


class DatabaseUnavailable(Exception):
    """Raised when a required database connection fails."""
