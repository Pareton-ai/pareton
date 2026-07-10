"""Database errors for Pareton API and workers."""


class DatabaseNotConfigured(Exception):
    """Raised when PARETON_DATABASE_URL is missing."""


class DatabaseUnavailable(Exception):
    """Raised when a required database connection fails."""
