class ApplicationError(Exception):
    """Base exception for application use cases."""


class UserAlreadyExistsError(ApplicationError):
    """Raised when a user is already registered."""


class UserNotFoundError(ApplicationError):
    """Raised when a user is not registered."""


class UserNotSyncedError(ApplicationError):
    """Raised when recommendation history has not been synchronized."""


class SyncJobNotFoundError(ApplicationError):
    """Raised when no synchronization job exists for a user."""
