class TrainingTimeoutException(Exception):
    """Exception raised when training exceeds the allocated time limit."""

    def __init__(self, message="Training exceeded the time limit."):
        super().__init__(message)
