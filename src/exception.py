"""
Custom exception handling for the Airline Passenger Satisfaction project.

Wraps any exception with detailed context: the Python file and line number
where it occurred, plus the original error message.

Usage::

    try:
        do_something()
    except Exception as e:
        raise CustomException(e, sys)
"""

import sys


def get_error_details(error_message: str) -> str:
    """
    Build an informative error string from the current exception context.

    Returns something like::

        Error in script [data_ingestion.py] line [42] message [FileNotFoundError: ...]
    """
    _, _, exc_traceback = sys.exc_info()
    if exc_traceback is not None:
        # Walk to the outermost frame (where the error originated).
        while exc_traceback.tb_next is not None:
            exc_traceback = exc_traceback.tb_next
        file_name = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
    else:
        file_name = "<unknown>"
        line_number = -1

    return (
        f"Error in script [{file_name}] "
        f"line [{line_number}] "
        f"message [{error_message}]"
    )


class CustomException(Exception):
    """Project-wide exception carrying file / line / original-message context."""

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = get_error_details(str(error_message))

    def __str__(self):
        return self.error_message
