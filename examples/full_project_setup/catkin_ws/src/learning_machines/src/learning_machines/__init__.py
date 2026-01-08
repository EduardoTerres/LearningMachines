from robobo_interface import IRobobo
from .mycontroller import find_object_and_turnR


def run_all_actions(rob: IRobobo) -> None:
    """
    This overrides the run_all_actions from test_actions.py
    and calls your controller instead.
    """
    find_object_and_turnR(rob)


__all__ = ("run_all_actions",)