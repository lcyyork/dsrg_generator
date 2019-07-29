"""
Helper functions for imap of multiprocessing module to handle functions with multiple arguments.
see https://docs.python.org/3/library/multiprocessing.html
"""


def calculate(func, args):
    return func(*args)


def calculate_star(args):
    return calculate(*args)

