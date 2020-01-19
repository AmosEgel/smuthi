import sys
import os


class Logger(object):
    """Allows to prompt messages both to terminal and to log file simultaneously.
    It also allows to print with indentation or to temporally mute the Logger.
    """
    def __init__(self, log_filename=None, log_to_file=True, log_to_terminal=True):
        if not log_to_terminal:
            f = open(os.devnull, 'w')
            self.terminal = f
        else:
            self.terminal = sys.__stdout__
        self.log_to_file = log_to_file
        if log_to_file:
            self.log = open(log_filename, "a")

    def __enter__(self):
        self.previous_logger = sys.stdout
        sys.stdout = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.previous_logger

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        if self.log_to_file:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()


class LoggerMuted:
    mute_logger = Logger(log_to_file=False, log_to_terminal=False)

    def __enter__(self):
        self.previous_logger = sys.stdout
        sys.stdout = self.mute_logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.stdout == self.mute_logger:
            sys.stdout = self.previous_logger


class LoggerIndented:
    def __init__(self, indendatation="   "):
        self.indentation = indendatation

    def __enter__(self):
        self.previous_write = sys.stdout.write

        def new_write(message):
            self.previous_write(self.indentation + message)
        self.new_write = new_write
        sys.stdout.write = self.new_write

    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.stdout.write == self.new_write:
            sys.stdout.write = self.previous_write

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def write_header(message):
    print(bcolors.HEADER + message + bcolors.ENDC)

def write_green(message):
    print(bcolors.OKGREEN + message + bcolors.ENDC)


def write_red(message):
    print(bcolors.FAIL + message + bcolors.ENDC)


def write_blue(message):
    print(bcolors.OKBLUE + message + bcolors.ENDC)