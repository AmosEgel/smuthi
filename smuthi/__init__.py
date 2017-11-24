import pkg_resources
import sys


def print_smuthi_header():
    version = pkg_resources.get_distribution("smuthi").version
    welcome_msg = ("\n" + "*" * 32 + "\n    SMUTHI version " + version + "\n" + "*" * 32 + "\n")
    sys.stdout.write(welcome_msg)
    sys.stdout.flush()

print_smuthi_header()