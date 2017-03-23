# -*- coding: utf-8 -*-

import sys
import smuthi.tests.debug_particle_coupling

def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    print("--------------------------------")
    print("|            SMUTHI            |")
    print("--------------------------------")
    smuthi.tests.debug_particle_coupling.execute()

    # Do argument parsing here (eg. with argparse) and anything else
    # you want your project to do.

if __name__ == "__main__":
    main()
