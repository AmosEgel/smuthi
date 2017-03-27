# -*- coding: utf-8 -*-

import sys
import smuthi.read_input


def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]

    simulation = smuthi.read_input.read_input_yaml('input.dat')
    simulation.run()


if __name__ == "__main__":
    main()
