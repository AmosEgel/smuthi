# -*- coding: utf-8 -*-
import argparse
import smuthi.utility.read_input
import pkg_resources
import os


def main():
    """This function is called when Smuthi is executed as a script with an 
    input file (rather than from within Python)"""
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', nargs='?', default=None, type=str,
                        help='Input file containing the parameters of the simulations.'
                             'See https://gitlab.com/AmosEgel/smuthi for further information. '
                             'Default is the shipped example_input.dat')
    args = parser.parse_args()

    if args.inputfile is None:
        datadirname = os.path.abspath(pkg_resources.resource_filename('smuthi', '_data'))
        args.inputfile = datadirname + '/example_input.dat'

    simulation = smuthi.utility.read_input.read_input_yaml(args.inputfile)
    simulation.run()


if __name__ == "__main__":
    main()
