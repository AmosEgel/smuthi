# -*- coding: utf-8 -*-

import argparse
import smuthi.read_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', nargs='?', default='input.dat', type=str,
                        help='Input file containing the parameters of the simulations.'
                             'See https://gitlab.com/AmosEgel/smuthi for further information. '
                             'Default is input.dat')
    args = parser.parse_args()
    simulation = smuthi.read_input.read_input_yaml(args.inputfile)
    simulation.run()


if __name__ == "__main__":
    main()
