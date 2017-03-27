# -*- coding: utf-8 -*-

import argparse
import smuthi.read_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', type=str)
    args = parser.parse_args()
    if args.inputfile is None:
        inputfile = 'input.dat'
    else:
        inputfile = args.inputfile

    simulation = smuthi.read_input.read_input_yaml(inputfile)
    simulation.run()


if __name__ == "__main__":
    main()
