# -*- coding: utf-8 -*-
import smuthi.read_input


def run_main():
    simulation = smuthi.read_input.read_input_yaml('input.dat')
    simulation.run()

