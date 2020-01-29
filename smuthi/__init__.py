import sys

from smuthi.version import __version__

try:
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
except:
    mpi_rank = 0


def print_smuthi_header():
    welcome_msg = ("\n" + "*" * 32 + "\n    SMUTHI version " + __version__ + "\n" + "*" * 32 + "\n")
    sys.stdout.write(welcome_msg)
    sys.stdout.flush()


#if mpi_rank == 0:
#    print_smuthi_header()
