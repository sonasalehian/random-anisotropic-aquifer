from mpi4py import MPI
import sys

if MPI.COMM_WORLD.rank == 0:
    print(f"n: {sys.argv[1]}")
    print(f"COMM_WORLD.size: MPI.COMM_WORLD.size")
