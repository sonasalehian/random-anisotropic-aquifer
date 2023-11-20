from mpi4py import MPI

def print_root(string):
    if MPI.COMM_WORLD.rank == 0:
        print(str(MPI.COMM_WORLD.rank) + ": " + string, flush=True)

def print_all(string):
    print(str(MPI.COMM_WORLD.rank) + ": " + string, flush=True)
