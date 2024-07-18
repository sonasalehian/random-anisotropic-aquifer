import numpy as np
from mpi4py import MPI
import dolfinx.mesh as mesh
import dolfinx.fem as fem
import basix
import basix.ufl
import adios4dolfinx
import pathlib
import shutil

# Calculate mean
msh = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
element = basix.ufl.element("Lagrange", msh.basix_cell(), 1)

V = fem.functionspace(msh, element)

n = 10
# Arithmetic progression with unit difference between consecutive terms
r = range(1, n)
fs = [fem.Function(V) for _ in r]
# TODO: Add on mean function.
for i, f in zip(r, fs):
    f.x.array[:] = np.float64(i)


def mean(fs):
    m = fem.Function(V)
    for f in fs:
        m.x.array[:] += f.x.array
    m.x.array[:] /= len(fs)
    return m


m = mean(fs)
# Mean of arithmetic series https://en.wikipedia.org/wiki/Arithmetic_progression
np.testing.assert_allclose(m.x.array, (r[-1] + r[0]) / 2)


def variance(fs):
    m = mean(fs)
    var = fem.Function(V)

    # TODO: Is this numerically stable?
    for f in fs:
        var.x.array[:] += (f.x.array - m.x.array) ** 2

    var.x.array[:] /= len(fs)

    return var


var = variance(fs)
# Variance of arithmetic series https://en.wikipedia.org/wiki/Arithmetic_progression
np.testing.assert_allclose(var.x.array, ((len(r) - 1.0) * (len(r) + 1.0)) / 12.0)

# That works, let's try writing things to files.
filename = pathlib.Path("output/test.bp")
# Delete any existing file
if MPI.COMM_WORLD.rank == 0:
    shutil.rmtree(filename)
MPI.COMM_WORLD.Barrier()

for i, f in zip(r, fs):
    adios4dolfinx.write_function(filename, f, "BP4", time=float(i))

del fs

fs = []
for i in r:
    f = fem.Function(V)
    adios4dolfinx.read_function(filename, f, "BP4", time=float(i))
    fs.append(f)

m = mean(fs)
np.testing.assert_allclose(m.x.array, (r[-1] + r[0]) / 2)

var = variance(fs)
np.testing.assert_allclose(var.x.array, ((len(r) - 1.0) * (len(r) + 1.0)) / 12.0)

del fs


class TimeReader:
    def __init__(self, file, timesteps):
        self.file = file
        self.timesteps = timesteps
        self.current = -1
        self.f = fem.Function(V)

    def __len__(self):
        return len(self.timesteps)

    def __iter__(self):
        return TimeReader(self.file, self.timesteps)

    def __next__(self):
        self.current += 1
        if self.current < len(self):
            adios4dolfinx.read_function(
                self.file, self.f, "BP4", time=float(self.timesteps[self.current])
            )
            return self.f
        raise StopIteration


timeseries = TimeReader(filename, r)
m = mean(timeseries)
np.testing.assert_allclose(m.x.array, (r[-1] + r[0]) / 2)

m = mean(timeseries)
np.testing.assert_allclose(m.x.array, (r[-1] + r[0]) / 2)

# var = var(timeseries)
# np.testing.assert_allclose(var.x.array, ((len(r) - 1.0) * (len(r) + 1.0)) / 12.0)
