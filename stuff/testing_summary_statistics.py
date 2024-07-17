import numpy as np
from mpi4py import MPI
import dolfinx
import dolfinx.mesh as mesh
import dolfinx.fem as fem
import basix
import basix.ufl

# Calculate mean
msh = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
element = basix.ufl.element("Lagrange", msh.basix_cell(), 1)

V = fem.functionspace(msh, element)

n = 10
r = range(1, n)
fs = [fem.Function(V) for i in r]
for i, f in zip(r, fs):
    f.x.array[:] = np.float64(i)


def mean(fs):
    m = fem.Function(V)
    for f in fs:
        m.x.array[:] += f.x.array
    m.x.array[:] /= len(fs)
    return m


# Mean of arithmetic series https://en.wikipedia.org/wiki/Arithmetic_progression
m = mean(fs)
np.testing.assert_allclose(m.x.array, (r[0] + r[-1]) / 2.0)


def variance(fs):
    m = mean(fs)
    var = fem.Function(V)

    for f in fs:
        var.x.array[:] += (f.x.array - m.x.array) ** 2

    var.x.array[:] /= len(fs)

    return var

# Variance of arithmetic series https://en.wikipedia.org/wiki/Arithmetic_progression
var = variance(fs)
np.testing.assert_allclose(var.x.array, ((len(r) - 1) * (len(r) + 1)) / 12)
