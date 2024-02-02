# Poroelasticity model for Anderson Junction aquifer

import numpy as np
import dolfinx
import ufl
import basix.ufl

import pprint
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io, mesh
from functools import partial
from utils import print_root, print_all
from dolfinx.fem import petsc, assemble_scalar, form


def boundary_conditions(parameters, domain, ft, V):
    pumpingrate = parameters["P_r"]

    top_marker, sidesx_marker, sidesy_marker, bottom_marker, drywell_marker, pumpingwell_marker = 4, 5, 6, 7, 8, 9
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    area = MPI.COMM_WORLD.allreduce(
        fem.assemble_scalar(fem.form(1.0*ds(pumpingwell_marker))))

    fdim = domain.topology.dim - 1

    U, _ = V.sub(2).collapse()
    u_D = fem.Function(U)

    UX, _ = V.sub(2).sub(0).collapse()
    u_Dx = fem.Function(UX)

    UY, _ = V.sub(2).sub(1).collapse()
    u_Dy = fem.Function(UY)

    # UZ, _ = V.sub(2).sub(2).collapse()
    # u_Dz = fem.Function(UZ)

    bcu_sidesx = fem.dirichletbc(u_Dx, fem.locate_dofs_topological(
        (V.sub(2).sub(0), UX), fdim, ft.find(sidesx_marker)), V.sub(2).sub(0))
    bcu_sidesy = fem.dirichletbc(u_Dy, fem.locate_dofs_topological(
        (V.sub(2).sub(1), UY), fdim, ft.find(sidesy_marker)), V.sub(2).sub(1))
    bcu_bottom = fem.dirichletbc(u_D, fem.locate_dofs_topological(
        (V.sub(2), U), fdim, ft.find(bottom_marker)), V.sub(2))
    # bcu_bottom = fem.dirichletbc(u_Dz, fem.locate_dofs_topological(
    #     (V.sub(2).sub(2), UZ), fdim, ft.find(bottom_marker)), V.sub(2).sub(2))

    Q, _ = V.sub(0).collapse()

    def f(x):
        values = np.zeros((x.shape[0], x.shape[1]))
        return values
    

    f_h = fem.Function(Q)
    f_h.interpolate(f)

    # Solve projection problem to find G \cdot n = g
    n = ufl.FacetNormal(domain)
    g = pumpingrate/area

    G_tr = ufl.TrialFunction(Q)
    G_t = ufl.TestFunction(Q)

    ds = ufl.Measure("ds", domain)
    dS = ufl.Measure("dS", domain)
    a = ufl.inner(ufl.inner(G_tr, n), ufl.inner(G_t, n)) * ds + \
        ufl.inner(ufl.inner(G_tr, n), ufl.inner(G_t, n))("+") * dS
    L = ufl.inner(g, ufl.inner(G_t, n)) * ds

    problem = dolfinx.fem.petsc.LinearProblem(a, L)
    G_h = problem.solve()

    bcp_bottom = fem.dirichletbc(f_h, fem.locate_dofs_topological(
        (V.sub(0), Q), fdim, ft.find(bottom_marker)), V.sub(0))
    bcp_top = fem.dirichletbc(f_h, fem.locate_dofs_topological(
        (V.sub(0), Q), fdim, ft.find(top_marker)), V.sub(0))
    bcp_sidesx = fem.dirichletbc(f_h, fem.locate_dofs_topological(
        (V.sub(0), Q), fdim, ft.find(sidesx_marker)), V.sub(0))
    bcp_sidesy = fem.dirichletbc(f_h, fem.locate_dofs_topological(
        (V.sub(0), Q), fdim, ft.find(sidesy_marker)), V.sub(0))
    bcp_drywell = fem.dirichletbc(f_h, fem.locate_dofs_topological(
        (V.sub(0), Q), fdim, ft.find(drywell_marker)), V.sub(0))
    bcp_pumpingwell = fem.dirichletbc(G_h, fem.locate_dofs_topological(
        (V.sub(0), Q), fdim, ft.find(pumpingwell_marker)), V.sub(0))

    bcs = {
        "bcu_bottom": bcu_bottom,
        # "bcuy_bottom": bcuy_bottom,
        "bcu_sidesx": bcu_sidesx,
        "bcu_sidesy": bcu_sidesy,
        "bcp_bottom": bcp_bottom,
        "bcp_top": bcp_top,
        "bcp_sidesx": bcp_sidesx,
        "bcp_sidesy": bcp_sidesy,
        "bcp_drywell": bcp_drywell,
        "bcp_pumpingwell": bcp_pumpingwell
    }

    return bcs


def equation_parameters(parameters, domain, mt):
    # parameters
    mu_f = parameters["mu_f"]

    # Aquitard layer
    alpha_aqtrd = parameters["alpha_aqtrd"]
    lmbda_aqtrd = parameters["lmbda_aqtrd"]
    G_aqtrd = parameters["G_aqtrd"]
    k_x_aqtrd = parameters["k_x_aqtrd"]
    k_y_aqtrd = parameters["k_y_aqtrd"]
    k_z_aqtrd = parameters["k_z_aqtrd"]
    S_epsilon_aqtrd = parameters["S_e_aqtrd"]

    # Aquifer layer
    alpha_aqfr = parameters["alpha_aqfr"]
    lmbda_aqfr = parameters["lmbda_aqfr"]
    G_aqfr = parameters["G_aqfr"]
    k_x = parameters["k_x_aqfr"]
    k_xy = parameters["k_xy_aqfr"]
    k_yx = parameters["k_yx_aqfr"]
    k_y = parameters["k_y_aqfr"]
    k_z = parameters["k_z_aqfr"]
    S_epsilon_aqfr = parameters["S_e_aqfr"]

    # Bed layer
    alpha_bed = parameters["alpha_bed"]
    lmbda_bed = parameters["lmbda_bed"]
    G_bed = parameters["G_bed"]
    k_x_bed = parameters["k_x_bed"]
    k_y_bed = parameters["k_y_bed"]
    k_z_bed = parameters["k_z_bed"]
    S_epsilon_bed = parameters["S_e_bed"]

    f_p = fem.Constant(domain, PETSc.ScalarType(0.0))

    # Subdomain
    # K = fem.FunctionSpace(domain, ufl.TensorElement("DG", domain.ufl_cell(), 0))
    # P = fem.FunctionSpace(domain, ("DG", 0))
    K = dolfinx.fem.functionspace(domain, basix.ufl.element("DG", "tetrahedron", 0, shape=(3,3)))
    P = dolfinx.fem.functionspace(domain, basix.ufl.element("Discontinuous Lagrange", "tetrahedron", 0))
    invkappa = fem.Function(K, dtype=PETSc.ScalarType)

    def invk_p(x, res):
        values = np.zeros(
            (domain.geometry.dim*domain.geometry.dim, x.shape[1]), dtype=np.float64)
        values[0] = res[0, 0]
        values[1] = res[0, 1]
        values[2] = res[0, 2]
        values[3] = res[1, 0]
        values[4] = res[1, 1]
        values[5] = res[1, 2]
        values[6] = res[2, 0]
        values[7] = res[2, 1]
        values[8] = res[2, 2]
        return values

    material_tags = np.unique(mt.values)
    S_e = fem.Function(P)
    lmbda = fem.Function(P)
    G = fem.Function(P)
    alpha = fem.Function(P)
    for tag in material_tags:
        cells = mt.find(tag)
        if tag == 1:
            res = np.linalg.inv(np.array(
                [[k_x_aqtrd/mu_f, 0, 0], [0, k_y_aqtrd/mu_f, 0], [0, 0, k_z_aqtrd/mu_f]]))
            invk = partial(invk_p, res=res)
            invkappa.interpolate(invk, cells)
            S_e.x.array[cells] = np.full_like(
                cells, S_epsilon_aqtrd, dtype=PETSc.ScalarType)
            lmbda.x.array[cells] = np.full_like(
                cells, lmbda_aqtrd, dtype=PETSc.ScalarType)
            G.x.array[cells] = np.full_like(
                cells, G_aqtrd, dtype=PETSc.ScalarType)
            alpha.x.array[cells] = np.full_like(
                cells, alpha_aqtrd, dtype=PETSc.ScalarType)
        elif tag == 2:
            res = np.linalg.inv(
                np.array([[k_x/mu_f, k_xy/mu_f, 0], [k_yx/mu_f, k_y/mu_f, 0], [0, 0, k_z/mu_f]]))
            invk = partial(invk_p, res=res)
            invkappa.interpolate(invk, cells)
            S_e.x.array[cells] = np.full_like(
                cells, S_epsilon_aqfr, dtype=PETSc.ScalarType)
            lmbda.x.array[cells] = np.full_like(
                cells, lmbda_aqfr, dtype=PETSc.ScalarType)
            G.x.array[cells] = np.full_like(
                cells, G_aqfr, dtype=PETSc.ScalarType)
            alpha.x.array[cells] = np.full_like(
                cells, alpha_aqfr, dtype=PETSc.ScalarType)
        elif tag == 3:
            res = np.linalg.inv(
                np.array([[k_x_bed/mu_f, 0, 0], [0, k_y_bed/mu_f, 0], [0, 0, k_z_bed/mu_f]]))
            invk = partial(invk_p, res=res)
            invkappa.interpolate(invk, cells)
            S_e.x.array[cells] = np.full_like(
                cells, S_epsilon_bed, dtype=PETSc.ScalarType)
            lmbda.x.array[cells] = np.full_like(
                cells, lmbda_bed, dtype=PETSc.ScalarType)
            G.x.array[cells] = np.full_like(
                cells, G_bed, dtype=PETSc.ScalarType)
            alpha.x.array[cells] = np.full_like(
                cells, alpha_bed, dtype=PETSc.ScalarType)

    return f_p, invkappa, S_e, lmbda, G, alpha


def solve(parameters):
    print_root(pprint.pformat(parameters))
    # Define temporal parameters
    t = parameters["t"]
    T = parameters["T"]
    num_steps = parameters["num_steps"]
    dt = T / num_steps  # time step size
    T2 = parameters["T2"]
    num_steps2 = parameters["num_steps2"]
    dt2 = (T2-T) / num_steps2

    # Define mesh
    Lx = parameters["Lx"]
    Ly = parameters["Ly"]
    Lz1 = parameters["Lz1"]
    Lz2 = parameters["Lz2"]
    Lz3 = parameters["Lz3"]
    Lz = Lz1 + Lz2 + Lz3

    gdim = parameters["gdim"]

    top_marker, sidesx_marker, sidesy_marker, bottom_marker, drywell_marker, pumpingwell_marker = 4, 5, 6, 7, 8, 9

    domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([Lx, Ly, Lz])],
                            [20, 6, 6], cell_type=mesh.CellType.tetrahedron)  
    
    domain.name = "aquifersys"
      
    print_root("Reading mesh...")
    with io.XDMFFile(MPI.COMM_WORLD, "output/mesh/mesh.xdmf", "r") as mrxdmf:
        domain = mrxdmf.read_mesh(name=domain.name)

    with io.XDMFFile(MPI.COMM_WORLD, "output/mesh/subdomain_tags.xdmf", "r") as srxdmf:
        mt = srxdmf.read_meshtags(domain, name=f"{domain.name}_cells")
        
    domain.topology.create_connectivity(2, 3)
    with io.XDMFFile(MPI.COMM_WORLD, "output/mesh/boundaries_tags.xdmf", "r") as brxdmf:    
        ft = brxdmf.read_meshtags(domain, name=f"{domain.name}_facets", xpath="Xdmf/Domain")

    # Defining the finite element function space
    # Q_el = ufl.FiniteElement("BDM", domain.ufl_cell(), 1)
    # P_el = ufl.FiniteElement("Discontinuous Lagrange", domain.ufl_cell(), 0)
    # U_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
    # V_el = ufl.MixedElement([Q_el, P_el, U_el])
    V_el = basix.ufl.mixed_element([basix.ufl.element("BDM", "tetrahedron", 1),
                                   basix.ufl.element("Discontinuous Lagrange", "tetrahedron", 0),
                                   basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))])
    V = dolfinx.fem.functionspace(domain, V_el)

    (q, p, u) = ufl.TrialFunctions(V)
    (q_t, p_t, u_t) = ufl.TestFunctions(V)

    Q, _ = V.sub(0).collapse()
    q_n = fem.Function(Q)
    q_n.name = "q_n"

    P, _ = V.sub(1).collapse()
    p_n = fem.Function(P)
    p_n.name = "p_n"

    U, _ = V.sub(2).collapse()
    u_n = fem.Function(U)
    u_n.name = "u_n"

    # Defining boundary condition
    print_root("Defining boundary conditions...")
    bcs_dict = boundary_conditions(parameters, domain, ft, V)
    print_root("Done.")
    bcs = list(bcs_dict.values())

    # Defining solution variable
    ah = fem.Function(V)

    # Parameters in equation
    print_root("Defining parameters...")
    [f_p, invkappa, S_e, lmbda, G, alpha] = equation_parameters(parameters,
                                                                domain, mt)
    print_root("Done.")

    # Stress functions definition
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def stress(u):
        return 2.0 * G * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(
            domain.topology.dim)

    def stress_bar(u, p):
        return stress(u) - alpha * p * ufl.Identity(domain.topology.dim)

    dx = ufl.Measure("dx", domain=domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=ft)
    n = ufl.FacetNormal(domain)

    # Weak Forms #
    a = ufl.inner(S_e * p, p_t) * dx + ufl.inner(alpha * ufl.div(u), p_t) * dx\
        + ufl.inner(dt * ufl.div(q), p_t) * dx \
        - ufl.inner(stress_bar(u, p), ufl.grad(u_t)) * dx \
        + ufl.inner(dt * p, ufl.div(q_t)) * dx\
        - ufl.inner(dt * invkappa * q, q_t) * dx\
        - (ufl.dot((stress_bar(u, p) * n), n))*(ufl.dot(u_t, n)) * ds(drywell_marker)\
        + (ufl.dot((stress_bar(u_t, p_t) * n), n)) * \
        (ufl.dot(u, n)) * ds(drywell_marker)
    L = ufl.inner(dt * f_p, p_t) * dx + ufl.inner(S_e * p_n, p_t) * dx\
        + ufl.inner(alpha * ufl.div(u_n), p_t) * dx
    # - ufl.inner(g_u, u_t) * ds(top_marker) \
    # - ufl.inner(f_u, u_t) * dx \
    # + ufl.inner(dt * p_d, ufl.inner(q_t, n)) * ds(4)

    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    print_root("Assembling bilinear form...")
    A = fem.petsc.assemble_matrix(bilinear_form, bcs=bcs)
    A.assemble()
    print_root("Done.")
    b = fem.petsc.create_vector(linear_form)

    # Define solver
    options = PETSc.Options()
    options["ksp_type"] = "preonly"
    options["pc_type"] = "lu"
    options["pc_factor_mat_solver_type"] = "mumps"

    print_root("Setting up KSP...")
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setFromOptions()
    print_root("Done.")

    # LOS displacement
    teta = np.radians(180)  # Rotation around the axis y
    phi = np.radians(36)  # Rotation around the axis z
    R1 = np.array([[np.cos(teta), 0, -np.sin(teta)],
                   [0, 1, 0], [np.sin(teta), 0, np.cos(teta)]])
    R2 = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

    incidence_angle = np.radians(43.86)
    satellite_heading = np.radians(-15)
    LOS = np.array([[np.sin(incidence_angle)*np.cos(satellite_heading), np.sin(
        incidence_angle)*np.sin(satellite_heading), np.cos(incidence_angle)]])

    T_list = LOS@R2@R1

    # Interpolate q into a different finite element space
    # Q0 = fem.VectorFunctionSpace(domain, ("Discontinuous Lagrange", 1))
    Q0 = dolfinx.fem.functionspace(domain, basix.ufl.element("Discontinuous Lagrange", "tetrahedron", 1, shape=(3,)))
    qh_Q0 = fem.Function(Q0, dtype=PETSc.ScalarType)
    qh_Q0.interpolate(q_n)

    # P0 = fem.FunctionSpace(domain, ("Discontinuous Lagrange", 1))
    P0 = dolfinx.fem.functionspace(domain, basix.ufl.element("Discontinuous Lagrange", "tetrahedron", 1))
    ph_P0 = fem.Function(P0, dtype=PETSc.ScalarType)
    ph_P0.interpolate(p_n)

    # W = fem.FunctionSpace(domain, ("Lagrange", 1))
    W = dolfinx.fem.functionspace(domain, basix.ufl.element("Lagrange", "tetrahedron", 1))
    T_ufl = ufl.as_matrix(T_list)
    u_los = ufl.dot(T_ufl, u_n)
    u_los_expr = fem.Expression(u_los, W.element.interpolation_points())
    
    u_los_h = fem.Function(W)
    u_los_h.interpolate(u_los_expr)

    pfile_vtx = io.VTXWriter(domain.comm, f"{parameters['output_dir']}/pressure.bp", [ph_P0], engine="BP4")
    pfile_vtx.write(t)

    qfile_vtx = io.VTXWriter(domain.comm, f"{parameters['output_dir']}/flux.bp", [qh_Q0], engine="BP4")
    qfile_vtx.write(t)

    ufile_vtx = io.VTXWriter(domain.comm, f"{parameters['output_dir']}/deformation.bp", [u_n], engine="BP4")
    ufile_vtx.write(t)

    losfile_vtx = io.VTXWriter(domain.comm, f"{parameters['output_dir']}/deformation_LOS.bp", [
        u_los_h], engine="BP4")
    losfile_vtx.write(t)

    # Submesh to reduce output size
    cells = dolfinx.mesh.compute_incident_entities(domain.topology, ft.find(top_marker), domain.topology.dim-1, domain.topology.dim)


    submesh, cell_map, _, _ = dolfinx.mesh.create_submesh(domain, domain.topology.dim, cells)

    U_sub = dolfinx.fem.functionspace(submesh, U.ufl_element())
    u_n_sub = dolfinx.fem.Function(U_sub)

    num_sub_cells = submesh.topology.index_map(submesh.topology.dim).size_local
    # for cell in range(num_sub_cells):
    #     sub_dofs = U_sub.dofmap.cell_dofs(cell)
    #     parent_dofs = U.dofmap.cell_dofs(cell_map[cell])
    #     assert U_sub.dofmap.bs == U.dofmap.bs
    #     for parent, child in zip(parent_dofs, sub_dofs):
    #         for bb in range(U_sub.dofmap.bs):
    #             u_n_sub.x.array[child*U_sub.dofmap.bs +
    #                         bb] = u_n.x.array[parent*U.dofmap.bs+bb]
                
    with dolfinx.io.XDMFFile(submesh.comm, f"{parameters['output_dir']}/submesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(submesh)
        xdmf.write_function(u_n_sub, t)


    # with dolfinx.io.XDMFFile(submesh.comm, "parentmesh.xdmf", "w") as xdmf:
    #     xdmf.write_mesh(mesh)
    #     xdmf.write_function(u)

    print_root("Starting timestepping...")

    for i in range(num_steps):
        # Updating the solution and right hand side per time step
        print_root(f"Started time step: {i + 1}")
        t += dt

        # Update the right hand side reusing the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        fem.petsc.assemble_vector(b, linear_form)

        # Apply Dirichlet boundary condition to the vector
        fem.petsc.apply_lifting(b, [bilinear_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs)

        # Solve linear problem
        print_root("Starting linear solve...")
        solver.solve(b, ah.vector)
        ah.x.scatter_forward()
        u_n_sub.x.scatter_forward()
        print_root("Finished linear solve.")
        # print(ah.x.norm())

        qh = ah.sub(0).collapse()
        ph = ah.sub(1).collapse()
        uh = ah.sub(2).collapse()

        u_los_h.interpolate(u_los_expr)

        # Update solution at previous time step (u_n)
        p_n.x.array[:] = ph.x.array
        q_n.x.array[:] = qh.x.array
        u_n.x.array[:] = uh.x.array
        for cell in range(num_sub_cells):
            sub_dofs = U_sub.dofmap.cell_dofs(cell)
            parent_dofs = U.dofmap.cell_dofs(cell_map[cell])
            assert U_sub.dofmap.bs == U.dofmap.bs
            for parent, child in zip(parent_dofs, sub_dofs):
                for bb in range(U_sub.dofmap.bs):
                    u_n_sub.x.array[child*U_sub.dofmap.bs +
                                bb] = u_n.x.array[parent*U.dofmap.bs+bb]

        if (i+1) % 20 == 0:
            # Interpolate q into a different finite element space
            qh_Q0.interpolate(q_n)
            ph_P0.interpolate(p_n)

            # Write solution to file
            pfile_vtx.write(t)
            qfile_vtx.write(t)
            ufile_vtx.write(t)
            losfile_vtx.write(t)
            xdmf.write_function(u_n_sub, t)

    print_root("Stop pumping.")
    print_root("Recalculating Dirichlet condition...")
    def f(x):
        values = np.zeros((x.shape[0], x.shape[1]))
        return values
    f_h = fem.Function(Q)
    f_h.interpolate(f)

    fdim = domain.topology.dim - 1

    bcp_pumpingwell = fem.dirichletbc(f_h, fem.locate_dofs_topological(
        (V.sub(0), Q), fdim, ft.find(pumpingwell_marker)), V.sub(0))
    bcs_dict["bcp_pumpingwell"] = bcp_pumpingwell
    bcs = list(bcs_dict.values())

    # bcs = [bcu_bottom, bcu_sidesx, bcu_sidesy, bcux_drywell, bcuy_drywell,
    #        bcux_pumpingwell, bcuy_pumpingwell, bcp_bottom, bcp_top, bcp_sidesx,
    #        bcp_sidesy, bcp_drywell, bcp_pumpingwell]
    print_root("Done.")

    # NOTE: No new KSP operator! Might be OK, needs checking.
    A = fem.petsc.assemble_matrix(bilinear_form, bcs=bcs)
    A.assemble()
    b = fem.petsc.create_vector(linear_form)
        
    for i in range(num_steps2):
        print_root(f"Started time step: {i + 361}")
        t += dt2

        # Update the right hand side reusing the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)

        fem.petsc.assemble_vector(b, linear_form)

        # Apply Dirichlet boundary condition to the vector
        fem.petsc.apply_lifting(b, [bilinear_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, bcs)

        # Solve linear problem
        print_root("Starting linear solve...")
        solver.solve(b, ah.vector)
        ah.x.scatter_forward()
        u_n_sub.x.scatter_forward()
        print_root("Finished linear solve.")
        # print(ah.x.norm())

        qh = ah.sub(0).collapse()
        ph = ah.sub(1).collapse()
        uh = ah.sub(2).collapse()

        # Interpolate LOS displacement
        u_los_h.interpolate(u_los_expr)

        # Update solution at previous time step (u_n)
        p_n.x.array[:] = ph.x.array
        q_n.x.array[:] = qh.x.array
        u_n.x.array[:] = uh.x.array
        for cell in range(num_sub_cells):
            sub_dofs = U_sub.dofmap.cell_dofs(cell)
            parent_dofs = U.dofmap.cell_dofs(cell_map[cell])
            assert U_sub.dofmap.bs == U.dofmap.bs
            for parent, child in zip(parent_dofs, sub_dofs):
                for bb in range(U_sub.dofmap.bs):
                    u_n_sub.x.array[child*U_sub.dofmap.bs +
                                bb] = u_n.x.array[parent*U.dofmap.bs+bb]

        if (i + 1) % 20 == 0:
            # Interpolate q into a different finite element space
            qh_Q0.interpolate(q_n)
            ph_P0.interpolate(p_n)

            # Write solution to file
            pfile_vtx.write(t)
            qfile_vtx.write(t)
            ufile_vtx.write(t)
            losfile_vtx.write(t)
            xdmf.write_function(u_n_sub, t)

    pfile_vtx.close()
    qfile_vtx.close()
    ufile_vtx.close()
    losfile_vtx.close()

    print_root('Finished solve.')
