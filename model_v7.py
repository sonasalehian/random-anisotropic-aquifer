# Poroelasticity model for Anderson Junction aquifer

import numpy as np
import dolfinx
import gmsh
import ufl

import pprint
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, io
from dolfinx.io import gmshio
from functools import partial
from utils import print_root, print_all
from dolfinx.fem import petsc, assemble_scalar, form


def create_mesh(parameters):
    # Parameters
    Lx = parameters["Lx"]
    Ly = parameters["Ly"]
    Lxw = parameters["Lxw"]
    Lyw = parameters["Lyw"]
    Lz1 = parameters["Lz1"]
    Lz2 = parameters["Lz2"]
    Lz3 = parameters["Lz3"]
    Ld1 = parameters["Ld1"]
    Ld2 = parameters["Ld2"]
    Lr = parameters["Lr"]
    Lz = Lz1 + Lz2 + Lz3

    gdim = parameters["gdim"]

    # Markers
    aquitard_marker = 1
    aquifer_marker = 2
    bed_marker = 3
    top_marker, sidesx_marker, sidesy_marker, bottom_marker, drywell_marker, pumpingwell_marker = 4, 5, 6, 7, 8, 9
    top, sidesx, sidesy, bottom, drywell, pumpingwell = [], [], [], [], [], []

    model = None
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    gmsh.initialize()
    if mesh_comm.rank == model_rank:
        print_root(f"Generating mesh on rank {model_rank}...")
        model = gmsh.model()
        model.add("Aquifer with well")
        aquitard = model.occ.addBox(0, 0, 0, Lx, Ly, Lz1)
        model.occ.synchronize()
        gmsh.model.occ.mesh.setSize(model.getEntities(dim=0), 39)
        aquifer = model.occ.addBox(0, 0, Lz1, Lx, Ly, Lz2)
        bed = model.occ.addBox(0, 0, Lz1+Lz2, Lx, Ly, Lz3)
        dry_well = model.occ.addCylinder(Lxw, Lyw, 0, 0, 0, Ld1, Lr)
        pumping_well = model.occ.addCylinder(Lxw, Lyw, Ld1, 0, 0, Ld2, Lr)
        model_dim_tags = model.occ.cut([(3, aquitard), (3, aquifer), (3, bed)], [
                                       (3, dry_well), (3, pumping_well)])
        model.occ.synchronize()

        volumes = gmsh.model.getEntities(dim=gdim)
        gmsh.model.addPhysicalGroup(
            volumes[0][0], [volumes[0][1]], aquitard_marker)
        gmsh.model.setPhysicalName(
            volumes[0][0], aquitard_marker, "aquitard volume")
        gmsh.model.addPhysicalGroup(
            volumes[1][0], [volumes[1][1]], aquifer_marker)
        gmsh.model.setPhysicalName(
            volumes[1][0], aquifer_marker, "aquifer volume")
        gmsh.model.addPhysicalGroup(volumes[2][0], [volumes[2][1]], bed_marker)
        gmsh.model.setPhysicalName(volumes[2][0], bed_marker, "bed volume")

        # Define boundaries
        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(
                boundary[0], boundary[1])
            if np.allclose(center_of_mass, [Lx/2, Ly/2, 0]):
                top.append(boundary[1])
            elif np.allclose(center_of_mass, [0, Ly/2, Lz1/2]) or np.allclose(center_of_mass, [0, Ly/2, Lz1+(Lz2/2)]) or np.allclose(center_of_mass, [0, Ly/2, Lz1+Lz2+(Lz3/2)]) or np.allclose(center_of_mass, [Lx, Ly/2, Lz1/2]) or np.allclose(center_of_mass, [Lx, Ly/2, Lz1+(Lz2/2)]) or np.allclose(center_of_mass, [Lx, Ly/2, Lz1+Lz2+(Lz3/2)]):
                sidesx.append(boundary[1])
            elif np.allclose(center_of_mass, [Lx/2, 0, Lz1/2]) or np.allclose(center_of_mass, [Lx/2, 0, Lz1+(Lz2/2)]) or np.allclose(center_of_mass, [Lx/2, 0, Lz1+Lz2+(Lz3/2)]) or np.allclose(center_of_mass, [Lx/2, Ly, Lz1/2]) or np.allclose(center_of_mass, [Lx/2, Ly, Lz1+(Lz2/2)]) or np.allclose(center_of_mass, [Lx/2, Ly, Lz1+Lz2+(Lz3/2)]):
                sidesy.append(boundary[1])
            elif np.allclose(center_of_mass, [Lx/2, Ly/2, Lz]):
                bottom.append(boundary[1])
            elif np.allclose(center_of_mass, [Lxw, Lyw, Ld1+(Ld2/2)]) or np.allclose(center_of_mass, [Lxw, Lyw, Ld1+Ld2]):
                pumpingwell.append(boundary[1])
            else:
                drywell.append(boundary[1])
        gmsh.model.addPhysicalGroup(2, top, top_marker)
        gmsh.model.setPhysicalName(2, top_marker, "top")
        gmsh.model.addPhysicalGroup(2, sidesx, sidesx_marker)
        gmsh.model.setPhysicalName(2, sidesx_marker, "sidesx")
        gmsh.model.addPhysicalGroup(2, sidesy, sidesy_marker)
        gmsh.model.setPhysicalName(2, sidesy_marker, "sidesy")
        gmsh.model.addPhysicalGroup(2, bottom, bottom_marker)
        gmsh.model.setPhysicalName(2, bottom_marker, "bottom")
        gmsh.model.addPhysicalGroup(2, drywell, drywell_marker)
        gmsh.model.setPhysicalName(2, drywell_marker, "drywell")
        gmsh.model.addPhysicalGroup(2, pumpingwell, pumpingwell_marker)
        gmsh.model.setPhysicalName(2, pumpingwell_marker, "pumpingwell")

        # Set mesh size
        res_min = Lr / 2
        pumping_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            pumping_dist, "FacesList", pumpingwell)
        pumping_thre = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(pumping_thre, "IField", pumping_dist)
        gmsh.model.mesh.field.setNumber(pumping_thre, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(pumping_thre, "LcMax", 60*res_min)
        gmsh.model.mesh.field.setNumber(pumping_thre, "DistMin", Lr/2)
        gmsh.model.mesh.field.setNumber(pumping_thre, "DistMax", Lr)

        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "FacesList", drywell)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(
            threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 60 * res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", Lr/2)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", Lr)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [
                                         threshold_field, pumping_thre])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 6)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 40)

        gmsh.model.occ.synchronize()
        model.mesh.generate(gdim)
        print_root("Finished mesh generation.")

    print_all("At barrier...")
    mesh_comm.barrier()
    print_all("Passed barrier.")
    return model


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
    K = fem.FunctionSpace(domain, ufl.TensorElement("DG", domain.ufl_cell(), 0))
    P = fem.FunctionSpace(domain, ("DG", 0))
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
                np.array([[k_x/mu_f, 0, 0], [0, k_y/mu_f, 0], [0, 0, k_z/mu_f]]))
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

    gdim = parameters["gdim"]

    top_marker, sidesx_marker, sidesy_marker, bottom_marker, drywell_marker, pumpingwell_marker = 4, 5, 6, 7, 8, 9

    # Cteating the mesh
    model = create_mesh(parameters)

    # Interfacing with GMSH in DOLFINx
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    domain, mt, ft = gmshio.model_to_mesh(model, mesh_comm, gmsh_model_rank,
                                          gdim=gdim)
    gmsh.finalize()
    print_all("gmsh finalized")

    domain.name = "aquifersys"
    mt.name = f"{domain.name}_cells"
    ft.name = f"{domain.name}_facets"

    domain.topology.create_connectivity(domain.topology.dim-1,
                                        domain.topology.dim)
    with io.XDMFFile(domain.comm, f"{parameters['output_dir']}/facet_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        domain.topology.create_connectivity(2, 3)
        xdmf.write_meshtags(
            mt, x=domain.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{domain.name}']/Geometry")
        xdmf.write_meshtags(
            ft, x=domain.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{domain.name}']/Geometry")

    # Defining the finite element function space
    Q_el = ufl.FiniteElement("BDM", domain.ufl_cell(), 1)
    P_el = ufl.FiniteElement("Discontinuous Lagrange", domain.ufl_cell(), 0)
    U_el = ufl.VectorElement("Lagrange", domain.ufl_cell(), 1)
    V_el = ufl.MixedElement([Q_el, P_el, U_el])
    V = fem.FunctionSpace(domain, V_el)

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
    Q0 = fem.VectorFunctionSpace(domain, ("Discontinuous Lagrange", 1))
    qh_Q0 = fem.Function(Q0, dtype=PETSc.ScalarType)
    qh_Q0.interpolate(q_n)

    P0 = fem.FunctionSpace(domain, ("Discontinuous Lagrange", 1))
    ph_P0 = fem.Function(P0, dtype=PETSc.ScalarType)
    ph_P0.interpolate(p_n)

    W = fem.FunctionSpace(domain, ("Lagrange", 1))
    T_ufl = ufl.as_matrix(T_list)
    u_los = ufl.dot(T_ufl, u_n)
    u_los_expr = fem.Expression(u_los, W.element.interpolation_points())
    
    u_los_h = fem.Function(W)
    u_los_h.interpolate(u_los_expr)

    pfile_vtx = io.VTXWriter(domain.comm, f"{parameters['output_dir']}/pressure.bp", [ph_P0])
    pfile_vtx.write(t)

    qfile_vtx = io.VTXWriter(domain.comm, f"{parameters['output_dir']}/flux.bp", [qh_Q0])
    qfile_vtx.write(t)

    ufile_vtx = io.VTXWriter(domain.comm, f"{parameters['output_dir']}/deformation.bp", [u_n])
    ufile_vtx.write(t)

    losfile_vtx = io.VTXWriter(domain.comm, f"{parameters['output_dir']}/deformation_LOS.bp", [
        u_los_h])
    losfile_vtx.write(t)

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
        print_root("Finished linear solve.")
        print(ah.x.norm())

        qh = ah.sub(0).collapse()
        ph = ah.sub(1).collapse()
        uh = ah.sub(2).collapse()

        u_los_h.interpolate(u_los_expr)

        # Update solution at previous time step (u_n)
        p_n.x.array[:] = ph.x.array
        q_n.x.array[:] = qh.x.array
        u_n.x.array[:] = uh.x.array

        if (i+1) % 20 == 0:
            # Interpolate q into a different finite element space
            qh_Q0.interpolate(q_n)
            ph_P0.interpolate(p_n)

            # Write solution to file
            pfile_vtx.write(t)
            qfile_vtx.write(t)
            ufile_vtx.write(t)
            losfile_vtx.write(t)

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
        print_root("Finished linear solve.")
        print(ah.x.norm())

        qh = ah.sub(0).collapse()
        ph = ah.sub(1).collapse()
        uh = ah.sub(2).collapse()

        # Interpolate LOS displacement
        u_los_h.interpolate(u_los_expr)

        # Update solution at previous time step (u_n)
        p_n.x.array[:] = ph.x.array
        q_n.x.array[:] = qh.x.array
        u_n.x.array[:] = uh.x.array

        if (i + 1) % 20 == 0:
            # Interpolate q into a different finite element space
            qh_Q0.interpolate(q_n)
            ph_P0.interpolate(p_n)

            # Write solution to file
            pfile_vtx.write(t)
            qfile_vtx.write(t)
            ufile_vtx.write(t)
            losfile_vtx.write(t)

    pfile_vtx.close()
    qfile_vtx.close()
    ufile_vtx.close()
    losfile_vtx.close()

    print_root('Finished solve.')
