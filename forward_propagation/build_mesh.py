from mpi4py import MPI

import gmsh
import numpy as np
from default_parameters import create_default_parameters
from utils import print_all, print_root

from dolfinx import io
from dolfinx.io import gmshio

parameters = create_default_parameters()


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

    gdim = 3

    # Markers
    aquitard_marker = 1
    aquifer_marker = 2
    bed_marker = 3
    top_marker, sidesx_marker, sidesy_marker, bottom_marker, drywell_marker, pumpingwell_marker = (
        4,
        5,
        6,
        7,
        8,
        9,
    )
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
        bed = model.occ.addBox(0, 0, Lz1 + Lz2, Lx, Ly, Lz3)
        dry_well = model.occ.addCylinder(Lxw, Lyw, 0, 0, 0, Ld1, Lr)
        pumping_well = model.occ.addCylinder(Lxw, Lyw, Ld1, 0, 0, Ld2, Lr)
        model.occ.cut(
            [(3, aquitard), (3, aquifer), (3, bed)], [(3, dry_well), (3, pumping_well)]
        )
        model.occ.synchronize()

        volumes = gmsh.model.getEntities(dim=gdim)
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], aquitard_marker)
        gmsh.model.setPhysicalName(volumes[0][0], aquitard_marker, "aquitard volume")
        gmsh.model.addPhysicalGroup(volumes[1][0], [volumes[1][1]], aquifer_marker)
        gmsh.model.setPhysicalName(volumes[1][0], aquifer_marker, "aquifer volume")
        gmsh.model.addPhysicalGroup(volumes[2][0], [volumes[2][1]], bed_marker)
        gmsh.model.setPhysicalName(volumes[2][0], bed_marker, "bed volume")

        # Define boundaries
        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [Lx / 2, Ly / 2, 0]):
                top.append(boundary[1])
            elif (
                np.allclose(center_of_mass, [0, Ly / 2, Lz1 / 2])
                or np.allclose(center_of_mass, [0, Ly / 2, Lz1 + (Lz2 / 2)])
                or np.allclose(center_of_mass, [0, Ly / 2, Lz1 + Lz2 + (Lz3 / 2)])
                or np.allclose(center_of_mass, [Lx, Ly / 2, Lz1 / 2])
                or np.allclose(center_of_mass, [Lx, Ly / 2, Lz1 + (Lz2 / 2)])
                or np.allclose(center_of_mass, [Lx, Ly / 2, Lz1 + Lz2 + (Lz3 / 2)])
            ):
                sidesx.append(boundary[1])
            elif (
                np.allclose(center_of_mass, [Lx / 2, 0, Lz1 / 2])
                or np.allclose(center_of_mass, [Lx / 2, 0, Lz1 + (Lz2 / 2)])
                or np.allclose(center_of_mass, [Lx / 2, 0, Lz1 + Lz2 + (Lz3 / 2)])
                or np.allclose(center_of_mass, [Lx / 2, Ly, Lz1 / 2])
                or np.allclose(center_of_mass, [Lx / 2, Ly, Lz1 + (Lz2 / 2)])
                or np.allclose(center_of_mass, [Lx / 2, Ly, Lz1 + Lz2 + (Lz3 / 2)])
            ):
                sidesy.append(boundary[1])
            elif np.allclose(center_of_mass, [Lx / 2, Ly / 2, Lz]):
                bottom.append(boundary[1])
            elif np.allclose(center_of_mass, [Lxw, Lyw, Ld1 + (Ld2 / 2)]) or np.allclose(
                center_of_mass, [Lxw, Lyw, Ld1 + Ld2]
            ):
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
        gmsh.model.mesh.field.setNumbers(pumping_dist, "FacesList", pumpingwell)
        pumping_thre = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(pumping_thre, "IField", pumping_dist)
        gmsh.model.mesh.field.setNumber(pumping_thre, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(pumping_thre, "LcMax", 60 * res_min)
        gmsh.model.mesh.field.setNumber(pumping_thre, "DistMin", Lr / 2)
        gmsh.model.mesh.field.setNumber(pumping_thre, "DistMax", Lr)

        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "FacesList", drywell)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 60 * res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", Lr / 2)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", Lr)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field, pumping_thre])
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


def write_mesh(parameters):
    # Cteating the mesh
    model = create_mesh(parameters)

    gdim = 3

    # Interfacing with GMSH in DOLFINx
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    domain, mt, ft = gmshio.model_to_mesh(model, mesh_comm, gmsh_model_rank, gdim=gdim)
    gmsh.finalize()
    print_all("gmsh finalized")

    domain.name = "aquifersys"
    mt.name = f"{domain.name}_cells"
    ft.name = f"{domain.name}_facets"

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    with io.XDMFFile(domain.comm, "output/mesh/mesh.xdmf", "w") as mxdmf:
        mxdmf.write_mesh(domain)

    domain.topology.create_connectivity(2, 3)
    with io.XDMFFile(domain.comm, "output/mesh/subdomain_tags.xdmf", "w") as sxdmf:
        sxdmf.write_meshtags(
            mt,
            x=domain.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{domain.name}']/Geometry",
        )

    with io.XDMFFile(domain.comm, "output/mesh/boundaries_tags.xdmf", "w") as bxdmf:
        bxdmf.write_meshtags(
            ft,
            x=domain.geometry,
            geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{domain.name}']/Geometry",
        )


write_mesh(parameters)
