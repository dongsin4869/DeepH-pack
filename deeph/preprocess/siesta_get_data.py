import os

import h5py
import numpy as np
import sisl


def siesta_parse(input_path, output_path):
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    os.makedirs(output_path, exist_ok=True)

    # finds system name
    f_list = os.listdir(input_path)
    for f_name in f_list:
        if f_name[::-1][0:9] == "XDNI_BRO.":
            system_name = f_name[:-9]

    # Read lattice and atom positions from STRUCT_OUT
    with open(f"{input_path}/{system_name}.STRUCT_OUT", "r") as struct:
        lattice = np.empty((3, 3))
        for i in range(3):
            line = struct.readline()
            linesplit = line.split()
            lattice[i, :] = linesplit[:]
        np.savetxt(f"{output_path}/lat.dat", np.transpose(lattice), fmt="%.18e")

        line = struct.readline()
        linesplit = line.split()
        num_atoms = int(linesplit[0])
        atom_coord = np.empty((num_atoms, 4))
        for i in range(num_atoms):
            line = struct.readline()
            linesplit = line.split()
            atom_coord[i, :] = linesplit[1:]
        np.savetxt(f"{output_path}/element.dat", atom_coord[:, 0], fmt="%d")

    atom_coord_cart = np.genfromtxt(f"{input_path}/{system_name}.XV", skip_header=4)
    atom_coord_cart = atom_coord_cart[:, 2:5] * 0.529177249
    np.savetxt(f"{output_path}/site_positions.dat", np.transpose(atom_coord_cart))

    # Load orbital index
    orb_indx = np.genfromtxt(
        f"{input_path}/{system_name}.ORB_INDX", skip_header=3, skip_footer=17
    )

    # Save reciprocal lattice
    a1, a2, a3 = lattice[0, :], lattice[1, :], lattice[2, :]
    b1 = 2 * np.pi * np.cross(a2, a3) / np.dot(a1, np.cross(a2, a3))
    b2 = 2 * np.pi * np.cross(a3, a1) / np.dot(a2, np.cross(a3, a1))
    b3 = 2 * np.pi * np.cross(a1, a2) / np.dot(a3, np.cross(a1, a2))
    rlattice = np.array([b1, b2, b3])
    np.savetxt(f"{output_path}/rlat.dat", np.transpose(rlattice), fmt="%.18e")

    # Save orbital types
    with open(f"{output_path}/orbital_types.dat", "w") as orb_type_f:
        i = 0
        atom_current = 0
        while True:
            if atom_current != orb_indx[i, 1]:
                if atom_current != 0:
                    for j in range(4):
                        for _ in range(int(atom_orb_cnt[j] / (2 * j + 1))):
                            orb_type_f.write(f"{j}  ")
                    orb_type_f.write("\n")

                atom_current = int(orb_indx[i, 1])
                atom_orb_cnt = np.array([0, 0, 0, 0])

            l = int(orb_indx[i, 6])
            atom_orb_cnt[l] += 1
            i += 1
            if i > len(orb_indx) - 1:
                for j in range(4):
                    for _ in range(int(atom_orb_cnt[j] / (2 * j + 1))):
                        orb_type_f.write(f"{j}  ")
                orb_type_f.write("\n")
                break

    # Parse HSX using sisl for Hamiltonian and overlap matrices
    hsx_file = f"{input_path}/{system_name}.HSX"
    sile_hsx = sisl.get_sile(hsx_file)

    # Read the Hamiltonian and overlap matrices
    H = sile_hsx.read_hamiltonian()
    S = sile_hsx.read_overlap()

    nou = H.shape[0]
    nspin = H.spin.size if hasattr(H, "spin") else 1
    atoms = H.geometry.atoms
    na = H.geometry.na

    H_block_sparse = {}
    S_block_sparse = {}

    for i in range(nspin):
        for j in range(nou):
            tmpt = H.tocsr().getrow(j).data
            for k in range(len(tmpt)):
                if j < len(atoms) and k < len(atoms):
                    atom_1 = atoms[j].Z
                    atom_2 = atoms[k].Z
                    Rijk = H.geometry.cell.astype(int)
                    key = f"[{Rijk[0, 0]}, {Rijk[1, 1]}, {Rijk[2, 2]}, {atom_1}, {atom_2}]"

                    if key not in H_block_sparse:
                        H_block_sparse[key] = []

                    H_block_sparse[key].append([j, k, tmpt[k]])

    for j in range(nou):
        tmpt = S.tocsr().getrow(j).data
        for k in range(len(tmpt)):
            if j < len(atoms) and k < len(atoms):
                atom_1 = atoms[j].Z
                atom_2 = atoms[k].Z
                Rijk = S.geometry.cell.astype(int)
                key = f"[{Rijk[0, 0]}, {Rijk[1, 1]}, {Rijk[2, 2]}, {atom_1}, {atom_2}]"

                if key not in S_block_sparse:
                    S_block_sparse[key] = []

                S_block_sparse[key].append([j, k, tmpt[k]])

    # Save Hamiltonian and Overlap matrices to HDF5
    with h5py.File(f"{output_path}/hamiltonians.h5", "w") as f:
        for key, sparse_block in H_block_sparse.items():
            f[key] = sparse_block

    with h5py.File(f"{output_path}/overlaps.h5", "w") as f:
        for key, sparse_block in S_block_sparse.items():
            f[key] = sparse_block

    print("Parsing completed. Hamiltonian and overlap matrices saved to HDF5 files.")
