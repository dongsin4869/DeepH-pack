import json
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

    # Save reciprocal lattice
    a1, a2, a3 = lattice[0, :], lattice[1, :], lattice[2, :]
    b1 = 2 * np.pi * np.cross(a2, a3) / np.dot(a1, np.cross(a2, a3))
    b2 = 2 * np.pi * np.cross(a3, a1) / np.dot(a2, np.cross(a3, a1))
    b3 = 2 * np.pi * np.cross(a1, a2) / np.dot(a3, np.cross(a1, a2))
    rlattice = np.array([b1, b2, b3])
    np.savetxt(f"{output_path}/rlat.dat", np.transpose(rlattice), fmt="%.18e")

    # Parse HSX using sisl for Hamiltonian and overlap matrices
    hsx_file = f"{input_path}/{system_name}.HSX"
    fdf_file = f"{input_path}/input.fdf"

    FDF = sisl.get_sile(fdf_file)
    HSX = sisl.get_sile(hsx_file)

    geom = FDF.read_geometry()
    nou = geom.no
    nau = geom.na

    # Read the Hamiltonian and overlap matrices
    H = HSX.read_hamiltonian(geometry=geom)
    S = HSX.read_overlap()
    ef = HSX.read_fermi_level()

    nou = H.shape[0]
    nspin = H.spin.size if hasattr(H, "spin") else 1
    atoms = geom.atoms
    orbitals = atoms.orbitals
    Rs = H.sc_off

    # hamiltonians.h5, density_matrixs.h5, overlap.h5
    info = {
        "nsites": nau,
        "isorthogonal": False,
        "isspinful": nspin != 1,
        "norbits": nou,
        "fermi_level": ef,
    }
    with open("{}/info.json".format(output_path), "w") as info_f:
        json.dump(info, info_f)

    # Load orbital index
    orb_indx = np.genfromtxt(
        f"{input_path}/{system_name}.ORB_INDX", skip_header=3, skip_footer=17
    )

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
            if i > nou - 1:
                for j in range(4):
                    for _ in range(int(atom_orb_cnt[j] / (2 * j + 1))):
                        orb_type_f.write(f"{j}  ")
                orb_type_f.write("\n")
                break

    H_block_sparse = {}
    S_block_sparse = {}

    # Hamiltonian pasring
    if nspin == 1:
        cum = np.cumsum(np.concatenate(([0], orbitals)))
        for i, R in enumerate(Rs):
            tmpt = H.tocsr(dim=0).toarray()[:, i * nou : (i + 1) * nou]
            for ia in range(num_atoms):
                for ja in range(num_atoms):
                    H_block = tmpt[cum[ia] : cum[ia + 1], cum[ja] : cum[ja + 1]]

                    if abs(H_block).max() < 1e-8:
                        continue

                    key = f"[{int(R[0])}, {int(R[1])}, {int(R[2])}, {ia + 1}, {ja + 1}]"

                    if key not in H_block_sparse:
                        H_block_sparse[key] = []

                    H_block_sparse[key] = H_block

    elif nspin == 8:
        H_upup = H.tocsr(dim=0) + 1j * H.tocsr(dim=1)
        H_updw = H.tocsr(dim=2) + 1j * H.tocsr(dim=3)
        H_dwup = H.tocsr(dim=4) + 1j * H.tocsr(dim=5)
        H_dwdw = H.tocsr(dim=6) + 1j * H.tocsr(dim=7)

        cum = np.cumsum(np.concatenate(([0], orbitals)))
        for i, R in enumerate(Rs):
            for ia, noi in enumerate(orbitals):
                for ja, noj in enumerate(orbitals):
                    Hij = np.zeros((noi * 2, noj * 2), dtype=np.complex128)

                    Hij[:noi, :noj] = H_upup[
                        cum[ia] : cum[ia + 1], cum[ja] : cum[ja + 1]
                    ].toarray()
                    Hij[:noi, noj:] = H_updw[
                        cum[ia] : cum[ia + 1], cum[ja] : cum[ja + 1]
                    ].toarray()
                    Hij[noi:, :noj] = H_dwup[
                        cum[ia] : cum[ia + 1], cum[ja] : cum[ja + 1]
                    ].toarray()
                    Hij[noi:, noj:] = H_dwdw[
                        cum[ia] : cum[ia + 1], cum[ja] : cum[ja + 1]
                    ].toarray()

                    if np.abs(Hij).max() < 1e-8:
                        continue

                    key = f"[{int(R[0])}, {int(R[1])}, {int(R[2])}, {ia + 1}, {ja + 1}]"

                    if key not in H_block_sparse:
                        H_block_sparse[key] = []

                    H_block_sparse[key] = Hij

    else:
        raise NotImplementedError("Only nspin=1 and nspin=8 are supported.")

    # Overlap parsing
    cum = np.cumsum(np.concatenate(([0], orbitals)))
    for i, R in enumerate(Rs):
        tmpt = S.tocsr(dim=0).toarray()[:, i * nou : (i + 1) * nou]
        for ia in range(num_atoms):
            for ja in range(num_atoms):
                S_block = tmpt[cum[ia] : cum[ia + 1], cum[ja] : cum[ja + 1]]

                if abs(S_block).max() < 1e-8:
                    continue

                key = f"[{int(R[0])}, {int(R[1])}, {int(R[2])}, {ia + 1}, {ja + 1}]"

                if key not in S_block_sparse:
                    S_block_sparse[key] = []

                S_block_sparse[key] = S_block

    # Save Hamiltonian and overlap matrix to HDF5
    with h5py.File(f"{output_path}/hamiltonians.h5", "w") as f:
        for atom_pair, sparse_block in H_block_sparse.items():
            f.create_dataset(atom_pair, data=sparse_block)

    with h5py.File(f"{output_path}/overlaps.h5", "w") as f:
        for atom_pair, sparse_block in S_block_sparse.items():
            f.create_dataset(atom_pair, data=sparse_block)
