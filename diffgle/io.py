"""Trajectory output (LAMMPS dump format, readable by OVITO/VMD)."""

atom_types_map = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
]


def write_xyz_dump(filename, atom_types, trajectories, box, mode="w"):
    """atom_types: per-particle index into atom_types_map, used only as a label for visualisation
    (CG water / CO2 beads are written as 7 -> 'O', confined-water walls as 5 -> 'C')."""
    box = box.diag()
    with open(filename, mode) as file:
        for timestep, positions in enumerate(trajectories):
            file.write("ITEM: TIMESTEP\n")
            file.write(f"{timestep}\n")
            file.write("ITEM: NUMBER OF ATOMS\n")
            file.write(f"{len(positions)}\n")
            file.write("ITEM: BOX BOUNDS pp pp pp\n")
            file.write(f"0.0000000000000000e+00 {box[0]:.16e}\n")
            file.write(f"0.0000000000000000e+00 {box[1]:.16e}\n")
            file.write(f"0.0000000000000000e+00 {box[2]:.16e}\n")
            file.write("ITEM: ATOMS id element x y z\n")
            for atom_id, (atom_idx, position) in enumerate(zip(atom_types, positions), start=1):
                x, y, z = position
                file.write(f"{atom_id} {atom_types_map[atom_idx]} {x} {y} {z}\n")
