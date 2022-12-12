from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openff.units import unit

__all__ = [
    "calculate_esp_in_angstrom",
    "calculate_esp_with_units",
    "calculate_dipole_in_angstrom",
    "calculate_dipole_with_units",
]


def _calculate_esp(
    grid_coordinates,
    atom_coordinates,
    charges,
):
    displacement = (
        grid_coordinates[:, None, :] - atom_coordinates[None, :, :]
    )  # N x M x 3
    distance = (displacement**2).sum(axis=-1) ** 0.5  # N x M
    inv_distance = 1 / distance
    esp = inv_distance @ charges
    return esp


def calculate_esp_in_angstrom(grid_coordinates, atom_coordinates, charges):
    BOHR_TO_ANGSTROM = 0.5291772109039775
    esp = _calculate_esp(grid_coordinates, atom_coordinates, charges)
    return BOHR_TO_ANGSTROM * esp


def calculate_esp_with_units(
    grid_coordinates: "unit.Quantity",  # N x 3
    atom_coordinates: "unit.Quantity",  # M x 3
    charges: "unit.Quantity",  # M
    with_units: bool = False,
) -> "unit.Quantity":
    AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge
    ke = 1 / (4 * np.pi * unit.epsilon_0)
    esp = ke * _calculate_esp(grid_coordinates, atom_coordinates, charges)

    esp_q = esp.m_as(AU_ESP)
    if not with_units:
        return esp_q
    return esp


def _calculate_dipole(atom_coordinates, charges):
    charges = charges.reshape((-1, 1))
    dipole = (atom_coordinates * charges).sum(axis=0)
    return dipole


def calculate_dipole_in_angstrom(atom_coordinates, charges):
    eA_to_debye = 0.20819433270935603
    dipole = _calculate_dipole(atom_coordinates, charges)
    return eA_to_debye * dipole


def calculate_dipole_with_units(
    atom_coordinates: "unit.Quantity",
    charges: "unit.Quantity",
    with_units: bool = False,
) -> np.ndarray:
    dipole = _calculate_dipole(atom_coordinates, charges)
    dipole_q = dipole.m_as(unit.debye)
    if not with_units:
        return dipole_q
    return dipole
