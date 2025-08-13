"""Functions using the OpenEye toolkit"""

import copy
from typing import Tuple, TYPE_CHECKING, List, Union

import numpy as np

from openff.units import unit
from requests import options

from openff.nagl.toolkits._base import NAGLToolkitWrapperBase
from openff.toolkit.utils.openeye_wrapper import OpenEyeToolkitWrapper
from openff.nagl.utils._types import HybridizationType


if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


class NAGLOpenEyeToolkitWrapper(NAGLToolkitWrapperBase, OpenEyeToolkitWrapper):
    name = "openeye"

    def _run_normalization_reactions(
        self,
        molecule: "Molecule",
        normalization_reactions: Tuple[str, ...] = tuple(),
        **kwargs,
    ):
        """
        Normalize the bond orders and charges of a molecule by applying a series of transformations to it.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to normalize
        normalization_reactions: Tuple[str, ...], default=tuple()
            A tuple of SMARTS reaction strings representing the reactions to apply to the molecule.

        Returns
        -------
        normalized_molecule: openff.toolkit.topology.Molecule
            The normalized molecule. This is a new molecule object, not the same as the input molecule.
        """

        from openeye import oechem

        oemol = self.to_openeye(molecule=molecule)

        for reaction_smarts in normalization_reactions:
            reaction = oechem.OEUniMolecularRxn(reaction_smarts)
            reaction.SetValidateKekule(False)
            options = reaction.GetOptions()
            # no idea what this does, this is completely undocumented
            options.SetHydrogenConversions(False)
            outcome = reaction(oemol)

        molecule = self.from_openeye(
            oemol,
            allow_undefined_stereo=True,
            _cls=molecule.__class__,
        )

        return molecule

    def get_molecule_hybridizations(
        self, molecule: "Molecule"
    ) -> List[HybridizationType]:
        """
        Get the hybridization of each atom in a molecule.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to get the hybridizations of.

        Returns
        -------
        hybridizations: List[HybridizationType]
            The hybridization of each atom in the molecule.
        """

        from openeye import oechem

        conversions = {
            oechem.OEHybridization_Unknown: HybridizationType.OTHER,
            oechem.OEHybridization_sp: HybridizationType.SP,
            oechem.OEHybridization_sp2: HybridizationType.SP2,
            oechem.OEHybridization_sp3: HybridizationType.SP3,
            oechem.OEHybridization_sp3d: HybridizationType.SP3D,
            oechem.OEHybridization_sp3d2: HybridizationType.SP3D2,
        }

        hybridizations = []
        oemol = self.to_openeye(molecule=molecule)
        oechem.OEAssignHybridization(oemol)

        for atom in oemol.GetAtoms():
            hybridization = atom.GetHyb()
            try:
                hybridizations.append(conversions[hybridization])
            except KeyError:
                raise ValueError(f"Unknown hybridization {hybridization}")
        return hybridizations

    def _molecule_from_openeye(
        self,
        oemol,
        as_smiles: bool = False,
        mapped_smiles: bool = False,
    ):
        """
        Create a Molecule from an OpenEye OEMol with charges

        Parameters
        ----------
        oemol: openeye.oechem.OEMol
            The molecule to create a Molecule from
        as_smiles: bool, default=False
            If True, return a SMILES string instead of an OpenFF Molecule
        mapped_smiles: bool, default=False
            If True, return a SMILES string with atom indices as atom map numbers.

        Returns
        -------
        molecule: openff.toolkit.topology.Molecule or str
        """

        from openff.toolkit.topology import Molecule

        has_charges = (
            OpenEyeToolkitWrapper._turn_oemolbase_sd_charges_into_partial_charges(oemol)
        )
        offmol = self.from_openeye(
            oemol,
            allow_undefined_stereo=True,
            _cls=Molecule,
        )
        if not has_charges:
            offmol.partial_charges = None
        if as_smiles:
            return offmol.to_smiles(mapped=mapped_smiles)
        return offmol

    def stream_molecules_from_sdf_file(
        self,
        file: str,
        as_smiles: bool = False,
        mapped_smiles: bool = False,
        include_sdf_data: bool = True,
        **kwargs,
    ):
        """
        Stream molecules from an SDF file.

        Parameters
        ----------
        file: str
            The path to the SDF file to stream molecules from.
        as_smiles: bool, default=False
            If True, return a SMILES string instead of an OpenFF Molecule
        mapped_smiles: bool, default=False
            If True, return a SMILES string with atom indices as atom map numbers.
        include_sdf_data: bool, default=True
            If True, include the SDF tag data in the output molecules.

        Returns
        -------
        molecules: Generator[openff.toolkit.topology.Molecule or str]

        """
        from openeye import oechem

        stream = oechem.oemolistream()
        stream.open(file)
        is_sdf = stream.GetFormat() == oechem.OEFormat_SDF
        for oemol in stream.GetOEMols():
            if is_sdf and hasattr(oemol, "GetConfIter"):
                for conf in oemol.GetConfIter():
                    confmol = conf.GetMCMol()

                    if include_sdf_data and not as_smiles:
                        for dp in oechem.OEGetSDDataPairs(oemol):
                            oechem.OESetSDData(confmol, dp.GetTag(), dp.GetValue())
                        for dp in oechem.OEGetSDDataPairs(conf):
                            oechem.OESetSDData(confmol, dp.GetTag(), dp.GetValue())
            else:
                confmol = oemol
            yield self._molecule_from_openeye(
                confmol, as_smiles=as_smiles, mapped_smiles=mapped_smiles
            )

    def to_openeye(self, molecule: "Molecule"):
        """
        Convert an OpenFF Molecule to an OpenEye OEMol with charges
        stored as SD data.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to convert

        Returns
        -------
        oemol: openeye.oechem.OEMol
        """
        from openeye import oechem

        oemol = super().to_openeye(molecule)

        if molecule.partial_charges is not None:
            partial_charges_list = [
                oeatom.GetPartialCharge() for oeatom in oemol.GetAtoms()
            ]
            partial_charges_str = " ".join([f"{val:f}" for val in partial_charges_list])
            oechem.OESetSDData(oemol, "atom.dprop.PartialCharge", partial_charges_str)
        return oemol

    def stream_molecules_to_file(self, file: str):
        """
        Stream molecules to an SDF file using a context manager.

        Parameters
        ----------
        file: str
            The path to the SDF file to stream molecules to.


        Examples
        --------

        >>> from openff.toolkit.topology import Molecule
        >>> from openff.toolkit.utils.toolkits import OpenEyeToolkitWrapper
        >>> toolkit_wrapper = OpenEyeToolkitWrapper()
        >>> molecule1 = Molecule.from_smiles("CCO")
        >>> molecule2 = Molecule.from_smiles("CCC")
        >>> with toolkit_wrapper.stream_molecules_to_file("molecules.sdf") as writer:
        ...     writer(molecule1)
        ...     writer(molecule2)

        """
        from openeye import oechem
        from openff.toolkit.topology import Molecule

        stream = oechem.oemolostream(file)

        def writer(molecule: Molecule):
            oechem.OEWriteMolecule(stream, self.to_openeye(molecule))

        yield writer

        stream.close()

    def get_best_rmsd(
        self,
        molecule: "Molecule",
        reference_conformer: Union[np.ndarray, unit.Quantity],
        target_conformer: Union[np.ndarray, unit.Quantity],
    ) -> unit.Quantity:
        """
        Compute the lowest all-atom RMSD between a reference and target conformer,
        allowing for symmetry-equivalent atoms to be permuted.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to compute the RMSD for
        reference_conformer: np.ndarray or openff.units.unit.Quantity
            The reference conformer to compare to the target conformer.
            If a numpy array, it is assumed to be in units of angstrom.
        target_conformer: np.ndarray or openff.units.unit.Quantity
            The target conformer to compare to the reference conformer.
            If a numpy array, it is assumed to be in units of angstrom.

        Returns
        -------
        rmsd: unit.Quantity

        Examples
        --------
        >>> from openff.units import unit
        >>> from openff.toolkit.topology import Molecule
        >>> from openff.toolkit.utils.toolkits import OpenEyeToolkitWrapper
        >>> toolkit_wrapper = OpenEyeToolkitWrapper()
        >>> molecule = Molecule.from_smiles("CCCCO")
        >>> molecule.generate_conformers(n_conformers=2)
        >>> rmsd = toolkit_wrapper.get_best_rmsd(molecule, molecule.conformers[0], molecule.conformers[1])
        >>> print(f"RMSD in angstrom: {rmsd.m_as(unit.angstrom)}")

        """
        from openeye import oechem

        if not isinstance(reference_conformer, unit.Quantity):
            reference_conformer = reference_conformer * unit.angstrom
        if not isinstance(target_conformer, unit.Quantity):
            target_conformer = target_conformer * unit.angstrom

        mol1 = copy.deepcopy(molecule)
        mol1._conformers = [reference_conformer]
        mol2 = copy.deepcopy(molecule)
        mol2._conformers = [target_conformer]

        oemol1 = self.to_openeye(mol1)
        oemol2 = self.to_openeye(mol2)

        # OERMSD(OEMolBase ref, OEMolBase fit, bool automorph=True, bool heavyOnly=True, bool overlay=False, double * rot=None, double * trans=None) -> double
        rmsd = oechem.OERMSD(oemol1, oemol2, True, False, True)
        return rmsd * unit.angstrom

    def get_atoms_are_in_ring_size(
        self,
        molecule: "Molecule",
        ring_size: int,
    ) -> List[bool]:
        """
        Determine whether each atom in a molecule is in a ring of a given size.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to compute ring perception for
        ring_size: int
            The size of the ring to check for.

        Returns
        -------
        in_ring_size: List[bool]

        """
        from openeye import oechem

        oemol = self.to_openeye(molecule)
        oechem.OEFindRingAtomsAndBonds(oemol)

        in_ring_size = [
            oechem.OEAtomIsInRingSize(atom, ring_size) for atom in oemol.GetAtoms()
        ]
        return in_ring_size

    def get_bonds_are_in_ring_size(
        self,
        molecule: "Molecule",
        ring_size: int,
    ) -> List[bool]:
        """
        Determine whether each bond in a molecule is in a ring of a given size.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to compute ring perception for
        ring_size: int
            The size of the ring to check for.

        Returns
        -------
        in_ring_size: List[bool]
            Bonds are in the same order as the molecule's ``bonds`` attribute.
        """
        from openeye import oechem

        oemol = self.to_openeye(molecule)
        oechem.OEFindRingAtomsAndBonds(oemol)

        is_in_ring_size = [None] * len(molecule.bonds)
        for oebond in oemol.GetBonds():
            oe_i = oebond.GetBgnIdx()
            oe_j = oebond.GetEndIdx()
            off_bond = molecule.get_bond_between(oe_i, oe_j)
            bond_index = off_bond.molecule_bond_index
            is_in_ring_size[bond_index] = oechem.OEBondIsInRingSize(oebond, ring_size)

        return is_in_ring_size

    # TODO: this only outputs 0 or 1.
    # def calculate_circular_fingerprint_similarity(
    #     self,
    #     molecule: "Molecule",
    #     reference_molecule: "Molecule",
    #     radius: int = 3,
    #     nbits: int = 2048,
    # ) -> float:
    #     """
    #     Compute the similarity between two molecules using a fingerprinting method.
    #     Uses a Morgan fingerprint with RDKit and a Circular fingerprint with OpenEye.

    #     Parameters
    #     ----------
    #     molecule: openff.toolkit.topology.Molecule
    #         The molecule to compute the fingerprint for.
    #     reference_molecule: openff.toolkit.topology.Molecule
    #         The molecule to compute the fingerprint for.
    #     radius: int, default 3
    #         The radius of the fingerprint to use.
    #     nbits: int, default 2048
    #         The length of the fingerprint to use. Not used in RDKit.

    #     Returns
    #     -------
    #     similarity: float
    #         The Dice similarity between the two molecules.

    #     """
    #     from openeye import oegraphsim

    #     oegraphsim.OEFPBondType_DefaultCircularBond

    #     # Connectivity: (Element, #heavy neighbors, #Hs, charge, isotope, inRing
    #     # Donor, Acceptor, Aromatic, Halogen, Basic, Acidic
    #     atypes = (
    #         oegraphsim.OEFPAtomType_AtomicNumber
    #         | oegraphsim.OEFPAtomType_HvyDegree
    #         | oegraphsim.OEFPAtomType_HCount
    #         | oegraphsim.OEFPAtomType_FormalCharge
    #         | oegraphsim.OEFPAtomType_InRing
    #         | oegraphsim.OEFPAtomType_Chiral
    #         | oegraphsim.OEFPAtomType_EqHBondDonor
    #         | oegraphsim.OEFPAtomType_EqHBondAcceptor
    #         | oegraphsim.OEFPAtomType_EqAromatic
    #         | oegraphsim.OEFPAtomType_EqHalogen
    #     )

    #     btypes = oegraphsim.OEFPBondType_BondOrder

    #     oemol1 = self.to_openeye(molecule)
    #     oemol2 = self.to_openeye(reference_molecule)

    #     fp1 = oegraphsim.OEFingerPrint()
    #     oegraphsim.OEMakeCircularFP(fp1, oemol1, nbits, radius, radius, atypes, btypes)
    #     fp2 = oegraphsim.OEFingerPrint()
    #     oegraphsim.OEMakeCircularFP(fp2, oemol2, nbits, radius, radius, atypes, btypes)

    #     similarity = oegraphsim.OEDice(fp1, fp2)

    #     return similarity


# import contextlib


# from openff.utilities import requires_package


# @contextlib.contextmanager
# @requires_package("openeye.oechem")
# def capture_oechem_warnings():  # pragma: no cover
#     from openeye import oechem

#     output_stream = oechem.oeosstream()
#     oechem.OEThrow.SetOutputStream(output_stream)
#     oechem.OEThrow.Clear()

#     yield

#     oechem.OEThrow.SetOutputStream(oechem.oeerr)


# @requires_package("openeye.oechem")
# def _normalize_molecule_oe(
#     molecule: "Molecule", reaction_smarts: List[str]
# ) -> "Molecule":  # pragma: no cover

#     from openeye import oechem
#     from openff.toolkit.topology import Molecule

#     oe_molecule: oechem.OEMol = molecule.to_openeye()

#     for pattern in reaction_smarts:

#         reaction = oechem.OEUniMolecularRxn(pattern)
#         reaction(oe_molecule)

#     return Molecule.from_openeye(oe_molecule, allow_undefined_stereo=True)
