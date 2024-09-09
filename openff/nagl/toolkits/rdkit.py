"""Functions using the RDKit toolkit"""


import copy
import functools
from typing import Tuple, TYPE_CHECKING, List, Union

import numpy as np

from openff.units import unit


from openff.nagl.toolkits._base import NAGLToolkitWrapperBase
from openff.toolkit.utils.rdkit_wrapper import RDKitToolkitWrapper
from openff.nagl.utils._types import HybridizationType

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


class NAGLRDKitToolkitWrapper(NAGLToolkitWrapperBase, RDKitToolkitWrapper):
    name = "rdkit"

    def _run_normalization_reactions(
        self,
        molecule,
        normalization_reactions: Tuple[str, ...] = tuple(),
        max_iter: int = 200,
    ):
        """
        Normalize the bond orders and charges of a molecule by applying a series of transformations to it.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to normalize
        normalization_reactions: Tuple[str, ...], default=tuple()
            A tuple of SMARTS reaction strings representing the reactions to apply to the molecule.
        max_iter: int, default=200
            The maximum number of iterations to perform for each transformation.

        Returns
        -------
        normalized_molecule: openff.toolkit.topology.Molecule
            The normalized molecule. This is a new molecule object, not the same as the input molecule.
        """

        from rdkit import Chem
        from rdkit.Chem import rdChemReactions

        rdmol = self.to_rdkit(molecule=molecule)
        original_smiles = new_smiles = Chem.MolToSmiles(rdmol)

        # track atoms
        for i, atom in enumerate(rdmol.GetAtoms(), 1):
            atom.SetAtomMapNum(i)

        for reaction_smarts in normalization_reactions:
            reaction = rdChemReactions.ReactionFromSmarts(reaction_smarts)
            n_iter = 0
            keep_changing = True

            while n_iter < max_iter and keep_changing:
                n_iter += 1
                old_smiles = new_smiles

                products = reaction.RunReactants((rdmol,), maxProducts=1)
                if not products:
                    break

                try:
                    ((rdmol,),) = products
                except ValueError:
                    raise ValueError(
                        f"Reaction produced multiple products: {reaction_smarts}"
                    )

                for atom in rdmol.GetAtoms():
                    atom.SetAtomMapNum(atom.GetIntProp("react_atom_idx") + 1)

                new_smiles = Chem.MolToSmiles(Chem.AddHs(rdmol))
                # stop changing when smiles converges to same product
                keep_changing = new_smiles != old_smiles

            if n_iter >= max_iter and keep_changing:
                raise ValueError(
                    f"Reaction {reaction_smarts} did not converge after "
                    f"{max_iter} iterations for molecule {original_smiles}"
                )

        new_mol = self.from_rdkit(
            rdmol,
            allow_undefined_stereo=True,
            _cls=molecule.__class__,
        )
        mapping = new_mol.properties.pop("atom_map")
        adjusted_mapping = dict((current, new - 1) for current, new in mapping.items())

        return new_mol.remap(adjusted_mapping, current_to_new=True)

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

        from rdkit.Chem import rdchem

        conversions = {
            rdchem.HybridizationType.S: HybridizationType.OTHER,
            rdchem.HybridizationType.SP: HybridizationType.SP,
            rdchem.HybridizationType.SP2: HybridizationType.SP2,
            rdchem.HybridizationType.SP3: HybridizationType.SP3,
            rdchem.HybridizationType.SP3D: HybridizationType.SP3D,
            rdchem.HybridizationType.SP3D2: HybridizationType.SP3D2,
            rdchem.HybridizationType.OTHER: HybridizationType.OTHER,
            rdchem.HybridizationType.UNSPECIFIED: HybridizationType.OTHER,
        }

        hybridizations = []
        rdmol = self.to_rdkit(molecule)
        for atom in rdmol.GetAtoms():
            hybridization = atom.GetHybridization()
            try:
                hybridizations.append(conversions[hybridization])
            except KeyError:
                raise ValueError(f"Unknown hybridization {hybridization}")
        return hybridizations

    def to_rdkit(self, molecule: "Molecule"):
        try:
            return super().to_rdkit(molecule)
        except AssertionError:
            # OpenEye just accepts all stereochemistry
            # unlike RDKit which e.g. does not allow stereogenic bonds in a ring < 8
            # try patching via smiles
            # smiles = "C1CC/C=C/(CC1)Cl"

            mapped_smiles = molecule.to_smiles(mapped=True)
            mol2 = Molecule.from_mapped_smiles(
                mapped_smiles,
                allow_undefined_stereo=True,
                toolkit_registry=RDKitToolkitWrapper(),
            )
            mol2_bonds = {
                (bd.atom1_index, bd.atom2_index): bd.stereochemistry
                for bd in mol2.bonds
            }

            molecule = copy.deepcopy(molecule)
            for bond in molecule.bonds:
                bond._stereochemistry = mol2_bonds[bond.atom1_index, bond.atom2_index]

            return molecule.to_rdkit()

    @classmethod
    def smiles_to_mapped_smiles(cls, smiles: str) -> str:
        """Convert a SMILES string to a mapped SMILES string.

        Parameters
        ----------
        smiles: str
            The SMILES string to convert.

        Returns
        -------
        mapped_smiles: str
            The mapped SMILES string.
        """
        molecule = cls.from_smiles(smiles, allow_undefined_stereo=True)
        return cls.to_smiles(molecule, mapped=True)

    @classmethod
    def mapped_smiles_to_smiles(cls, mapped_smiles: str) -> str:
        """Convert a mapped SMILES string to a SMILES string.

        Parameters
        ----------
        mapped_smiles: str
            The mapped SMILES string to convert.

        Returns
        -------
        smiles: str
            The SMILES string.
        """
        molecule = cls.from_smiles(mapped_smiles, allow_undefined_stereo=True)
        return cls.to_smiles(molecule)

    @classmethod
    def stream_molecules_from_sdf_file(
        cls,
        file: str,
        as_smiles: bool = False,
        mapped_smiles: bool = False,
        explicit_hydrogens: bool = True,
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
        explicit_hydrogens: bool, default=True
            If True, keep explicit hydrogens in SMILES outputs.

        Returns
        -------
        molecules: Generator[openff.toolkit.topology.Molecule or str]

        """
        from rdkit import Chem

        wrapper = cls()

        if as_smiles:
            if mapped_smiles:
                converter = cls.smiles_to_mapped_smiles
            else:
                converter = functools.partial(
                    Chem.MolToSmiles, allHsExplicit=explicit_hydrogens
                )
        else:
            converter = functools.partial(
                wrapper.from_rdkit,
                allow_undefined_stereo=True,
                _cls=Molecule
            )

        if file.endswith(".gz"):
            file = file[:-3]

        for rdmol in Chem.SupplierFromFilename(
            file, removeHs=False, sanitize=True, strictParsing=True
        ):
            if rdmol is not None:
                yield converter(rdmol)

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
        >>> from openff.toolkit.utils.toolkits import RDKitToolkitWrapper
        >>> toolkit_wrapper = RDKitToolkitWrapper()
        >>> molecule1 = Molecule.from_smiles("CCO")
        >>> molecule2 = Molecule.from_smiles("CCC")
        >>> with toolkit_wrapper.stream_molecules_to_file("molecules.sdf") as writer:
        ...     writer(molecule1)
        ...     writer(molecule2)

        """
        from rdkit import Chem

        stream = Chem.SDWriter(file)

        def writer(molecule):
            rdmol = self.to_rdkit(molecule)
            n_conf = rdmol.GetNumConformers()
            if not n_conf:
                stream.write(rdmol)
            else:
                for i in range(n_conf):
                    stream.write(rdmol, confId=i)
            stream.flush()

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
        >>> from openff.toolkit.utils.toolkits import RDKitToolkitWrapper
        >>> toolkit_wrapper = RDKitToolkitWrapper()
        >>> molecule = Molecule.from_smiles("CCCCO")
        >>> molecule.generate_conformers(n_conformers=2)
        >>> rmsd = toolkit_wrapper.get_best_rmsd(molecule, molecule.conformers[0], molecule.conformers[1])
        >>> print(f"RMSD in angstrom: {rmsd.m_as(unit.angstrom)}")

        """

        from rdkit.Chem import rdMolAlign

        if not isinstance(reference_conformer, unit.Quantity):
            reference_conformer = reference_conformer * unit.angstrom
        if not isinstance(target_conformer, unit.Quantity):
            target_conformer = target_conformer * unit.angstrom

        mol1 = copy.deepcopy(molecule)
        mol1._conformers = [reference_conformer]
        mol2 = copy.deepcopy(molecule)
        mol2._conformers = [target_conformer]

        rdmol1 = self.to_rdkit(mol1)
        rdmol2 = self.to_rdkit(mol2)

        rmsd = rdMolAlign.GetBestRMS(rdmol1, rdmol2)
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
        rdmol = self.to_rdkit(molecule)
        in_ring_size = [atom.IsInRingSize(ring_size) for atom in rdmol.GetAtoms()]
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
        rdmol = self.to_rdkit(molecule)
        in_ring_size = []
        for bond in molecule.bonds:
            rdbond = rdmol.GetBondBetweenAtoms(bond.atom1_index, bond.atom2_index)
            in_ring_size.append(rdbond.IsInRingSize(ring_size))
        return in_ring_size

    def assign_partial_charges(
        self,
        molecule,
        partial_charge_method=None,
        use_conformers=None,
        strict_n_conformers=False,
        normalize_partial_charges=True,
        _cls=None,
    ):
        """
        Compute partial charges with RDKit, and assign
        the new values to the partial_charges attribute.

        .. warning :: This API is experimental and subject to change.

        Parameters
        ----------

        molecule : openff.toolkit.topology.Molecule
            Molecule for which partial charges are to be computed
        partial_charge_method : str, optional, default=None
            The charge model to use. One of ['mmff94', 'gasteiger']. If None, 'mmff94' will be used.

            * 'mmff94': Applies partial charges using the Merck Molecular Force Field
                        (MMFF). This method does not make use of conformers, and hence
                        ``use_conformers`` and ``strict_n_conformers`` will not impact
                        the partial charges produced.
        use_conformers : iterable of unit-wrapped numpy arrays, each with
            shape (n_atoms, 3) and dimension of distance. Optional, default = None
            Coordinates to use for partial charge calculation. If None, an appropriate number of
            conformers will be generated.
        strict_n_conformers : bool, default=False
            Whether to raise an exception if an invalid number of conformers is provided for
            the given charge method.
            If this is False and an invalid number of conformers is found, a warning will be raised.
        normalize_partial_charges : bool, default=True
            Whether to offset partial charges so that they sum to the total formal charge of the molecule.
            This is used to prevent accumulation of rounding errors when the partial charge generation method has
            low precision.
        _cls : class
            Molecule constructor

        Raises
        ------

        ChargeMethodUnavailableError
            if the requested charge method can not be handled by this toolkit
        ChargeCalculationError
            if the charge method is supported by this toolkit, but fails

        """

        # TODO: Remove when superseded by toolkit update

        import numpy as np
        from rdkit.Chem import AllChem

        if not partial_charge_method == "gasteiger":
            return super().assign_partial_charges(
                molecule,
                partial_charge_method=partial_charge_method,
                use_conformers=use_conformers,
                strict_n_conformers=strict_n_conformers,
                normalize_partial_charges=normalize_partial_charges,
                _cls=_cls,
            )

        rdkit_molecule = self.to_rdkit(molecule)
        AllChem.ComputeGasteigerCharges(rdkit_molecule)
        charges = [
            float(rdatom.GetProp("_GasteigerCharge"))
            for rdatom in rdkit_molecule.GetAtoms()
        ]

        charges = np.asarray(charges)
        molecule.partial_charges = unit.Quantity(charges, unit.elementary_charge)

        if normalize_partial_charges:
            molecule._normalize_partial_charges()

    def calculate_circular_fingerprint_similarity(
        self,
        molecule: "Molecule",
        reference_molecule: "Molecule",
        radius: int = 3,
        nbits: int = 2048,
    ) -> float:
        """
        Compute the similarity between two molecules using a fingerprinting method.
        Uses a Morgan fingerprint with RDKit and a Circular fingerprint with OpenEye.

        Parameters
        ----------
        molecule: openff.toolkit.topology.Molecule
            The molecule to compute the fingerprint for.
        reference_molecule: openff.toolkit.topology.Molecule
            The molecule to compute the fingerprint for.
        radius: int, default 3
            The radius of the fingerprint to use.
        nbits: int, default 2048
            The length of the fingerprint to use. Not used in RDKit.

        Returns
        -------
        similarity: float
            The Dice similarity between the two molecules.

        """
        from rdkit import Chem
        from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
        from rdkit.DataStructs import DiceSimilarity

        rdmol1 = self.to_rdkit(molecule)
        rdmol2 = self.to_rdkit(reference_molecule)

        fp1 = GetMorganFingerprint(rdmol1, radius)
        fp2 = GetMorganFingerprint(rdmol2, radius)

        return DiceSimilarity(fp1, fp2)
