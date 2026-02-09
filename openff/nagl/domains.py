import typing

from openff.nagl._base.base import ImmutableModel
from openff.nagl.toolkits import NAGLToolkitRegistry
from openff.nagl.toolkits.openff import validate_toolkit_registry

try:
    from pydantic.v1 import Field
except ImportError:
    from pydantic import Field

if typing.TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

class ChemicalDomain(ImmutableModel):
    """A domain of chemical space to which a molecule can belong

    Used for determining if a molecule is represented in the
    training data for a given model.
    """
    allowed_elements: typing.Tuple[int, ...] = Field(
        description="The atomic numbers of the elements allowed in the domain",
        default_factory=tuple
    )
    forbidden_patterns: typing.Tuple[str, ...] = Field(
        description="The SMARTS patterns which are forbidden in the domain",
        default_factory=tuple
    )

    @validate_toolkit_registry
    def check_molecule(
        self,
        molecule: "Molecule",
        return_error_message: bool = False,
        toolkit_registry: NAGLToolkitRegistry | None = None
    ) -> typing.Union[bool, typing.Tuple[bool, str]]:
        checks = [
            self.check_allowed_elements,
            self.check_forbidden_patterns
        ]
        for check in checks:
            is_allowed, err = check(molecule, return_error_message=True, toolkit_registry=toolkit_registry)
            if not is_allowed:
                if return_error_message:
                    return False, err
                return False
        if return_error_message:
            return True, ""
        return True
        
    @validate_toolkit_registry
    def check_allowed_elements(
        self,
        molecule: "Molecule",
        return_error_message: bool = False,
        toolkit_registry: NAGLToolkitRegistry | None = None
    ) -> typing.Union[bool, typing.Tuple[bool, str]]:
        if not self.allowed_elements:
            return True
        atomic_numbers = [atom.atomic_number for atom in molecule.atoms]
        for atomic_number in atomic_numbers:
            if atomic_number not in self.allowed_elements:
                if return_error_message:
                    err = f"Molecule contains forbidden element {atomic_number}"
                    return False, err
                return False
        if return_error_message:
            return True, ""
        return True

    @validate_toolkit_registry
    def check_forbidden_patterns(
        self,
        molecule: "Molecule",
        return_error_message: bool = False,
        toolkit_registry: NAGLToolkitRegistry | None = None
    ) -> typing.Union[bool, typing.Tuple[bool, str]]:
        for pattern in self.forbidden_patterns:
            if molecule.chemical_environment_matches(pattern, toolkit_registry=toolkit_registry):
                err = f"Molecule contains forbidden SMARTS pattern {pattern}"
                if return_error_message:
                    return False, err
                return False
        if return_error_message:
            return True, ""
        return True