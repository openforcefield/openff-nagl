from typing import Generator, List, Optional, Tuple

from rdkit import Chem


class PathGenerator:
    def __init__(self, rdkit_molecule: Chem.Mol):
        self.rdkit_molecule = Chem.Mol(rdkit_molecule)
        self.n_atoms = self.rdkit_molecule.GetNumAtoms()
        self.cache = {}

    def get_bonded_atoms(self, ix: int) -> Generator[int, None, None]:
        atom = self.rdkit_molecule.GetAtomWithIdx(ix)
        for bond in atom.GetBonds():
            yield bond.GetOtherAtomIdx(ix)

    def all_simple_paths(
        self,
        source: int,
        target: int,
        cutoff: Optional[int] = None,
    ) -> Generator[List[int], None, None]:

        if cutoff is None:
            cutoff = self.n_atoms - 1

        visited = [source]
        stack: List[Generator[int]] = [self.get_bonded_atoms(source)]
        while stack:
            children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                visited.pop()

            elif len(visited) < cutoff:
                if child == target:
                    yield visited + [target]

                elif child not in visited:
                    visited.append(child)
                    stack.append(self.get_bonded_atoms(child))

            else:
                if child == target or target in children:
                    yield visited + [target]
                stack.pop()
                visited.pop()

    def all_odd_node_simple_paths(
        self,
        source: int,
        target: int,
        cutoff: Optional[int] = None,
    ) -> Tuple[Tuple[int, ...]]:

        key = (source, target)
        if key in self.cache:
            return self.cache[key]

        paths = tuple(
            tuple(path)
            for path in self.all_simple_paths(source, target, cutoff)
            if len(path) % 2
        )
        self.cache[key] = paths
        self.cache[key[::-1]] = tuple(x[::-1] for x in paths)
        return paths
