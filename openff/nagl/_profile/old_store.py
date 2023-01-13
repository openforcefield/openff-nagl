import logging
import enum
import os
import functools
import pathlib
import tqdm
from collections import defaultdict, Counter
from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, List, Tuple, ContextManager, Literal, Optional
import warnings

from openff.nagl.utils.types import Pathlike


from sqlalchemy import (  # Enum,
    Column,
    ForeignKey,
    Integer,
    PickleType,
    String,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session, sessionmaker
from sqlalchemy import create_engine

from openff.nagl.storage.record import Record
from openff.nagl.profile.environments import ChemicalEnvironment
from openff.nagl.utils.openff import capture_toolkit_warnings

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

DBBase = declarative_base()

@functools.lru_cache(maxsize=1000)
def unmap_smiles(smiles: str) -> str:
    from openff.toolkit.topology import Molecule
    from openff.nagl.utils.openff import capture_toolkit_warnings

    with capture_toolkit_warnings():
        offmol = Molecule.from_smiles(
            smiles,
            allow_undefined_stereo=True
        )
        return offmol.to_smiles(mapped=False)



class vdWTypeRecord(Record):
    forcefield: str
    type_counts: Dict[str, int]


class MoleculeInfoRecord(Record):
    smiles: str
    chemical_environment_counts: Dict[ChemicalEnvironment, int] = {}
    element_counts: Dict[int, int] = {}
    vdw_type_counts: Dict[str, vdWTypeRecord] = {}

    @classmethod
    def retrieve_or_create(
        cls,
        smiles: str,
        store: Optional["MoleculeInfoStore"] = None,
        forcefield_files: Tuple[str, ...] = tuple(),
    ):
        if store is not None:
            unmapped = unmap_smiles(smiles)
            records = store.retrieve(smiles=[unmapped])
            if len(records):
                record = records[0]
                for ff_file in forcefield_files:
                    try:
                        record.count_vdw_types(ff_file)
                    except ValueError:
                        pass
                return record
        return cls.from_smiles(smiles, forcefield_files)

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        forcefield_files: Tuple[str, ...] = tuple(),
    ):
        from openff.toolkit.topology import Molecule
        from openff.nagl.profile.environments import analyze_functional_groups

        with capture_toolkit_warnings():
            offmol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        zs = Counter([atom.atomic_number for atom in offmol.atoms])
        element_counts = {k: zs[k] for k in sorted(zs)}

        vdw_type_counts = {}
        for ff_file in forcefield_files:
            ff_name = pathlib.Path(ff_file).stem
            counts = cls._count_vdw_types(offmol, ff_file)
            vdw_type_counts[ff_name] = vdWTypeRecord(
                forcefield=ff_name,
                type_counts=counts
            )

        chemical_environment_counts = analyze_functional_groups(offmol)
        obj = cls(
            smiles=smiles,
            element_counts=element_counts,
            vdw_type_counts=vdw_type_counts,
            chemical_environment_counts=chemical_environment_counts
        )
        return obj


    @staticmethod
    def _count_vdw_types(
        molecule: "Molecule",
        forcefield_file: str,
    ) -> Dict[str, int]:
        from openff.toolkit.typing.engines.smirnoff import ForceField
        
        ff = ForceField(forcefield_file)
        labels = ff.label_molecules(molecule.to_topology())[0]["vdW"]
        parameter_counts = Counter(labels.values())
        counts = {k.id: v for k, v in parameter_counts.items()}
        sorted_counts = {k: counts[k] for k in sorted(counts)}
        return sorted_counts


    def count_vdw_types(
        self,
        forcefield_file: str,
        forcefield_name: Optional[str] = None
    ):
        from openff.toolkit.typing.engines.smirnoff import ForceField
        from openff.toolkit.topology import Molecule

        if forcefield_name is None:
            forcefield_name = pathlib.Path(forcefield_file).stem
        
        if forcefield_name in self.vdw_type_counts:
            raise ValueError(f"{forcefield_name} already has vdW type counts")

        with capture_toolkit_warnings():
            offmol = Molecule.from_smiles(
                self.smiles,
                allow_undefined_stereo=True
            )
        counts = _count_vdw_types(offmol, forcefield_file)
        self.vdw_type_counts[forcefield_name] = counts
        return self.vdw_type_counts[forcefield_name]
    


class DBvdWTypeRecord(DBBase):
    __tablename__ = "lennard_jones_types"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    forcefield = Column(String(20), nullable=False)
    type_counts = Column(PickleType, nullable=False)

    __table_args__ = (
        UniqueConstraint("parent_id", "forcefield", name="_vdw_parent_forcefield_uc"),
    )


class DBMoleculeInfoRecord(DBBase):
    __tablename__ = "molecules"

    id = Column(Integer, primary_key=True, index=True)
    smiles = Column(String, nullable=False)

    chemical_environment_counts = Column(PickleType, nullable=False)
    element_counts = Column(PickleType, nullable=False)
    vdw_type_counts = relationship("DBvdWTypeRecord", cascade="all, delete-orphan")

    def store_vdw_type_record(
        self,
        vdw_type_record: vdWTypeRecord
    ):
        existing_ffs = [record.forcefield for record in self.vdw_type_counts]
        if vdw_type_record.forcefield in existing_ffs:
            warnings.warn(
                f"{vdw_type_record.forcefield} force field already stored "
                f"for {self.smiles}. Skipping."
            )
            return
        vdw_data = DBvdWTypeRecord(
            forcefield=vdw_type_record.forcefield,
            type_counts=vdw_type_record.type_counts
        )
        self.vdw_type_counts.append(vdw_data)

class MoleculeInfoStore:

    def __init__(self, database_path: Pathlike = "molecule-store.sqlite"):

        database_path = pathlib.Path(database_path)
        if not database_path.suffix.lower() == ".sqlite":
            raise NotImplementedError(
                "Only paths to SQLite databases ending in .sqlite "
                f"are supported. Given: {database_path}"
            )

        self.database_url = f"sqlite:///{database_path.resolve()}"
        self.engine = create_engine(self.database_url)
        DBBase.metadata.create_all(self.engine)

        self._sessionmaker = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    @contextmanager
    def _get_session(self) -> ContextManager[Session]:
        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except BaseException as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_smiles(self) -> List[str]:
        with self._get_session() as db:
            return [
                smiles
                for (smiles,) in db.query(DBMoleculeInfoRecord.smiles).distinct()
            ]

    def store(self, records: Tuple[MoleculeInfoRecord, ...] = tuple()):
        if isinstance(records, MoleculeInfoRecord):
            records = [records]
        
        with self._get_session() as db:
        
            # for record in tqdm.tqdm(records, desc="Storing MoleculeInfoRecords"):
            for record in records:
                smiles = unmap_smiles(record.smiles)
                existing = (
                    db.query(DBMoleculeInfoRecord)
                    .filter(DBMoleculeInfoRecord.smiles == smiles)
                    .all()
                )
                if not len(existing):
                    existing_record = DBMoleculeInfoRecord(
                        smiles=smiles,
                        chemical_environment_counts=record.chemical_environment_counts,
                        element_counts=record.element_counts,
                    )
                    db.add(existing_record)
                else:
                    existing_record = existing[0]
                
                for vdw_record in record.vdw_type_counts.values():
                    existing_record.store_vdw_type_record(vdw_record)


    def retrieve(
        self,
        smiles: Optional[List[str]] = None,
        forcefields: Optional[List[str]] = None,
    ):
        if smiles is not None and not smiles:
            return []
        if forcefields is not None and not forcefields:
            return []

        with self._get_session() as db:
            results = (
                db.query(
                    DBMoleculeInfoRecord.id,
                    DBMoleculeInfoRecord.smiles,
                    DBMoleculeInfoRecord.chemical_environment_counts,
                    DBMoleculeInfoRecord.element_counts,
                    DBvdWTypeRecord.id,
                    DBvdWTypeRecord.forcefield,
                    DBvdWTypeRecord.type_counts,
                )
                .order_by(DBMoleculeInfoRecord.id)
                .join(
                    DBvdWTypeRecord,
                    DBvdWTypeRecord.parent_id == DBMoleculeInfoRecord.id
                )
            )
            if forcefields is not None:
                results = results.filter(DBvdWTypeRecord.forcefield.in_(forcefields))
            if smiles is not None:
                results = results.filter(DBMoleculeInfoRecord.smiles.in_(smiles))
            
            records = defaultdict(lambda: {
                "smiles": None,
                "chemical_environment_counts": {},
                "element_counts": {},
                "vdw_type_counts": {}
            })
            for molid, molsmiles, molenv, molel, vdwid, vdwff, vdwtypes in results:
                data = records[molid]
                data["smiles"] = molsmiles
                data["chemical_environment_counts"] = molenv
                data["element_counts"] = molel
                data["vdw_type_counts"][vdwff] = vdWTypeRecord(
                    forcefield=vdwff,
                    type_counts=vdwtypes
                )
        return [MoleculeInfoRecord(**kwargs) for kwargs in records.values()]