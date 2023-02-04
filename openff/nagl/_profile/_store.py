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
    Enum,
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
        offmol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        return offmol.to_smiles(mapped=False)


class vdWTypeRecord(Record):
    forcefield: str
    type_counts: Dict[str, int]


class ElementRecord(Record):
    element: str
    count: int


class ChemicalEnvironmentRecord(Record):
    environment: ChemicalEnvironment
    count: int


class MoleculeInfoRecord(Record):
    smiles: str
    chemical_environment_counts: Dict[
        ChemicalEnvironment, ChemicalEnvironmentRecord
    ] = {}
    element_counts: Dict[str, ElementRecord] = {}
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

        zs = Counter([atom.symbol for atom in offmol.atoms])
        element_counts = {k: ElementRecord(element=k, count=zs[k]) for k in sorted(zs)}

        vdw_type_counts = {}
        for ff_file in forcefield_files:
            ff_name = pathlib.Path(ff_file).stem
            counts = cls._count_vdw_types(offmol, ff_file)
            vdw_type_counts[ff_name] = vdWTypeRecord(
                forcefield=ff_name, type_counts=counts
            )

        chemical_environment_counts = analyze_functional_groups(offmol)
        environment_records = {
            k: ChemicalEnvironmentRecord(environment=k, count=v)
            for k, v in chemical_environment_counts.items()
        }
        obj = cls(
            smiles=smiles,
            element_counts=element_counts,
            vdw_type_counts=vdw_type_counts,
            chemical_environment_counts=environment_records,
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
        self, forcefield_file: str, forcefield_name: Optional[str] = None
    ):
        from openff.toolkit.typing.engines.smirnoff import ForceField
        from openff.toolkit.topology import Molecule

        if forcefield_name is None:
            forcefield_name = pathlib.Path(forcefield_file).stem

        if forcefield_name in self.vdw_type_counts:
            raise ValueError(f"{forcefield_name} already has vdW type counts")

        with capture_toolkit_warnings():
            offmol = Molecule.from_smiles(self.smiles, allow_undefined_stereo=True)
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


class DBElementRecord(DBBase):
    __tablename__ = "elements"
    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    element = Column(String(3), nullable=False)
    count = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("parent_id", "element", name="_parent_element_uc"),
    )


class DBChemicalEnvironmentRecord(DBBase):
    __tablename__ = "chemical_environments"
    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    environment = Column(Enum(ChemicalEnvironment), nullable=False)
    count = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("parent_id", "environment", name="_parent_environment_uc"),
    )


class DBMoleculeInfoRecord(DBBase):
    __tablename__ = "molecules"

    id = Column(Integer, primary_key=True, index=True)
    smiles = Column(String, nullable=False)

    # chemical_environment_counts = Column(PickleType, nullable=False)
    # element_counts = Column(PickleType, nullable=False)
    chemical_environment_counts = relationship(
        "DBChemicalEnvironmentRecord", cascade="all, delete-orphan"
    )
    element_counts = relationship("DBElementRecord", cascade="all, delete-orphan")

    vdw_type_counts = relationship("DBvdWTypeRecord", cascade="all, delete-orphan")

    def store_vdw_type_record(self, vdw_type_record: vdWTypeRecord):
        existing_ffs = [record.forcefield for record in self.vdw_type_counts]
        if vdw_type_record.forcefield in existing_ffs:
            warnings.warn(
                f"{vdw_type_record.forcefield} force field already stored "
                f"for {self.smiles}. Skipping."
            )
            return
        vdw_data = DBvdWTypeRecord(
            forcefield=vdw_type_record.forcefield,
            type_counts=vdw_type_record.type_counts,
        )
        self.vdw_type_counts.append(vdw_data)

    def store_count_data(self, record):
        environment_records = [
            DBChemicalEnvironmentRecord(environment=rec.environment, count=rec.count)
            for rec in record.chemical_environment_counts.values()
        ]
        # for rec in environment_records:
        #     self.chemical_environment_counts.append(rec)
        self.chemical_environment_counts.extend(environment_records)

        element_records = [
            DBElementRecord(element=rec.element, count=rec.count)
            for rec in record.element_counts.values()
        ]
        # for rec in element_records:
        #     self.element_counts.append(rec)
        self.element_counts.extend(element_records)


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
                smiles for (smiles,) in db.query(DBMoleculeInfoRecord.smiles).distinct()
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
                    environment_counts = [
                        DBChemicalEnvironmentRecord(
                            environment=rec.environment, count=rec.count
                        )
                        for rec in record.chemical_environment_counts.values()
                    ]
                    element_counts = [
                        DBElementRecord(element=rec.element, count=rec.count)
                        for rec in record.element_counts.values()
                    ]
                    existing_record = DBMoleculeInfoRecord(
                        smiles=smiles,
                        chemical_environment_counts=environment_counts,
                        element_counts=element_counts
                        # record.chemical_environment_counts,
                        # element_counts=record.element_counts,
                    )
                    db.add(existing_record)
                    # existing_record.store_count_data(record)
                    # db.add(existing_record)
                else:
                    existing_record = existing[0]

                for vdw_record in record.vdw_type_counts.values():
                    existing_record.store_vdw_type_record(vdw_record)

    def retrieve(
        self,
        smiles: Optional[List[str]] = None,
        forcefields: Optional[List[str]] = None,
        chemical_environments: Optional[List[ChemicalEnvironment]] = None,
        elements: Optional[List[str]] = None,
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
                    DBChemicalEnvironmentRecord.environment,
                    DBChemicalEnvironmentRecord.count,
                    DBElementRecord.element,
                    DBElementRecord.count,
                    # DBMoleculeInfoRecord.chemical_environment_counts,
                    # DBMoleculeInfoRecord.element_counts,
                    DBvdWTypeRecord.id,
                    DBvdWTypeRecord.forcefield,
                    DBvdWTypeRecord.type_counts,
                )
                .order_by(DBMoleculeInfoRecord.id)
                .join(
                    DBvdWTypeRecord,
                    DBvdWTypeRecord.parent_id == DBMoleculeInfoRecord.id,
                )
                .join(
                    DBChemicalEnvironmentRecord,
                    DBChemicalEnvironmentRecord.parent_id == DBMoleculeInfoRecord.id,
                )
                .join(
                    DBElementRecord,
                    DBElementRecord.parent_id == DBMoleculeInfoRecord.id,
                )
            )
            if forcefields is not None:
                results = results.filter(DBvdWTypeRecord.forcefield.in_(forcefields))
            if smiles is not None:
                results = results.filter(DBMoleculeInfoRecord.smiles.in_(smiles))
            if chemical_environments is not None:
                results = results.filter(
                    DBChemicalEnvironmentRecord.environment.in_(chemical_environments)
                )
            if elements is not None:
                results = results.filter(DBElementRecord.element.in_(elements))

            records = defaultdict(
                lambda: {
                    "smiles": None,
                    "chemical_environment_counts": defaultdict(dict),
                    "element_counts": defaultdict(dict),
                    "vdw_type_counts": {},
                }
            )
            for (
                molid,
                molsmiles,
                env,
                envcount,
                el,
                elcount,
                vdwid,
                vdwff,
                vdwtypes,
            ) in results:
                data = records[molid]
                data["smiles"] = molsmiles
                data["chemical_environment_counts"][env] = ChemicalEnvironmentRecord(
                    environment=env, count=envcount
                )
                data["element_counts"][el] = ElementRecord(element=el, count=elcount)
                data["vdw_type_counts"][vdwff] = vdWTypeRecord(
                    forcefield=vdwff, type_counts=vdwtypes
                )
        return [MoleculeInfoRecord(**kwargs) for kwargs in records.values()]
