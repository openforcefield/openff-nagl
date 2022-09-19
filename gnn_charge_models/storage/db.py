import logging
from typing import TYPE_CHECKING, Dict, List

from sqlalchemy import (
    Column,
    Enum,
    ForeignKey,
    Integer,
    PickleType,
    String,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from .record import ChargeMethod, WibergBondOrderMethod

DBBase = declarative_base()

DB_VERSION = 1

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .record import ConformerRecord


class DBPartialChargeSet(DBBase):

    __tablename__ = "partial_charge_sets"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("conformers.id"), nullable=False, index=True)

    method = Column(Enum(ChargeMethod), nullable=False)
    values = Column(PickleType, nullable=False)

    __table_args__ = (
        UniqueConstraint("parent_id", "method", name="_pc_parent_method_uc"),
    )


class DBWibergBondOrderSet(DBBase):

    __tablename__ = "wiberg_bond_order_sets"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("conformers.id"), nullable=False, index=True)

    method = Column(Enum(WibergBondOrderMethod), nullable=False)
    values = Column(PickleType, nullable=False)

    __table_args__ = (
        UniqueConstraint("parent_id", "method", name="_wbo_parent_method_uc"),
    )


class DBConformerRecord(DBBase):

    __tablename__ = "conformers"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    coordinates = Column(PickleType, nullable=False)

    partial_charges = relationship("DBPartialChargeSet", cascade="all, delete-orphan")
    bond_orders = relationship("DBWibergBondOrderSet", cascade="all, delete-orphan")

    def _store_new_data(
        self,
        new_record,
        mapped_smiles,
        db_class=DBPartialChargeSet,
        container_name: str = "partial_charges",
    ):
        db_container = getattr(self, container_name)
        existing_methods = [x.method for x in db_container]
        for new_data in getattr(new_record, container_name).values():
            if new_data.method in existing_methods:
                raise RuntimeError(
                    f"{new_data.method.value} {container_name} already stored for {mapped_smiles} "
                    f"with coordinates {new_record.coordinates}"
                )
            db_data = db_class(method=new_data.method, values=new_data.values)
            db_container.append(db_data)


class DBMoleculeRecord(DBBase):

    __tablename__ = "molecules"

    id = Column(Integer, primary_key=True, index=True)

    inchi_key = Column(String(20), nullable=False, index=True)
    mapped_smiles = Column(String, nullable=False)

    conformers = relationship("DBConformerRecord", cascade="all, delete-orphan")

    def store_conformer_records(self, records: List["ConformerRecord"]):
        """Store a set of conformer records in an existing DB molecule record."""

        if len(self.conformers) > 0:
            LOGGER.warning(
                f"An entry for {self.mapped_smiles} is already present in the molecule store. "
                f"Trying to find matching conformers."
            )

        conformer_matches = match_conformers(
            self.mapped_smiles, self.conformers, records
        )

        # Create new database conformers for those unmatched conformers.
        for i, record in enumerate(records):
            if i in conformer_matches:
                continue

            db_conformer = DBConformerRecord(coordinates=record.coordinates)
            self.conformers.append(db_conformer)
            conformer_matches[i] = len(self.conformers) - 1

        DB_CLASSES = {
            "partial_charges": DBPartialChargeSet,
            "bond_orders": DBWibergBondOrderSet,
        }

        for index, db_index in conformer_matches.items():
            db_record: DBConformerRecord = self.conformers[db_index]
            record = records[index]
            for container_name, db_class in DB_CLASSES.items():
                db_record._store_new_data(
                    record, self.mapped_smiles, db_class, container_name
                )


class DBGeneralProvenance(DBBase):

    __tablename__ = "general_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=False)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBSoftwareProvenance(DBBase):

    __tablename__ = "software_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=False)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBInformation(DBBase):
    """A class which keeps track of the current database
    settings.
    """

    __tablename__ = "db_info"

    version = Column(Integer, primary_key=True)

    general_provenance = relationship(
        "DBGeneralProvenance", cascade="all, delete-orphan"
    )
    software_provenance = relationship(
        "DBSoftwareProvenance", cascade="all, delete-orphan"
    )


def match_conformers(
    indexed_mapped_smiles: str,
    db_conformers: List[DBConformerRecord],
    query_conformers: List["ConformerRecord"],
) -> Dict[int, int]:
    """A method which attempts to match a set of new conformers to store with
    conformers already present in the database by comparing the RMS of the
    two sets.

    Args:
        indexed_mapped_smiles: The indexed mapped_smiles pattern associated with the conformers.
        db_conformers: The database conformers.
        conformers: The conformers to store.

    Returns:
        A dictionary which maps the index of a conformer to the index of a database
        conformer. The indices of conformers which do not match an existing database
        conformer are not included.
    """

    from openff.toolkit.topology import Molecule

    from gnn_charge_models.utils.openff import is_conformer_identical

    molecule = Molecule.from_mapped_smiles(
        indexed_mapped_smiles, allow_undefined_stereo=True
    )

    # See if any of the conformers to add are already in the DB.
    matches = {}

    for q_index, query in enumerate(query_conformers):
        for db_index, db_conformer in enumerate(db_conformers):
            if is_conformer_identical(
                molecule, query.coordinates, db_conformer.coordinates
            ):
                matches[q_index] = db_index
                break
    return matches
