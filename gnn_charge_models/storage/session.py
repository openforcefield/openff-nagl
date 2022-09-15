from typing import NamedTuple, Optional, Union, Type, TYPE_CHECKING, List, Dict
from collections import defaultdict

import numpy as np

from .db import (
    DB_VERSION,
    DBConformerRecord,
    DBInformation,
    DBMoleculeRecord,
    DBPartialChargeSet,
    DBWibergBondOrderSet,
    DBSoftwareProvenance,
    DBGeneralProvenance,

)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from .record import MoleculeRecord


class DBQueryResult(NamedTuple):
    molecule_id: int
    molecule_smiles: str
    conformer_id: int
    conformer_coordinates: np.ndarray
    method: str
    values: np.ndarray

    def to_nested_dict(self):
        return {
            self.molecule_id: {
                "mapped_smiles": self.molecule_smiles,
                "conformers": {
                    self.conformer_id: {
                        "coordinates"
                    }
                }
            }
        }


class IncompatibleDBVersion(ValueError):
    """An exception raised when attempting to load a store whose
    version does not match the version expected by the framework.
    """

    def __init__(self, found_version: int, expected_version: int):
        """

        Parameters
        ----------
        found_version
            The version of the database being loaded.
        expected_version
            The expected version of the database.
        """

        super(IncompatibleDBVersion, self).__init__(
            f"The database being loaded is currently at version {found_version} "
            f"while the framework expects a version of {expected_version}. There "
            f"is no way to upgrade the database at this time, although this may "
            f"be added in future versions."
        )

        self.found_version = found_version
        self.expected_version = expected_version


class DBSessionManager:

    @staticmethod
    def map_records_by_smiles(db_records: List[DBMoleculeRecord]) -> Dict[str, List[DBMoleculeRecord]]:
        """Maps a list of DB records by their SMILES representation.

        Parameters
        ----------
        records
            The records to map.

        Returns
        -------
        A dictionary mapping the SMILES representation of a record to the record.
        """

        from openff.toolkit.topology import Molecule

        records = defaultdict(list)
        for record in db_records:
            offmol = Molecule.from_smiles(
                record.mapped_smiles,
                allow_undefined_stereo=True,
            )
            canonical_smiles = offmol.to_smiles(mapped=False)
            records[canonical_smiles].append(record)
        return records

    def __init__(self, session: "Session"):
        self.session = session
        self._db_info = None

    def query(self, *args, **kwargs):
        # print(args)
        return self.db.query(args, **kwargs)

    def check_version(self, version=DB_VERSION):
        if not self.db_info:
            db_info = DBInformation(version=version)
            self.db.add(db_info)
            self._db_info = db_info

        if self.db_info.version != version:
            raise IncompatibleDBVersion(self.db_info.version, version)
        return self.db_info.version

    def get_general_provenance(self):
        return {
            provenance.key: provenance.value
            for provenance in self.db_info.general_provenance
        }

    def get_software_provenance(self):
        return {
            provenance.key: provenance.value
            for provenance in self.db_info.software_provenance
        }

    def set_provenance(
        self,
        general_provenance: Dict[str, str],
        software_provenance: Dict[str, str],
    ):
        self.db_info.general_provenance = [
            DBGeneralProvenance(key=key, value=value)
            for key, value in general_provenance.items()
        ]
        self.db_info.software_provenance = [
            DBSoftwareProvenance(key=key, value=value)
            for key, value in software_provenance.items()
        ]

    @property
    def db_info(self):
        if self._db_info is None:
            self._db_info = self.db.query(DBInformation).first()
        return self._db_info

    @property
    def db(self):
        return self.session

    def query_by_method(
        self,
        model_type: Union[Type[DBPartialChargeSet], Type[DBWibergBondOrderSet]],
        allowed_methods: Optional[List[str]] = None,
    ) -> List[DBQueryResult]:
        """Returns the results of querying the DB for records associated with either a
        set of partial charge or bond order methods

        Returns:
            A list of tuples of the form::

                (
                    DBMoleculeRecord.id,
                    DBMoleculeRecord.smiles,
                    DBConformerRecord.id,
                    DBConformerRecord.coordinates,
                    model_type.method,
                    model_type.values
                )
        """
        if allowed_methods is not None and not allowed_methods:
            return []

        results = (
            self.db.query(
                DBMoleculeRecord.id,
                DBMoleculeRecord.mapped_smiles,
                DBConformerRecord.id,
                DBConformerRecord.coordinates,
                model_type.method,
                model_type.values,
            )
            .order_by(DBMoleculeRecord.id)
            .join(
                DBConformerRecord,
                DBConformerRecord.parent_id == DBMoleculeRecord.id,
            )
            .join(
                model_type,
                model_type.parent_id == DBConformerRecord.id,
            )
        )
        if allowed_methods is not None:
            results = results.filter(model_type.method.in_(allowed_methods))

        return [DBQueryResult(*result) for result in results.all()]

    def store_records_with_smiles(
        self,
        inchi_key: str,
        records: List["MoleculeRecord"],
        existing_db_record: Optional[DBMoleculeRecord] = None,
    ):
        """Stores a set of records which all store information for molecules with the
        same SMILES representation AND the same fixed hydrogen InChI key.

        Parameters
        ----------
        db
            The current database session.
        inchi_key
            The **fixed hydrogen** InChI key representation of the molecule stored in
            the records.
        records
            The records to store.
        """

        if existing_db_record is None:
            existing_db_record = DBMoleculeRecord(
                inchi_key=inchi_key, mapped_smiles=records[0].mapped_smiles)

        # Retrieve the DB indexed SMILES that defines the ordering the atoms in each
        # record should have and re-order the incoming records to match.
        expected_smiles = existing_db_record.mapped_smiles

        conformer_records = [
            conformer_record
            for record in records
            for conformer_record in record.reorder(expected_smiles).conformers
        ]

        existing_db_record.store_conformer_records(conformer_records)
        self.db.add(existing_db_record)

    def store_records_with_inchi_key(
        self, inchi_key: str, records: List["MoleculeRecord"]
    ):
        """Stores a set of records which all store information for molecules with the
        same fixed hydrogen InChI key.

        Parameters
        ----------
        db
            The current database session.
        inchi_key
            The **fixed hydrogen** InChI key representation of the molecule stored in
            the records.
        records
            The records to store.
        """

        existing_db_records: List[DBMoleculeRecord] = (
            self.db.query(DBMoleculeRecord)
            .filter(DBMoleculeRecord.inchi_key == inchi_key)
            .all()
        )

        db_records_by_smiles = self.map_records_by_smiles(existing_db_records)
        # Sanity check that no two DB records have the same InChI key AND the
        # same canonical SMILES pattern.
        multiple = [
            smiles
            for smiles, dbrecords in db_records_by_smiles.items()
            if len(dbrecords) > 1
        ]
        if multiple:
            raise RuntimeError(
                "The database is not self consistent."
                "There are multiple records with the same InChI key and SMILES."
                f"InChI key: {inchi_key} and SMILES: {multiple}"
            )
        db_records_by_smiles = {k: v[0]
                                for k, v in db_records_by_smiles.items()}

        records_by_smiles = self.map_records_by_smiles(records)
        for smiles, smiles_records in records_by_smiles.items():
            db_record = db_records_by_smiles.get(smiles, None)
            self.store_records_with_smiles(inchi_key, smiles_records, db_record)
