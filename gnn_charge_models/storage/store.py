import logging
import pathlib
from collections import defaultdict
from contextlib import contextmanager
import time
from typing import ContextManager, Dict, List, Optional, Tuple

from gnn_charge_models.utils.types import Pathlike

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from .db import (
    DBBase,
    DBPartialChargeSet,
    DBWibergBondOrderSet,
    DBMoleculeRecord,
)

from .record import (
    ChargeMethod,
    WibergBondOrderMethod,
    PartialChargeRecord,
    WibergBondOrderRecord,
    ConformerRecord, MoleculeRecord,
)
from .session import (
    DBSessionManager,
    DBQueryResult
)

from gnn_charge_models.utils.utils import as_iterable
from gnn_charge_models.utils.time import PerformanceTimer

LOGGER = logging.getLogger(__name__)


def db_columns_to_models(
    db_partial_charge_columns: List[DBQueryResult],
    db_bond_order_columns: List[DBQueryResult],
) -> List[MoleculeRecord]:
    """Maps a set of database records into their corresponding data models,
    optionally retaining only partial charge sets and WBO sets computed with a
    specified method.

    Args:

    Returns:
        The mapped data models.
    """
    raw_objects = defaultdict(
        lambda: {
            "smiles": None,
            "conformers": defaultdict(
                lambda: {
                    "coordinates": None,
                    PartialChargeRecord: {},
                    WibergBondOrderRecord: {},
                }
            ),
        }
    )

    MODEL_TYPES = {
        PartialChargeRecord: db_partial_charge_columns,
        WibergBondOrderRecord: db_bond_order_columns,
    }

    for model_type, db_columns in MODEL_TYPES.items():
        for result in db_columns:
            molecule = raw_objects[result.molecule_id]
            molecule["smiles"] = result.molecule_smiles
            conformer = molecule["conformers"][result.conformer_id]
            conformer["coordinates"] = result.conformer_coordinates
            model = model_type.construct(
                method=result.method, values=result.values)
            conformer[model_type][result.method] = model

    records = []
    for molecule_args in raw_objects.values():
        conformers = []
        for conformer_args in molecule_args["conformers"].values():
            conformer = ConformerRecord.construct(
                coordinates=conformer_args["coordinates"],
                partial_charges=conformer_args[PartialChargeRecord],
                bond_orders=conformer_args[WibergBondOrderRecord],
            )
            conformers.append(conformer)

        molecule = MoleculeRecord.construct(
            mapped_smiles=molecule_args["smiles"], conformers=conformers)
        records.append(molecule)

    return records


class MoleculeStore:

    # match_conformers = staticmethod(match_conformers)
    # store_conformer_records = staticmethod(store_conformer_records)

    def __len__(self):
        with self._get_session() as db:
            return db.query(DBMoleculeRecord.smiles).count()

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
            autocommit=False, autoflush=False, bind=self.engine,
        )

        with self._get_session() as db:
            self.db_version = db.check_version()
            self.general_provenance = db.get_general_provenance()
            self.software_provenance = db.get_software_provenance()

    @contextmanager
    def _get_session(self) -> ContextManager[Session]:
        session = self._sessionmaker()
        try:
            yield DBSessionManager(session)
            session.commit()
        except BaseException as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def set_provenance(
        self,
        general_provenance: Dict[str, str],
        software_provenance: Dict[str, str],
    ):
        """Set the stores provenance information.

        Parameters
        ----------
        general_provenance
            A dictionary storing provenance about the store such as the author,
            when it was generated etc.
        software_provenance
            A dictionary storing the provenance of the software and packages used
            to generate the data in the store.
        """

        with self._get_session() as db:
            db.set_provenance(general_provenance=general_provenance,
                              software_provenance=software_provenance)

    def get_charge_methods(self) -> List[str]:
        """A list of the methods used to compute the partial charges in the store."""

        with self._get_session() as db:
            return [
                method for (method,) in db.query(DBPartialChargeSet.method).distinct()
            ]

    def get_wbo_methods(self) -> List[str]:
        """A list of the methods used to compute the WBOs in the store."""

        with self._get_session() as db:
            return [
                method for (method,) in db.query(DBWibergBondOrderSet.method).distinct()
            ]

    def get_smiles(self) -> List[str]:
        with self._get_session() as db:
            return [
                smiles for (smiles,) in db.query(DBMoleculeRecord.smiles).distinct()
            ]

    def retrieve(
        self,
        partial_charge_methods: Optional[List[ChargeMethod]] = None,
        bond_order_methods: Optional[List[WibergBondOrderMethod]] = None,
    ) -> List[MoleculeRecord]:
        """Retrieve records stored in this data store

        Args:
            partial_charge_methods: The (optional) list of charge methods to retrieve
                from the store. By default (`None`) all charges will be returned.
            bond_order_methods: The (optional) list of bond order methods to retrieve
                from the store. By default (`None`) all bond orders will be returned.

        Returns:
            The retrieved records.
        """
        if partial_charge_methods is not None:
            partial_charge_methods = [
                ChargeMethod(x) for x in as_iterable(partial_charge_methods)]
        if bond_order_methods is not None:
            bond_order_methods = [
                WibergBondOrderMethod(x) for x in as_iterable(bond_order_methods)]

        with self._get_session() as db:
            db_partial_charge_columns = []
            db_bond_order_columns = []

            with PerformanceTimer(
                LOGGER,
                start_message="performing SQL queries",
                end_message="performed SQL query"
            ):
                db_partial_charge_columns = db.query_by_method(
                    DBPartialChargeSet,
                    allowed_methods=partial_charge_methods,
                )

                db_bond_order_columns = db.query_by_method(
                    DBWibergBondOrderSet,
                    allowed_methods=bond_order_methods
                )

            with PerformanceTimer(
                LOGGER,
                start_message="converting SQL columns to entries",
                end_message="converted SQL columns to entries",
            ):
                records = db_columns_to_models(
                    db_partial_charge_columns, db_bond_order_columns
                )

        return records

    def store(self, records: Tuple[MoleculeRecord] = tuple(), suppress_toolkit_warnings: bool = True):
        """Store the molecules and their computed properties in the data store.

        Parameters
        ----------
        records
            The records to store.
        """
        from gnn_charge_models.utils.openff import smiles_to_inchi_key, capture_toolkit_warnings

        records = as_iterable(records)

        with capture_toolkit_warnings(run=suppress_toolkit_warnings):
            records_by_inchi_key = defaultdict(list)

            for record in tqdm(records, desc="grouping records to store by InChI key"):
                inchi_key = smiles_to_inchi_key(record.mapped_smiles)
                records_by_inchi_key[inchi_key].append(record)

            with self._get_session() as db:
                for inchi_key, inchi_records in tqdm(
                    records_by_inchi_key.items(),
                    desc="storing grouped records"
                ):
                    db.store_records_with_inchi_key(inchi_key, inchi_records)
