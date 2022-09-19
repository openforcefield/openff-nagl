# def match_conformers(
#     indexed_smiles: str,
#     db_conformers: List[DBConformerRecord],
#     query_conformers: List[ConformerRecord],
# ) -> Dict[int, int]:
#     """A method which attempts to match a set of new conformers to store with
#     conformers already present in the database by comparing the RMS of the
#     two sets.

#     Args:
#         indexed_smiles: The indexed SMILES pattern associated with the conformers.
#         db_conformers: The database conformers.
#         conformers: The conformers to store.

#     Returns:
#         A dictionary which maps the index of a conformer to the index of a database
#         conformer. The indices of conformers which do not match an existing database
#         conformer are not included.
#     """

#     from openff.toolkit.topology import Molecule
#     from gnn_charge_models.utils.openff import is_conformer_identical

#     molecule = Molecule.from_mapped_smiles(
#         indexed_smiles, allow_undefined_stereo=True
#     )

#     # See if any of the conformers to add are already in the DB.
#     matches = {}

#     for q_index, query in enumerate(query_conformers):
#         for db_index, db_conformer in enumerate(db_conformers):
#             if is_conformer_identical(molecule, query.coordinates, db_conformer.coordinates):
#                 matches[q_index] = db_index
#                 break
#     return matches

# def _store_new_data(
#     smiles,
#     db_record,
#     new_record,
#     db_class = DBPartialChargeSet,
#     container_name: str = "partial_charges",
# ):
#     db_container = getattr(db_record, container_name)
#     existing_methods = [x.method for x in db_container]
#     for new_data in getattr(new_record, container_name):
#         if new_data.method in existing_methods:
#             raise RuntimeError(
#                 f"{new_data.method} {container_name} already stored for {smiles} "
#                 f"with coordinates {new_record.coordinates}"
#             )
#         db_data = db_class(method=new_data.method, values=new_data.values)
#         db_container.append(db_data)


# def store_conformer_records(
#     db_parent: DBMoleculeRecord, records: List[ConformerRecord]
# ):
#     """Store a set of conformer records in an existing DB molecule record."""

#     if len(db_parent.conformers) > 0:
#         LOGGER.warning(
#             f"An entry for {db_parent.smiles} is already present in the molecule store. "
#             f"Trying to find matching conformers."
#         )

#     conformer_matches = match_conformers(db_parent.smiles, db_parent.conformers, records)

#     # Create new database conformers for those unmatched conformers.
#     for i, record in enumerate(records):
#         if i in conformer_matches:
#             continue

#         db_conformer = DBConformerRecord(coordinates=record.coordinates)
#         db_parent.conformers.append(db_conformer)
#         conformer_matches[i] = len(db_parent.conformers) - 1

#     DB_CLASSES = {
#         "partial_charges": DBPartialChargeSet,
#         "bond_orders": DBWibergBondOrderSet,
#     }

#     for index, db_index in conformer_matches.items():
#         db_record = db_parent.conformers[db_index]
#         record = records[index]
#         for container_name, db_class in DB_CLASSES.items():
#             _store_new_data(db_parent.smiles, db_record, record, db_class, container_name)
