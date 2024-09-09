# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
The rules for this file:
  * entries are sorted newest-first.
  * summarize sets of changes - don't reproduce every git log comment here.
  * don't ever delete anything.
  * keep the format consistent (79 char width, Y/M/D date format) and do not
    use tabs but use spaces for formatting
  * accompany each entry with github issue/PR number (Issue #xyz)
-->

## ??

### Authors
- [@lilyminium]

### Added
- General linear fit target and example (PR #131)

### Changed
- Removed unused, undocumented code paths, and updated docs (PR #132)


## v0.4.0 -- 2024-07-18

This version adds support for lookup tables.

### Authors
- [@lilyminium]
- [@mattwthompson]
- [@j-wags]

### Added
- Added lookup table support (`openff.nagl.lookups.AtomPropertiesLookupTable`) (PR #122, #129)
- Pins to DGL for compatibility (PR #117)

## v0.3.8 -- 2024-04-11

This version drops pyarrow and rich as core dependencies. It also removes Python 3.9 support.

### Authors
- [@lilyminium]

## What's Changed
- Switch to plain importlib (PR #109)
- Add unprocessed methods(PR #111)
- Remove dgl config (PR #110)
- Make pyarrow optional (PR #113)
- Remove rich library (PR #115)


## v0.3.7 -- 2024-04-05

### Authors
- [@lilyminium]
- [@mattwthompson]

### Fixed
- Refactored resonance enumeration to avoid generating whole
  molecules with itertools.product where possible (PR #102, Issue #101)
- Refactored NXMolGraph.in_edges to use np.where to speed code up (PR #102, Issue #101)
- Removed Binder tooling (PR #99)


## v0.3.6 -- 2024-03-22

### Authors
- [@lilyminium]

### Fixed
- Fixed typing of conformer generation from RDKit (PR #97)

## v0.3.5 -- 2024-03-21

### Authors
<!-- GitHub usernames of contributors to this release -->
- [@Yoshanuikabundi]
- [@lilyminium]

### Fixed
- Fixes to docs and examples (Issue #92, PR #73)


## v0.3.4 -- 2024-02-15

### Authors
<!-- GitHub usernames of contributors to this release -->
- [@lilyminium]
- [@mattwthompson]

### Fixed
<!-- Bug fixes -->
- Fixed CUDA launch error (Issue #81, PR #83)
- Updated versioneer (PR #86)
- Fixed batch distribution hardcoding (Issue #80, PR #82)
- Fixed node and edge typing, adding DGL 2.0 compatibility (Issue #78, PR #79)


## v0.3.3 - 2023-12-22

### Authors
<!-- GitHub usernames of contributors to this release -->
- [@lilyminium]
- [@IAlibay]

### Fixed
<!-- Bug fixes -->
- Fixed OpenFF molecule conversion with toolkit wrappers (Issue #69, PR #71)
- Bug report templates

## v0.3.2 - 2023-10-26

### Authors
<!-- GitHub usernames of contributors to this release -->
- [@lilyminium]

### Changed
- Added documentation link in README (PR #66, Issue #63)
- Removed `capture_toolkit_warnings` usage in Molecule
  creation as OpenFF Toolkit v0.14.4 no longer needs it,
  and removed warnings filter from method (PR #65, Issue #64)

## v0.3.1 - 2023-09-08

### Authors
<!-- GitHub usernames of contributors to this release -->
- [@lilyminium]

### Changed
<!-- Changes in existing functionality -->
- Guarded openff.toolkit imports (PR #61, Issue #56)

## v0.3.0 - 2023-08-29

### Authors
<!-- GitHub usernames of contributors to this release -->
- [@lilyminium]
- [@mattwthompson]

### Reviewers
- [@lilyminium]

### Added
- Codecov coverage (PR #43)
- Multiple target objectives and configuratin files (PR #45)
- Support for pydantic v2 (PR #46)
- Added `ChemicalDomain` and model versioning (PR #54)


### Fixed
<!-- Bug fixes -->
- pytest runs (PR #44)
- documentation badge and links (PR #51, #52)

### Changed
<!-- Changes in existing functionality -->
- Major refactor to move to using Arrow databases (PR #45, PR #48)
- Removed importing `OFFMolecule` in favour of `Molecule` (PR #50, Issue #13)

### Removed
- Old `_app` and `_cli` utilities that were not well tested
  are not supported in refactor (PR #49)

## v0.2.3 - 2023-07-05

### Authors
<!-- GitHub usernames of contributors to this release -->
- [@lilyminium]
- [@Yoshanuikabundi]

### Reviewers
- [@lilyminium]

### Added
- Link to OpenFF install guide (PR #37)

### Fixed
<!-- Bug fixes -->
- Versioneer version prefix (PR #42)

### Changed
<!-- Changes in existing functionality -->
- Update examples for central examples page (PR #40)

## v0.2.2 - 2023-03-29

### Authors
<!-- GitHub usernames of contributors to this release -->
- [@lilyminium]
- [@Yoshanuikabundi]

### Reviewers
- [@lilyminium]

### Added
- `GNNModel.load` (PR #26)
- `GNNModel.save` function (PR #29)
- General documentation (PR #21, PR #33)
- Documentation of creation of custom features (PR #36)

### Fixed
<!-- Bug fixes -->
- Fix edge ordering in NXMol graph (Issue #28, PR #34)

### Changed
<!-- Changes in existing functionality -->
- Migrate away from pkg_resources to importlib_resources (PR #35)
- Expose base classes and hide metaclasses (PR #32)
- Update examples for central examples page (PR #40)

## v0.2.1 - 2023-03-02

### Authors
<!-- GitHub usernames of contributors to this release -->
- [@lilyminium]

### Reviewers
- [@mattwthompson]

### Added
- `GNNModel.load` function (PR #26)
- `convolution_dropout` and `readout_dropout` keywords to GNNModel (PR #26)

### Fixed
<!-- Bug fixes -->
- Versioneer `__version__` string (PR #25)

## v0.2.0 - 2023-02-05

### Authors
<!-- GitHub usernames of contributors to this release -->
- [@SimonBoothroyd]
- [@jthorton]
- [@mattwthompson]
- [@lilyminium]

### Added
<!-- New added features -->
- New toolkit wrappers (PR #14, PR #16)

### Fixed
<!-- Bug fixes -->

### Changed
<!-- Changes in existing functionality -->
- Refactored code so DGL is not required for install or model inference (PR #23)

### Deprecated
<!-- Soon-to-be removed features -->

### Removed
<!-- Removed features -->

[@lilyminium]: https://github.com/lilyminium
[@mattwthompson]: https://github.com/mattwthompson
[@Yoshanuikabundi]: https://github.com/Yoshanuikabundi
[@IAlibay]: https://github.com/IAlibay
[@jthorton]: https://github.com/jthorton
[@SimonBoothroyd]: https://github.com/SimonBoothroyd
