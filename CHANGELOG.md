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

## v0.3.3 - 2023-12-22

### Authors
<!-- GitHub usernames of contributors to this release -->
- @lilyminium
- @IAlibay

### Fixed
<!-- Bug fixes -->
- Fixed OpenFF molecule conversion with toolkit wrappers (Issue #69, PR #71)
- Bug report templates

## v0.3.2 - 2023-10-26

### Authors
<!-- GitHub usernames of contributors to this release -->
- @lilyminium

### Changed
- Added documentation link in README (PR #66, Issue #63)
- Removed `capture_toolkit_warnings` usage in Molecule
  creation as OpenFF Toolkit v0.14.4 no longer needs it,
  and removed warnings filter from method (PR #65, Issue #64)

## v0.3.1 - 2023-09-08

### Authors
<!-- GitHub usernames of contributors to this release -->
- @lilyminium

### Changed
<!-- Changes in existing functionality -->
- Guarded openff.toolkit imports (PR #61, Issue #56)

## v0.3.0 - 2023-08-29

### Authors
<!-- GitHub usernames of contributors to this release -->
- @lilyminium
- @mattwthompson

### Reviewers
- @lilyminium

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
- @lilyminium
- @yoshanuikabundi

### Reviewers
- @lilyminium

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
- @lilyminium
- @yoshanuikabundi

### Reviewers
- @lilyminium

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
- @lilyminium

### Reviewers
- @mattwthompson

### Added
- `GNNModel.load` function (PR #26)
- `convolution_dropout` and `readout_dropout` keywords to GNNModel (PR #26)

### Fixed
<!-- Bug fixes -->
- Versioneer `__version__` string (PR #25)

## v0.2.0 - 2023-02-05

### Authors
<!-- GitHub usernames of contributors to this release -->
- @sboothroyd
- @jthorton
- @mattwthompson
- @lilyminium

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
