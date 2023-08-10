# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--
The rules for this file:
  * entries are sorted newest-first.
  * summarize sets of changes - don't reproduce every git log comment here.
  * don't ever delete anything.
  * keep the format consistent (79 char width, M/D/Y date format) and do not
    use tabs but use spaces for formatting
  * accompany each entry with github issue/PR number (Issue #xyz)
-->


## [Unreleased]

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


### Fixed
<!-- Bug fixes -->
- pytest runs (PR #44)

### Changed
<!-- Changes in existing functionality -->
- Major refactor to move to using Arrow databases (PR #45)

## v0.2.3

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

## v0.2.2

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

## v0.2.1

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

## v0.2.0

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
