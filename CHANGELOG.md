# Changelog

All notable changes to this project will be documented in this file.

## v0.6.1 (2026-04-29)

### Fix

- change color of the lines in the inventory table for coherence

## v0.6.0 (2026-04-24)

### Feat

- add clearer message error (message with missing calibration)
- add proper inventory table (`matisse reduce --check-files`)
- change parameter name --spectralBinning to --spectralAverage (with backward compatibility).
- add resol and band into in blocs_status. Change default value of band as `ALL` to reduce all available resolution (previously `LOW`)
- add default master calibration for magic numbers
- add sub-band option to compute metrics over L/M/N band only (for bcd)
- perform matisse format at the end of reduce (instead of separated command). By default, `matisse reduce` produces `reduced/` and `reduced_OIFITS/`.

### Fix

- update change about parameters on tests and flux
- timespan in hours for calibrator selection
- update help with the new reduced output

### Refactor

- remove prefix and change default compute directory
- add doc to calibrate (--cumul-block expert only)
- polish log of calibrate
- centralize outputs into single directory and drop max iter concept

## v0.5.0 (2026-03-24)

### Feat

- add reference calibrator database from zenodo
- add calibration status in doctor
- add cli command flux_calibrate to perform the calibration
- add set of function/helpers to perform flux calibration

### Fix

- reverse airmass correction order for L band
- timeout issue on starflux fallback to local database

### Refactor

- remove legacy calibrator database, point to cache
- update esorex command execution to use subprocess for better output handling

## v0.4.0 (2026-03-03)

### Feat

- Add interactive mode of show in the CLI (tested)

### Fix

- Update telescope color handling in case of multiple exposure
- Implement BCD extraction from filename and add corresponding tests

## v0.3.0 (2026-02-24)

### Feat

- add merge command to BCD CLI and enhance find_sci_filename to include CAL files
- enhance category determination in _read_category method for OIFitsReader
- add remove command to handle BCD ordering in SCI OIFITS files
- improve path handling in generate_sof_files and run_esorex_calibration functions
- add compare command for BCD corrections and update visualization utilities
- add plot of BCD correction
- add quality check of the BCD correction
- add BCD command group for magic numbers computation and correction

### Fix

- update help text for file directories and apply function description in BCD CLI
- remove unused '--no-chopping' option from BCD CLI test
- update CLI commands from 'magic' to 'bcd compute' in test cases

### Refactor

- update import statements for Sequence from collections.abc

## v0.2.0 (2026-02-04)

### Feat

- add ESO pipeline installation guidance in doctor command

## v0.1.1 (2026-02-03)

- Use of commitizen to test the automatized version updating

## v0.1.0 (2026-01-01)

- Initial release of MATISSE pipeline for pypi
- Core calibration pipeline interface
- CLI interface with `matisse` command
        - Automated calibration (`matisse calibrate`)
        - Automated data reduction (`matisse reduce`)
- Interactive viewer with Plotly visualization
- BCD (Beam-commuting device) correction module
- OIFITS data reader and processor
- Configuration system for pipeline parameters
- Comprehensive test suite

### Features

- Support for MATISSE interferometric data
- Python 3.10+ support
- CLI tools: calibrate, reduce, magic, show, format, etc.
