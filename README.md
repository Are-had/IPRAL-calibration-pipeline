# SIRTA Lidar Calibration Pipeline

Calibration pipeline for IPRAL lidar data (355 nm) and comparison with ATLID satellite observations.

## Overview

This pipeline processes ground-based lidar data from SIRTA (IPRAL instrument) and performs:
- Signal calibration (analog and photocounting channels)
- ATB (Attenuated Total Backscatter) retrieval
- Klett inversion for aerosol backscatter coefficient
- Comparison with EarthCARE/ATLID L2 products

## Calibration Steps

### L0 → L1 (Signal Calibration)
1. **Range correction** - Apply r² correction to raw signals
2. **Background subtraction** - Remove background noise (delta)
3. **Calibration constant** - Apply K factor: `ATB = K × (RCS - delta)`
4. **Channel merging** - Merge analog + photocounting using Hanning window
5. **Polarization sum** - Combine parallel + perpendicular channels

### L1 → L2 (Inversion)
1. **Molecular profile** - Calculate AMB from radiosonde data (GRUAN)
2. **Klett inversion** - Retrieve aerosol backscatter coefficient (β_aer)

## Data

- **Date**: 2025-07-04
- **Instrument**: IPRAL @ SIRTA
- **Wavelength**: 355 nm
- **Channels**: Parallel & Perpendicular (Analog + Photocounting)

## Files

| File | Description |
|------|-------------|
| `pipeline.ipynb` | Main calibration notebook |
| `tools.py` | Calibration and inversion tools |

## Output

- Calibrated ATB profiles (clear & cloudy sky)
- Aerosol backscatter coefficient (Klett inversion)
- Comparison plots with ATLID data
- HDF5 file with all calibration parameters and results

## Requirements

- Python 3.x
- numpy, matplotlib, h5py, scipy, netCDF4
