# Workspace Organization

## Current Root Directory Structure

The workspace has been organized to keep only essential files in the root directory:

### Essential Files (Root Directory)
- `config.py` - Main configuration file
- `run_analysis.py` - Analysis launcher script
- `nasdaq_fmp_analysis_corrected.py` - FMP-based analysis (primary)
- `nasdaq_yahoo_analysis.py` - Yahoo Finance analysis (for comparison)
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `DATA_SOURCE_GUIDE.md` - Data source configuration guide

### Active Data & Cache
- `fmp_cache_corrected/` - Current FMP API cache directory
- `yahoo_cache/` - Yahoo Finance cache directory

### Results & Outputs
- `results/charts/` - Current chart outputs
- `results/csv/` - Current CSV rebalancing event files

### Archive Structure
- `archive/old_analysis_scripts/` - Previous versions of analysis scripts
- `archive/old_charts/` - Previous chart outputs
- `archive/old_documentation/` - Previous documentation versions
- `archive/old_cache/` - Previous cache directories
- `archive/old_csv_outputs/` - Previous CSV outputs
- `archive/Backups/` - Original backup folder
- `archive/C_Figment-Tindex/` - Other project files
- `archive/Lionscrest_Analysis/` - Other project files
- `archive/src/` - Old source directory
- `archive/data/` - Old data directory

## Usage

To run the analysis:
1. Configure `DATA_SOURCE` in `config.py` ("FMP" or "YAHOO")
2. Run: `python run_analysis.py`
3. Charts will be saved to `results/charts/`
4. CSV files will be saved to `results/csv/`

All old/backup files have been moved to `archive/` to keep the workspace clean and organized.
