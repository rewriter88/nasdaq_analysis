# Workspace Organization

## Current Root Directory Structure

The workspace has been cleaned up to keep only essential files in the root directory:

### Essential Files (Root Directory)
- `config.py` - Main configuration file with FMP API key and settings
- `run_analysis.py` - Analysis launcher script
- `nasdaq_fmp_analysis_corrected.py` - FMP-based momentum strategy analysis
- `simple_rolling_analysis.py` - **NEW**: Rolling period analysis for consistency testing
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation
- `DATA_SOURCE_GUIDE.md` - Data source configuration guide
- `WORKSPACE_ORGANIZATION.md` - This file

### Active Data & Cache
- `fmp_cache_corrected/` - FMP API cache directory

### Results & Outputs
- `results/charts/` - Current chart outputs
- `results/csv/` - Current CSV rebalancing event files

### Archive Structure
- `archive/old_analysis_scripts/` - Previous versions of analysis scripts
  - `nasdaq_yahoo_analysis.py` - Yahoo Finance version (archived)
- `archive/old_charts/` - Previous chart outputs
- `archive/old_documentation/` - Previous documentation versions
- `archive/old_cache/` - Previous cache directories
  - `yahoo_cache/` - Yahoo Finance cache (archived)
- `archive/old_csv_outputs/` - Previous CSV outputs
- `archive/Backups/` - Original backup folder
- `archive/C_Figment-Tindex/` - Other project files
- `archive/Lionscrest_Analysis/` - Other project files
- `archive/src/` - Old source directory
- `archive/data/` - Old data directory

## Usage

**Main Analysis** (single time period):
1. Ensure your FMP API key is set in `config.py`
2. Run: `python run_analysis.py`
3. Charts will be saved to `results/charts/`
4. CSV files will be saved to `results/csv/`

**Rolling Period Analysis** (NEW - tests consistency across multiple time periods):
1. Set `ENABLE_ROLLING_ANALYSIS = True` in `config.py` (if not already)
2. Run: `python simple_rolling_analysis.py`
3. Creates CSV tables showing:
   - Which portfolio won in each time period
   - Summary of how many times each portfolio won first place
4. Results saved to `results/csv/rolling_period_analysis_[timestamp].csv`

The analysis uses Financial Modeling Prep API for premium data quality with proper survivorship bias correction.

All old/backup files have been moved to `archive/` to keep the workspace clean and organized.

## Related Projects

- **Lionscrest Analysis**: Located at `/Users/ricardoellstein/lionscrest-analysis/` (separate workspace)
  - Professional ETF analysis framework
  - Multi-source data validation
  - Can be opened as a separate VS Code workspace
