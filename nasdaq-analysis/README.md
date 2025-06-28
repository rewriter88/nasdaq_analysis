# Nasdaq Analysis Project

This project analyzes the performance of selected Nasdaq stocks using historical price data. It implements strategies to calculate portfolio returns and visualize the results.

## Project Structure

- `src/main.py`: The main entry point of the application, containing the core logic for fetching data, calculating portfolio returns, and plotting the results.
- `src/utils/helpers.py`: Utility functions that assist with various tasks in the main application, such as building weight matrices and calculating cumulative returns.
- `data/`: Directory for storing data files. The `.gitkeep` file ensures this directory is tracked by Git.
- `requirements.txt`: Lists the Python dependencies required to run the project.
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd nasdaq-analysis
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the analysis, execute the following command:
```
python src/main.py
```

## Dependencies

This project requires the following Python packages:
- pandas
- numpy
- yfinance
- matplotlib

## License

This project is licensed under the MIT License.