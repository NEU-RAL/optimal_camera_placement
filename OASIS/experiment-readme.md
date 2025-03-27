# Experiment Runner Documentation

This document explains how to use the experiment runner.

## Quick Start

To run experiments with default settings:

```bash
python test_space.py
```

This will run Frank-Wolfe, randomized rounding, contention resolution, and greedy algorithms on a problem of size n=100, m=100.

## Command Line Options

The scripy supports various options to customize the experiments:

```bash
python test_space.py --n 50 100 --m 20 50 --algorithms fw rounding cr greedy --runs 3 --verbose --output-format json --output-dir results/
```

### Problem Parameters

- `--n`: Problem dimension values to test (list of integers)
  - Example: `--n 50 100 150` tests three different problem sizes
  - Default: `[100]`

- `--m`: Constraint dimension values to test (list of integers)
  - Example: `--m 20 50` tests two different constraint sizes
  - Default: `[100]`

### Algorithm Selection

- `--algorithms`: Which algorithms to run (list of algorithm names)
  - Options: `fw`, `rounding`, `cr`, `greedy`, `gurobi`, `all`
  - Example: `--algorithms fw cr greedy` runs only those three algorithms
  - Use `--algorithms all` to run all available algorithms
  - Default: `['fw', 'rounding', 'cr', 'greedy']`

### Experiment Parameters

- `--runs`: Number of runs per (n,m) pair with different random seeds
  - Example: `--runs 5` performs five runs and averages the results
  - Default: `1`

- `--time-limit`: Maximum runtime in seconds for algorithms that support it
  - Example: `--time-limit 300` sets a 5-minute limit
  - Default: `None` (no time limit)

- `--verbose`: Enable detailed progress output
  - Default: Disabled (minimal output)

### Output Options

- `--output-format`: Format for storing results
  - Options: `json`, `pickle`, `hdf5`
  - Default: `json`

- `--output-dir`: Directory where results will be saved
  - Example: `--output-dir experiments/results/`
  - Default: `./` (current directory)

## Algorithms

The experiment runner supports the following algorithms:

1. **Frank-Wolfe (FW)**: Solves the continuous relaxation of the problem.
   - Command line flag: `fw`
   - Result key: `continuous_relaxation`

2. **Randomized Rounding (RR)**: Rounds the continuous solution using randomized rounding.
   - Command line flag: `rounding`
   - Result key: `randomized_rounding`
   - Note: Works when `fw` is also enabled

3. **Contention Resolution (CR)**: Uses contention resolution to round the continuous solution.
   - Command line flag: `cr`
   - Result key: `contention_resolution`
   - Note: Works when `fw` is also enabled

4. **Greedy Algorithm**: Greedily selects elements to maximize the objective.
   - Command line flag: `greedy`
   - Result key: `greedy`

5. **Gurobi Branch & Cut**: Uses Gurobi to find the optimal solution (if available).
   - Command line flag: `gurobi`
   - Result key: `gurobi`
   - Note: Requires Gurobi to be installed

## Output File Structure

### JSON Format

The output JSON files are structured as follows:

```json
{
  "params": {
    "n": 100,
    "m": 50,
    "algorithms": ["fw", "rounding", "cr", "greedy"],
    "num_runs": 3,
    "time_limit": null
  },
  "runs": {
    "run_0": {
      "continuous_relaxation": {
        "obj": 1234.5678,
        "time": 1.234,
        "iterations": 100,
        "selection": [0.1, 0.5, 0.8, ...],
        "variance_info": [...]
      },
      "randomized_rounding": {
        "obj": 1200.1234,
        "time": 0.123,
        "selection": [0, 1, 1, ...],
        "variance_info": [...]
      },
      "contention_resolution": {
        "obj": 1195.6789,
        "time": 0.456,
        "selection": [0, 1, 1, ...],
        "variance_info": [...]
      },
      "greedy": {
        "obj": 1150.1234,
        "time": 2.345,
        "iterations": 15,
        "obj_evaluations": 1500,
        "selection": [0, 1, 0, ...],
        "variance_info": [...]
      }
    },
    "run_1": { ... },
    "run_2": { ... }
  },
  "stats": {
    "n": 100,
    "m": 50,
    "num_runs": 3,
    "continuous_relaxation": {
      "obj_mean": 1234.5678,
      "obj_std": 12.3456,
      "obj_min": 1220.1234,
      "obj_max": 1245.6789,
      "time_mean": 1.234,
      "time_std": 0.123,
      "iterations_mean": 100.0,
      "iterations_std": 5.0
    },
    "randomized_rounding": { ... },
    "contention_resolution": { ... },
    "greedy": { ... }
  }
}
```

### Key Sections

1. **params**: Input parameters used for the experiment
   - Problem dimensions and algorithm selection
   - Number of runs and time limits

2. **runs**: Detailed results for each individual run
   - One entry per run, containing results for each algorithm
   - For each algorithm:
     - `obj`: Objective function value (higher is better)
     - `time`: Execution time in seconds
     - `selection`: Selected elements (continuous or binary vector)
     - `variance_info`: Information about variance in the solution
     - Algorithm-specific metrics (iterations, evaluations, etc.)

3. **stats**: Aggregated statistics across all runs
   - For each algorithm:
     - `obj_mean`, `obj_std`, `obj_min`, `obj_max`: Statistics of objective values
     - `time_mean`, `time_std`: Statistics of execution times
     - Algorithm-specific statistics (e.g., iterations)

## Interpreting Results

The summary table printed at the end of the experiment provides a quick way to compare algorithm performance across different problem sizes.

For each algorithm, the summary shows:
- **obj**: Average objective value (higher is better)
- **time**: Average execution time in seconds

Example summary table:
```
FINAL SUMMARY TABLE
--------------------------------------------------------------------------------
     n       m |    FW_obj      FW_time |    RR_obj      RR_time |    CR_obj      CR_time |   GRD_obj     GRD_time
--------------------------------------------------------------------------------
   100      50 |   1234.5678      1.23 |   1200.1234      0.12 |   1195.6789      0.46 |   1150.1234      2.34
   100     100 |   1100.1234      2.34 |   1050.6789      0.23 |   1045.1234      0.56 |   1000.6789      4.56
--------------------------------------------------------------------------------
```
For detailed analysis, examine the full JSON output which contains complete statistics and individual run data.
