# Online Recalibrator

This library contains the codes for CORP.

## Requirements

- Python>=3.9
- [NeuralDecoder](../NeuralDecoder/)
- [LanguageModelDecoder](../LanguageModelDecoder/)

## Installation

```bash
pip install -e .
```

## Usage

```bash
cd examples
./run_recalibration.sh  # Run recalibration with CORP
./run_no_recalibration.sh  # Run no recalibration baseline
```

Plot the results with `examples/plot_offline_error_rate_curve.ipynb`
