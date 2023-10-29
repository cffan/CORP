# Continual Online Recalibration with Pseudo-labels (CORP)

This is the repo for paper [Plug-and-Play Stability for Intracortical Brain-Computer Interfaces: A One-Year Demonstration of Seamless Brain-to-Text Communication](https://openreview.net/forum?id=STqaMqhtDi).

We provide code and data to reproduce the results in the paper.

## Requirements

The code has been tested on CentOS 7 with Python 3.9 and Tensorflow 2.7.0.
Running CORP requires a GPU with at least 12GB of memory, and at least 40GB of main memory.

## Usage

1. Install the following packages first:
    - Install [NeuralDecoder](./NeuralDecoder/)
    - Install [LanguageModelDecoder](./LanguageModelDecoder/)
    - Install [OnlineRecalibrator](./OnlineRecalibrator/)
2. Download the [data]()
3. See the README in [OnlineRecalibrator](./OnlineRecalibrator/README.md) for how to run CORP.

## Citation
