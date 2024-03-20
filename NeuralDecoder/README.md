# Neural Sequence Decoder

This repo is for the RNN decoder.

## Requirements

Python >= 3.9

## Installation

Install the package with the following command:
```
pip install -e .
```

## Train seed model

```bash
cd examples
./train_seed_model.sh  # See comments in the script for how to locate your downloaded data.
```

## A new baseline model for [Willett Nature 2021](https://www.nature.com/articles/s41586-021-03506-2)

Raw character error rate (CER) on heldout blocks:
|                    |   GRU       | Bidirectional-GRU |
|--------------------|-------------|-------------------|
|   CER   | 3.3% ± 0.1% |    3.2% ± 0.1%    |

To train the model:
```bash
cd examples
./train_willett_nature_baseline.sh
```

Pre-made tfrecords can be downloaded [here](https://office365stanford-my.sharepoint.com/:u:/g/personal/stfan_stanford_edu/ESITughFEPhFu3hnfdFQvxoBFApZD0EP_Z3zQlQKUrj6Qg?e=bBS6SA).
