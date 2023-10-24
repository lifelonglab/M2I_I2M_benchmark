# From MNIST to ImageNet and Back: Benchmarking Continual Curriculum Learning

# Introduction
This repository contains a code for carrying out experiments for the paper ''From MNIST to ImageNet and Back: Benchmarking Continual Curriculum Learning''.
 The details of the algorithm are described here (preprint): https://arxiv.org/abs/2303.11076


# Environment 
1. The project was developed and tested leveraging Python 3.8
2. All python dependencies are listed in `requirements.txt` file. You can install them using `pip install -r requirements.txt`.
3. The project was tested on NVIDIA GPUs, including NVIDIA A100.


# How to run experiments?

1. The main file that can be used to run the experiments is `experiment.py`. 
2. The experiments can be configured using the configuration files in the `config` folder.
3. The configuration files are in the YAML format. There is a separate configuration file for each strategy.
4. You can run the experiments using the following command: `python3 experiment.py --config config/class_incremental/config_file_name.yml`
5. The results will be put in `logs/out` directory.

# The structure of the repository
- `config` - folder with configuration files for experiments
- `models` - folder with code for models
- `scenarios` - folder with code for scenarios and datasets
- `strategies` - folder with code for strategies

