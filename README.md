# TrajGPT: Controlled Synthetic Trajectory Generation Using a Multitask Transformer-Based Spatiotemporal Model
(We will update this README with details of TrajGPT once the paper is publicly available.)

## Requirements
Recommended setup: GPU with more than 4GB VRAM and CUDA version 12.3.

## Installation
Use conda to create a virtual environment and pip to install the requirements. 
```bash
conda create --name trajgpt python==3.10.13
conda activate trajgpt
pip install -r requirements.txt
```

## Usage
To reproduce the results of TrajGPT on GeoLife, please follow these instructions: 

1. Enter the root directory of TrajGPT.
```bash
cd your/path/to/TrajGPT
```

2. Download dataset from [GeoLife GPS Trajectories](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/).
```bash
wget -O data/geolife.zip https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip
```

3. Unzip the data to the `data` folder.
```bash
unzip data/geolife.zip -d data
```

3. Convert point-by-point trajectories to sequences of visits (a.k.a. staypoints).
```bash
python3 utils/preprocess.py
```
This step takes around 5 minutes and only needs to be run once.

4. Run `main.py`. It took around 10 minutes on A100, or reaching `patience` after hundreds of epochs.
Please specify one of `--next_prediction` or `--infilling` as the argument.
```bash
python3 main.py --next_prediction
```
or
```bash
python3 main.py --infilling
```

## Authors
This software is produced by [Shang-Ling (Kate) Hsu](https://ktxlh.github.io/), the first author of TrajGPT. The subsequent authors of TrajGPT are: Emmanuel Tung, Dr. John Krumm, Dr. Cyrus Shahabi, and Dr. Khurram Shafique.

## Support
We are committed to open-sourcing this project and ensuring that everyone can reproduce the results presented in our paper. If you have any questions, please feel free to open an issue.

## License

[MIT](https://choosealicense.com/licenses/mit/)
