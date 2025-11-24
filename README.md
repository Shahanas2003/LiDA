
# LiDA: Language-Independent Data Augmentation for Text Classification
This repository contains the source code and datasets for LiDA, a framework for augmenting text classification data in any language by working at the sentence embedding level.
## Overview
LiDA generates synthetic training data from sentence embeddings using linear noise, autoencoders, and denoising autoencoders. It is designed for low-resource languages (e.g., Bengali) and supports multilingual experiments with English, Chinese, Indonesian, etc.
## Requirements
- Python 3.8+
- PyTorch
- sentence-transformers
- numpy, pandas
- matplotlib, scikit-learn
Install all dependencies:

```
pip install -r requirements.txt

```
## Datasets
- Included: English, Chinese, Indonesian, Bengali sentiment datasets.
- All datasets are in CSV format with sentence and label columns.
- For details and format, see `data/README.md` or explore the `/data/` directory.
## Usage
### Training Baseline Models
#### LSTM

# Usage

## Training Baseline models

### LSTM

```
python main.py
```

### SBERT

```
cd sbert
python main.py
```

## Run the experiment

### LSTM
Open the experiment.py file and adjust the parameters before running it.
```
python experiment.py
```
## Results
Example classification accuracy improvements (Bengali, 10% data):
| Model         | Without LiDA | With LiDA |
|---------------|--------------|-----------|
| LSTM Baseline | 0.63         | 0.71      |
Sample plot (see `/results/plots/`):
![Bengali Sentiment Classification Accuracy With/Without LiDA Augmentation](results/plots/bengali_accuracy_vs_dataset_size.png)

## License
This project is licensed under the MIT License. See `LICENSE` for details.
## Citation
If you use LiDA in your research, please cite:


### SBERT
Go to sbert folder, open the experiment.py file and adjust the parameters before running it.
```
python experiment.py
```
