# Comparative Analysis of RNN Architectures for Sentiment Classification

This project implements and evaluates multiple Recurrent Neural Network (RNN) architectures for sentiment classification on the IMDb Movie Review Dataset.

## Project Overview

This project systematically compares different RNN configurations for binary sentiment classification (positive/negative) on movie reviews. The following variations are tested:

- **Architectures**: RNN, LSTM, Bidirectional LSTM
- **Activation Functions**: Sigmoid, ReLU, Tanh
- **Optimizers**: Adam, SGD, RMSProp
- **Sequence Lengths**: 25, 50, 100 words
- **Stability Strategies**: With and without gradient clipping

## Project Structure

```
.
├── data/
│   └── IMDB Dataset.csv
├── src/
│   ├── preprocess.py       # Data preprocessing and tokenization
│   ├── models.py            # RNN model architectures
│   ├── train.py             # Training loop and optimization
│   ├── evaluate.py          # Model evaluation and metrics
│   ├── utils.py             # Utility functions (reproducibility, timing)
│   └── run_experiments.py   # Main experiment runner
├── results/
│   ├── metrics.csv          # Summary of all experiments
│   ├── detailed_results.json # Detailed results including training history
│   ├── vocabulary.pkl       # Saved vocabulary
│   └── plots/               # Generated visualizations
├── requirements.txt
├── README.md
└── problem statement.pdf
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Setup Instructions

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the dataset is in place**:
   - The IMDb dataset should be located at `data/IMDB Dataset.csv`
   - If not present, download from [Kaggle IMDb Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Usage

### Running All Experiments

To run the complete set of experiments:

```bash
cd src
python run_experiments.py
```

This will:
- Load and preprocess the IMDb dataset
- Run all experiment configurations systematically
- Save results to `results/metrics.csv` and `results/detailed_results.json`
- Generate all required plots in `results/plots/`

**Expected Runtime**:
- With GPU: ~2-3 hours for all experiments
- With CPU: ~8-12 hours for all experiments

### Running Individual Components

**Test preprocessing**:
```bash
cd src
python preprocess.py
```

**Test model creation**:
```bash
cd src
python models.py
```

**Train a single model**:
```bash
cd src
python train.py
```

**Evaluate a model**:
```bash
cd src
python evaluate.py
```

## Dataset Information

- **Dataset**: IMDb Movie Review Dataset
- **Total Samples**: 50,000 reviews
- **Split**: 25,000 training, 25,000 testing (50/50 split)
- **Classes**: Binary (Positive/Negative)
- **Preprocessing**:
  - Lowercase conversion
  - Removal of punctuation and special characters
  - Tokenization
  - Vocabulary limited to top 10,000 words
  - Sequence padding/truncating to lengths: 25, 50, 100

## Model Configuration

All models use the following base configuration:
- **Embedding dimension**: 100
- **Hidden dimension**: 64
- **Number of layers**: 2
- **Dropout**: 0.4
- **Batch size**: 32
- **Epochs**: 10
- **Loss function**: Binary Cross-Entropy
- **Output activation**: Sigmoid

## Experiments

The project runs the following systematic experiments:

1. **Architecture Comparison**: RNN vs. LSTM vs. Bidirectional LSTM
2. **Activation Function Comparison**: Sigmoid vs. ReLU vs. Tanh
3. **Optimizer Comparison**: Adam vs. SGD vs. RMSProp
4. **Sequence Length Comparison**: 25 vs. 50 vs. 100 words
5. **Gradient Clipping**: With vs. without gradient clipping

Each experiment varies one factor while keeping others fixed to isolate effects.

## Output Files

After running experiments, the following files are generated:

- `results/metrics.csv`: Summary table with all experiment results
- `results/detailed_results.json`: Complete results including training history
- `results/plots/accuracy_f1_vs_seqlen.png`: Performance vs. sequence length
- `results/plots/loss_best_worst_models.png`: Training curves for best/worst models
- `results/plots/comparison_by_experiment_type.png`: Grouped comparisons

## Evaluation Metrics

Models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **F1-Score (Macro)**: Macro-averaged F1 score
- **Training Time**: Average time per epoch (seconds)

## Reproducibility

Random seeds are fixed for reproducibility:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

Hardware information is automatically logged and saved with results.

## Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only
- **Recommended**: 16GB RAM, CUDA-compatible GPU
- **Storage**: ~500MB for dataset + models

The code automatically detects and uses GPU if available via CUDA.

## Results

After running all experiments, check:
1. `results/metrics.csv` for the summary table
2. `results/plots/` for all visualizations
3. Console output for detailed training logs

## Troubleshooting

**Out of Memory Error**:
- Reduce batch size in `run_experiments.py`
- Use shorter sequence lengths
- Reduce model hidden dimension

**Slow Training**:
- Ensure GPU is available and detected
- Reduce number of epochs for testing
- Use smaller dataset subset for debugging

**Missing Data File**:
- Ensure `data/IMDB Dataset.csv` exists
- Check file path and spelling

## Citation

If using this code, please cite the IMDb dataset:
```
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  year      = {2011}
}
```

## License

This project is for educational purposes as part of DATA641-NLP coursework.

## Contact

For questions or issues, please refer to the course materials or contact the instructor.
