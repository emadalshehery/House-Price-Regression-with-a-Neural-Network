# House Price Regression with a Neural Network

**Notebook:** `day1\_1.ipynb`

Builds and trains an MLP in PyTorch to predict house prices from the Kaggle **House Prices** dataset.

## Pipeline

1. **Load** `yasserh/housing-prices-dataset` with `kagglehub`.
2. **Encode** text columns with `LabelEncoder`.
3. **Filter outliers**: keep `price <= 7,500,000` and `area <= 8000` (545 → 456 houses).
4. **Split** 80/20, convert to float32 tensors, and build DataLoaders (batch size 32, 12 training batches).

## Model: `HousePricePredictor`

Fully connected network with ReLU activations and a single linear output neuron (the price).

```
Linear(in → h1) → Linear(h1 → h2) → Linear(h2 → h2) → Linear(h2 → h1) → Linear(h1 → 1)
```

## Training Setup

|Setting|Value|
|-|-|
|Hidden dims|64 and 32|
|Loss|`nn.L1Loss` (MAE)|
|Optimizer|Adam, lr = 0.001|
|Epochs|100|
|Batch size|32|

Helper functions: `train\_one\_epoch` (forward, loss, `zero\_grad`, backward, step) and `validate\_one\_epoch` (`model.eval()` + `torch.no\_grad()`, reports loss and MAE).

## Results

Validation MAE fell from about $4.35M (epoch 1) to about **$0.91M** and plateaued from around epoch 15.

## Visualizations

* Training vs validation loss curves and validation MAE curve.
* Predictions plot: actual prices (dots) vs the model line over `area` (other features held at 0).

## Notes

* Features are **not scaled** in this notebook (unlike `day1\_lab1.ipynb`), which likely explains the high error. The prediction plot assumes scaled inputs (0 = average).
* The markdown mentions MSE and 50 epochs, but the code uses L1 loss and 100 epochs.

## Requirements

```bash
pip install torch pandas scikit-learn kagglehub matplotlib tqdm
```

