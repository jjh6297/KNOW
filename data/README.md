
# Meta-dataset for "Learning from Oblivion: Predicting Knowledge-Overflowed Weights via Retrodiction of Forgetting"

## 0. Data Download
The dataset can be downloaded from https://drive.google.com/drive/folders/1J9BWTdn1xrNHuLSpTMzdslKDMlkMDKQu?usp=sharing

## 1. Basic Python Loading Code

The generated `.mat` files are saved in the HDF5 format (MATLAB `-v7.3` compatible). You can load them easily using the `h5py` library in Python.

```python
import h5py

# Define the path to your dataset file
file_path = 'File_Directory'

# Load the dataset
with h5py.File(file_path, 'r') as f:
    # Load datasets into memory and ensure float32 precision
    conv_weights = f['ConvWeights'][:].astype('float32')
    fc_weights = f['FCWeights'][:].astype('float32')
    bias = f['Bias'][:].astype('float32')

# Print shapes to verify dimensions
print(f"ConvWeights Shape: {conv_weights.shape}")
print(f"FCWeights Shape: {fc_weights.shape}")
print(f"Bias Shape: {bias.shape}")
```

## 2. Variables and Dimensions

The dataset organizes all model weights and biases into three distinct variables. To accommodate both CNNs and ViTs, 3x3 kernels are separated from the rest of the flattened weights.

* **`ConvWeights`**: `(3, 3, N, 7)`
  * **[Height, Width, Total_Channels, Session]**
  * Contains 3x3 convolution kernel weights. `N` is the aggregated number of channels across all 3x3 layers in the model.
* **`FCWeights`**: `(M, 7)`
  * **[Flattened_Weights, Session]**
  * Contains all weights that are not 3x3 kernels (e.g., Fully Connected layers, ViT patch embeddings). These are flattened into a 1D array of size `M`.
* **`Bias`**: `(K, 7)`
  * **[Flattened_Bias, Session]**
  * Contains the flattened bias values from all layers in the model, totaling `K` elements.

## 3. Explanation of the 7-Dimension

The last dimension of every variable (of size 7) represents the weights collected by our progressive finetuning with intentional forgetting. Therefore, **it is possible to predict higher-indexed weights from lower-indexed weights as we aimed.**

* **Closer to Index 0**: Represents weights finetuned with the **smallest dataset**.
* **Index 6 (The 7th Session)**: Represents the final weights trained with **100% of the dataset**.

*(Note: Since Python uses 0-based indexing, the 7 dimensions are accessed using indices `0` through `6`.)*

## 4. Hyperparameters in the Filename

Important experimental hyperparameters are not stored inside the `.mat` file but are explicitly indicated in the **filename** itself. You can parse the filename to extract the following conditions:
* **Architecture and Dataset**: The architecture and trained dataset (e.g., `ViT_Small_CIFAR10...`)
* **SamplingRate**: The downsampling rate of the dataset (e.g., `SamplingRate_0.344...`)
* **LR**: Learning Rate used during training (e.g., `LR_0.000898...`)
* **BS**: Batch Size used during training (e.g., `BS_33`)