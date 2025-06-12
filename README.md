# Applied Machine Learning for Trading: EURUSD Forecasting with Time Series Transformer
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97_Transformers-black?logo=hugging-face)](https://huggingface.co/docs/transformers/index)

## Short Description
This project develops and fine-tunes a **TimeSeriesTransformer** model to forecast EURUSD 5-minute closing prices. It serves as a modern, attention-based counterpart to the baseline LSTM model developed in a separate [repository](https://github.com/ilahuerta-IA/applied-ml-trading-lstm-eurusd). The primary goal is to create a robust predictive "tool" that can generate signals for an algorithmic trading strategy and to compare its performance against a traditional recurrent architecture.

## Project Objective
The objective is to build, evaluate, and systematically optimize a TimeSeriesTransformer model for short-term currency exchange rate prediction. This project documents the experimentation process required to fine-tune the model on a large financial dataset and provides a clear performance baseline that can be directly compared to other architectures like LSTMs.

## Dataset
*   **Asset:** EURUSD (Euro / US Dollar)
*   **Frequency:** 5-minute intervals
*   **Period:** 10 years
*   **Source:** `EURUSD_5m_10Yea.csv` (Included in the repository)
*   **Columns used:** `Timestamp` (derived), `Close`

## Features
*   Data loading and time series indexing with Pandas.
*   Feature engineering of time-based positional encodings.
*   Chronological data splitting into Train, Validation, and Test sets (60/20/20).
*   Standard scaling of target data using Scikit-learn.
*   Custom PyTorch `Dataset` and `DataLoader` for efficient batching.
*   TimeSeriesTransformer model construction using Hugging Face `transformers`.
*   Manual PyTorch training loop with integrated validation and early stopping.
*   Model evaluation using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
*   Systematic hyperparameter tuning documented in `EXPERIMENTS.md`.

## Technologies Used
*   Python 3.9+
*   PyTorch
*   Hugging Face Transformers & Accelerate
*   Pandas & NumPy
*   Scikit-learn
*   Matplotlib
*   TQDM

## Setup and Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/applied-ml-trading-transformer-eurusd.git
    cd applied-ml-trading-transformer-eurusd
    ```
2.  (Recommended) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    _`requirements.txt` should contain:_
    ```
    pandas
    numpy
    matplotlib
    torch
    transformers
    accelerate
    scikit-learn
    tqdm
    jupyter
    ```

## Usage
The entire development and training process is contained within the Jupyter Notebook: `TimeSeries_Transformer_EURUSD_Forecasting.ipynb`.

1.  Ensure the dataset (`EURUSD_5m_10Yea.csv`) is in the root directory.
2.  Open and run the notebook. It is recommended to use an environment with a GPU for faster training.
3.  Modify the hyperparameters in **Cell 2** of the notebook to run new experiments. The notebook will automatically train the model, perform validation-based early stopping, and save the best model as `best_model.pth`.
4.  The final cell will load the best model and evaluate its performance on the unseen test set, printing the final MAE and RMSE metrics.

## Model Architecture & Hyperparameter Tuning
The TimeSeriesTransformer is an encoder-decoder architecture that uses self-attention mechanisms to capture temporal dependencies. Unlike LSTMs, it does not process data sequentially, allowing it to identify relationships between distant time steps more effectively.

Key hyperparameters that were systematically tuned include:
*   **`CONTEXT_LENGTH` (Lookback Window):** Defines how many past 5-minute bars the encoder uses.
*   **`D_MODEL` (Model Dimensionality):** The size of the model's hidden layers.
*   **`*_LAYERS` & `*_HEADS`:** The depth and complexity of the attention mechanism.
*   **`DROPOUT`:** The rate of dropout for regularization.
*   **`LEARNING_RATE` & `BATCH_SIZE`:** Critical parameters for the training process.

Detailed experimental results and the process for selecting the optimal configuration are documented in **[EXPERIMENTS.md](EXPERIMENTS.md)**.

## Performance Comparison vs. LSTM
This project's primary value is in its direct comparison to a well-optimized LSTM model.

| Model                   | Test Set MAE (EURUSD) | Test Set MAE (Pips) |
| ----------------------- | --------------------- | ------------------- |
| Optimized LSTM (V1.0)   | 0.000237              | 2.37                |
| **TimeSeriesTransformer** | *[Fill]* | *[Fill]* |

*(Note: The LSTM result is sourced from the [reference repository](https://github.com/ilahuerta-IA/applied-ml-trading-lstm-eurusd). The Transformer result should be filled in after running the final experiment.)*

## Contributing
Contributions, issues, and feature requests are welcome. Please feel free to fork the repository, make changes, and open a pull request.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
