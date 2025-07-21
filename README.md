# Applied Machine Learning for Trading: EURUSD Forecasting with Time Series Transformer
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97_Transformers-black?logo=hugging-face)](https://huggingface.co/docs/transformers/index)

## Short Description
This project develops and fine-tunes a **TimeSeriesTransformer** model to forecast EURUSD 5-minute closing prices. It serves as a modern, attention-based counterpart to a baseline LSTM model developed in a separate [repository](https://github.com/ilahuerta-IA/applied-ml-trading-lstm-eurusd). The primary goal is to create a robust predictive "tool" that can generate signals for an algorithmic trading strategy and to compare its performance against a traditional recurrent architecture.

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
*   A complete PyTorch training loop with integrated validation and early stopping to prevent overfitting.
*   Model evaluation using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
*   Systematic hyperparameter tuning documented in `EXPERIMENTS.md`.
*   Export of final model, scaler, and configuration for deployment.

## Technologies Used
*   Python 3.9+
*   PyTorch
*   Hugging Face Transformers & Accelerate
*   Pandas & NumPy
*   Scikit-learn
*   Matplotlib
*   TQDM
*   Joblib

## Setup and Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/ilahuerta-IA/applied-ml-trading-transformer-eurusd.git
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
    _The `requirements.txt` file should contain the packages listed under "Technologies Used"._

## Usage
The entire development and training process is contained within the Jupyter Notebook: **`TimeSeries_Transformer_EURUSD_Forecasting.ipynb`**.

1.  Ensure the dataset (`EURUSD_5m_10Yea.csv`) is in the root directory.
2.  Open and run the notebook. It is recommended to use an environment with a GPU for faster training.
3.  Modify the hyperparameters in **Cell 2** of the notebook to run new experiments. The notebook will automatically train the model, perform validation-based early stopping, and save the best model artifacts.
4.  The final cells will load the best model and evaluate its performance, printing the final metrics and generating visualizations.

## Model Architecture & Hyperparameter Tuning
The TimeSeriesTransformer is an encoder-decoder architecture that uses self-attention mechanisms to capture temporal dependencies. This project undertakes a rigorous, multi-stage hyperparameter tuning process to find the optimal configuration.

Key hyperparameters systematically tested include:
*   `CONTEXT_LENGTH` (Lookback Window)
*   `D_MODEL`, `*_LAYERS`, & `*_HEADS` (Model Complexity)
*   `DROPOUT` (Regularization)
*   `LEARNING_RATE` & `BATCH_SIZE` (Training Dynamics)

Detailed experimental results and the process for selecting the final optimal configuration are documented in **[EXPERIMENTS.md](EXPERIMENTS.md)**.

## Performance Comparison vs. LSTM
This project's primary value is in its direct comparison to a well-optimized LSTM model from a parallel research effort. After comprehensive tuning, the Transformer model demonstrates superior predictive accuracy and robustness.

| Model                               | Test Set MAE (EURUSD) | Test Set MAE (Pips) | Test Set RMSE (EURUSD) |
| :---------------------------------- | :-------------------- | :------------------ | :--------------------- |
| Optimized LSTM (V1.0)               | 0.000237              | 2.37                | 0.000417               |
| **TimeSeriesTransformer (Best Run)** | **0.000207**          | **2.07**            | **0.000320**           |

*(Note: The LSTM result is sourced from the [reference repository](https://github.com/ilahuerta-IA/applied-ml-trading-lstm-eurusd). Multiple runs of the Transformer yielded a consistent Test MAE between 0.000207 and 0.000229, demonstrating model stability.)*

### Final Conclusion
The final optimized TimeSeriesTransformer model achieved a **best Test Set Mean Absolute Error of 0.000207 (~2.07 pips)**.

This result is superior to the LSTM baseline in two key ways:
1.  **Higher Accuracy:** The MAE is lower, indicating a better average prediction.
2.  **Greater Robustness:** The RMSE is significantly lower (by ~23%), indicating the Transformer makes fewer large, erroneous predictions, which is critical for risk management in a live trading environment.

This project successfully demonstrates that for this large, high-frequency financial dataset, the modern attention-based Transformer architecture provides a measurable performance edge over a traditional, highly-optimized recurrent neural network.

---

## Exported Model Artifacts & Backtrader Integration
Upon successful execution of the notebook, a complete package for deployment is saved in the `Models/` directory.

*   `best_transformer_model.pth`: The trained model's state dictionary (the weights).
*   `target_scaler.pkl`: The fitted `StandardScaler` object for data normalization.
*   `model_config.json`: A JSON file containing the model's architectural hyperparameters.

These three files are all that is needed to use the model in an external application like a `backtrader` script.

### ***A Note on Versioning and Reproducibility***
**IMPORTANT:** The saved artifacts, particularly the `target_scaler.pkl` file, are dependent on the library versions used during training (e.g., `scikit-learn==1.6.1`). Loading these artifacts in an environment with a different version (e.g., `1.7.1`) may produce an `InconsistentVersionWarning` and could lead to breaking code or invalid results.

For maximum reliability in a production or backtesting environment, it is **highly recommended to re-run the `TimeSeries_Transformer_EURUSD_Forecasting.ipynb` notebook to retrain the model and generate fresh artifacts** using the exact library versions installed in your target environment. The final, optimized hyperparameters are already set in Cell 2 of the notebook for this purpose.

### Conceptual Guide for `backtrader` Integration
Integrating this model into a `backtrader` strategy involves creating a custom `Indicator` or directly calling a prediction function within your `Strategy` class's `next()` method.

**Prediction Workflow:**
1.  **Load Artifacts:** In your script's `__init__`, load the model config, build the model architecture, load the weights from the `.pth` file, and load the scaler from the `.pkl` file.
2.  **Gather Data:** In the `next` method of your strategy, access the last `CONTEXT_LENGTH` + `max(lags_sequence)` closing prices from your `backtrader` data feed.
3.  **Preprocess:**
    *   Create the required time features (hour, day of week, etc.) from the timestamps.
    *   Use the loaded `target_scaler` to `.transform()` the prices into their scaled representation.
4.  **Predict:** Feed the scaled prices and time features into your loaded model to get a scaled prediction for the next bar.
5.  **Post-process:** Use the loaded `target_scaler` to `.inverse_transform()` the model's output back into a real, understandable EURUSD price prediction.
6.  **Generate Signal:** Use this final prediction to generate a trading signal.

A separate repository dedicated to the backtesting and strategy implementation using these artifacts is the recommended next step.

---
## Contributing
Contributions, issues, and feature requests are welcome. Please feel free to fork the repository, make changes, and open a pull request.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.