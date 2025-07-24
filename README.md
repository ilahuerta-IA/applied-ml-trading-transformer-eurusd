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
*   **Source:** `EURUSD_5m_10Yea.csv`
*   **Columns used:** `Timestamp` (derived), `Close`

## Features
*   Systematic hyperparameter tuning documented in `EXPERIMENTS.md`.
*   A definitive training script (`Train_Transformer_EURUSD_Model.ipynb`) that uses the optimal hyperparameters.
*   A robust PyTorch training loop with validation-based early stopping to prevent overfitting.
*   Export of final, synchronized model artifacts (`.pth`, `.pkl`, `.json`) ready for deployment.

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

## Usage and Model Training
The final, optimized training process is contained within the Jupyter Notebook: **`Train_Transformer_EURUSD_Model.ipynb`**.

This notebook is pre-configured with the best hyperparameters found during the research phase (documented in `EXPERIMENTS.md`). To generate the final, deployable set of model artifacts, simply run all cells in the notebook. Upon completion, a new, perfectly synchronized set of deployment artifacts will be saved into the `Models/` directory.

---

## Performance Comparison vs. LSTM
This project's primary value is in its direct comparison to a well-optimized LSTM model from a parallel research effort. After comprehensive tuning and a definitive retraining run with a valid configuration, the final TimeSeriesTransformer model demonstrates superior predictive accuracy and robustness.

| Model                               | Test Set MAE (EURUSD) | Test Set MAE (Pips) | Test Set RMSE (EURUSD) |
| :---------------------------------- | :-------------------- | :------------------ | :--------------------- |
| Optimized LSTM (V1.0)               | 0.000237              | 2.37                | 0.000417               |
| **TimeSeriesTransformer (Final)**   | **0.000203**          | **2.03**            | **0.000318**           |

*(Note: The LSTM result is sourced from the [reference repository](https://github.com/ilahuerta-IA/applied-ml-trading-lstm-eurusd).)*

### Final Conclusion
The final optimized TimeSeriesTransformer model achieved a **Test Set Mean Absolute Error of 0.000203 (~2.03 pips)**.

This result is superior to the LSTM baseline in two key ways:
1.  **Higher Accuracy:** The MAE is **14% lower** than the LSTM's, indicating a more precise average prediction.
2.  **Greater Robustness:** The RMSE is **24% lower**, indicating the Transformer makes significantly fewer large, erroneous predictions, which is critical for risk management in a live trading environment.

This project successfully demonstrates that for this large, high-frequency financial dataset, the modern attention-based Transformer architecture provides a measurable performance edge over a traditional, highly-optimized recurrent neural network.

---

## Exported Model Artifacts & Deployment Notes

After running the final training notebook, a complete and synchronized package for deployment is saved in the `Models/` directory.

*   `best_transformer_model.pth`: The trained model's state dictionary (the weights).
*   `target_scaler.pkl`: The fitted `StandardScaler` object for data normalization.
*   `model_config.json`: A JSON file containing the model's architectural hyperparameters.

### Critical Note on `CONTEXT_LENGTH` and `lags_sequence`
A key architectural requirement of the Hugging Face `TimeSeriesTransformer` is that the `CONTEXT_LENGTH` (the main lookback window) **must be greater than the maximum value in the `lags_sequence`**.

During initial experiments, a `ValueError` was encountered because a configuration with `CONTEXT_LENGTH=30` was incompatible with a `lags_sequence` containing a lag of `21`. The final, optimal model was retrained with a valid configuration (`CONTEXT_LENGTH=30` and `lags_sequence=[1, 2, 3, 4, 5, 6, 7]`), which resolved this issue. This is a critical check for anyone adapting this model.

### A Note on Versioning and Reproducibility
To avoid potential library versioning conflicts (e.g., `scikit-learn InconsistentVersionWarning`) or model configuration mismatches, the recommended approach for deployment is to **re-run the `Train_Transformer_EURUSD_Model.ipynb` notebook** in your target environment. This generates a fresh set of artifacts (`.pth`, `.pkl`, and `.json`) that are guaranteed to be compatible.

---
## Contributing
Contributions, issues, and feature requests are welcome. Please feel free to fork the repository, make changes, and open a pull request.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.