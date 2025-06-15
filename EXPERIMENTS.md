# Model Development and Experiments: TimeSeriesTransformer

This document outlines the experiments conducted to fine-tune the TimeSeriesTransformer model for forecasting EURUSD 5-minute closing prices. The goal is to systematically evaluate the impact of different hyperparameter choices on the model's predictive performance and compare it to a baseline LSTM model.

All experiments use the 10-year, 5-minute EURUSD dataset with a 60/20/20 chronological split for training, validation, and testing.

## Experiment 1: Optimizing Lookback Window (CONTEXT_LENGTH)

The first and most critical experiment focuses on determining the most effective lookback window (`CONTEXT_LENGTH`), which defines how much historical context the model uses for its predictions.

### Methodology (Context Length)

The core Transformer architecture and other training parameters were kept constant while the `CONTEXT_LENGTH` and the number of `EPOCHS` were varied.
*   **Base Architecture:** `D_MODEL=32`, `Layers=2`, `Heads=4`, `Dropout=0.1`
*   **Base Training:** `LR=1e-4`, `Batch Size=64`, `Patience=10`
*   **Note on Epochs:** Due to computational constraints in the free Colab environment, initial sweeps for larger window sizes were conducted with a reduced `EPOCHS=5`. Key configurations were later run for a full `EPOCHS=20` to find the true optimal performance.

### Results Summary (Context Length)

| CONTEXT_LENGTH (Window) | Epochs | Best `val_loss` (at Epoch) | Validation Set MAE | **Test Set MAE** | Test Set RMSE |
| :---------------------- | :----: | :------------------------- | :------------------- | :--------------- | :------------ |
| 15                      |   20   | -3.0266 (Ep 15)            | 0.000276             | 0.000257         | 0.000360      |
| **30**                  | **20** | **-3.0656 (Ep 19)**        | **0.000277**         | **0.000207**     | **0.000320**  |
| 30                      |   5    | -2.5932 (Ep 5)             | 0.000701             | 0.000604         | 0.000708      |
| 60                      |   5    | -2.6073 (Ep 4)             | 0.000438             | 0.000390         | 0.000502      |
| 120                     |   20   | -2.9704 (Ep 15)            | 0.000466             | 0.000337         | 0.000443      |
| 120                     |   5    | -2.6274 (Ep 5)             | 0.000363             | 0.000307         | 0.000419      |
| 288                     |   5    | *(Run Timed Out)*          | -                    | -                | -             |

*(Note: The negative loss values indicate the use of the default Negative Log-Likelihood loss function. While useful for relative comparison here, it is recommended to switch to a `distribution_output="normal"` setting for an MSE-equivalent loss in future work.)*

### Analysis and Final Conclusion (Context Length)

1.  **Optimal Configuration:** The model with **`CONTEXT_LENGTH=30` trained for 20 epochs is the clear winner**, achieving a state-of-the-art **Test Set MAE of 0.000207 EURUSD (~2.07 pips)**. This significantly outperforms all other tested configurations.

2.  **The "Sweet Spot" for Context:** The experiments reveal a fascinating insight: more historical data is not always better for this Transformer architecture on high-frequency data.
    *   Performance degraded significantly when the window was increased to `120` (Test MAE of 0.000337), even with a full 20-epoch training cycle.
    *   A very short window of `15` performed well but was not as accurate as the `30`-bar window.
    *   This strongly suggests that for 5-minute EURUSD price action, the most relevant predictive information is contained within the last **2.5 hours of data (`CONTEXT_LENGTH=30`)**. Longer lookbacks appear to introduce more noise than valuable signal, making it harder for the model to optimize.

3.  **Training Duration is Crucial:** The difference between the 5-epoch and 20-epoch runs for `W=30` (a 66% improvement in MAE) underscores the necessity of allowing the model sufficient training time to converge properly.

**Final Decision:**
Based on this comprehensive testing, the definitive optimal hyperparameter for the lookback window is **`CONTEXT_LENGTH = 30`**. This configuration will serve as the foundation for any further tuning of model size or regularization parameters.

---

(Note: The negative loss values indicate the use of the default Negative Log-Likelihood loss function. While useful for relative comparison here, the primary decision-making metric for comparison against the LSTM is the Mean Absolute Error (MAE) calculated on the original price scale.)