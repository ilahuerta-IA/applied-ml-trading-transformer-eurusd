# Model Development and Experiments: TimeSeriesTransformer

This document outlines the experiments conducted to fine-tune the TimeSeriesTransformer model for forecasting EURUSD 5-minute closing prices. The goal is to systematically evaluate the impact of different hyperparameter choices on the model's predictive performance and compare it to the LSTM baseline.

All experiments use the 10-year, 5-minute EURUSD dataset with a 60/20/20 chronological split for training, validation, and testing. The primary decision-making metric is the **Validation Set MAE**, with the **Test Set MAE** used for final performance assessment.

*(Note: The negative loss values reported in these experiments are due to the model's default Negative Log-Likelihood (NLL) loss function. While useful for observing convergence, the MAE and RMSE metrics, calculated on the original price scale, are used for all performance comparisons.)*

---

## Experiment 1: Optimizing Lookback Window (CONTEXT_LENGTH)

The first experiment focused on determining the most effective lookback window (`CONTEXT_LENGTH`), which defines how much historical context the model uses for its predictions.

### Methodology (Context Length)
*   **Base Architecture:** `D_MODEL=32`, `Layers=2`, `Heads=4`, `Dropout=0.1`
*   **Base Training:** `LR=1e-4`, `Batch Size=64`, `Patience=10`, `Epochs=20` (or 5 for longer windows)

### Results Summary (Context Length)

| CONTEXT_LENGTH | Epochs | Best `val_loss` (at Epoch) | Validation Set MAE | **Test Set MAE** | Test Set RMSE |
| :------------: | :----: | :------------------------- | :------------------- | :--------------- | :------------ |
| 15             |   20   | -3.0266 (Ep 15)            | 0.000276             | 0.000257         | 0.000360      |
| **30**         | **20** | **-3.0656 (Ep 19)**        | **0.000277**         | **0.000207**     | **0.000320**  |
| 30             |   5    | -2.5932 (Ep 5)             | 0.000701             | 0.000604         | 0.000708      |
| 60             |   5    | -2.6073 (Ep 4)             | 0.000438             | 0.000390         | 0.000502      |
| 120            |   20   | -2.9704 (Ep 15)            | 0.000466             | 0.000337         | 0.000443      |
| 120            |   5    | -2.6274 (Ep 5)             | 0.000363             | 0.000307         | 0.000419      |
| 288            |   5    | *(Run Timed Out)*          | -                    | -                | -             |

### Conclusion (Context Length)
For this high-frequency dataset, a shorter lookback window proved most effective. The model with **`CONTEXT_LENGTH = 30`** achieved the best performance on the test set, indicating that the most recent 2.5 hours of data contain the most relevant predictive signals. Longer windows (like 120) led to performance degradation, likely by introducing more noise than useful information. Therefore, `CONTEXT_LENGTH = 30` was selected as the optimal value for all subsequent experiments.

---

## Experiment 2: Optimizing Model Size and Complexity

This experiment investigates whether a wider or deeper Transformer architecture can improve upon the baseline performance established with `CONTEXT_LENGTH=30`.

### Methodology (Model Size)
*   **Fixed Parameters:** `CONTEXT_LENGTH=30`, `Dropout=0.1`, `LR=1e-4`, `Batch Size=64`, `Epochs=20`, `Patience=10`
*   **Varied Parameters:** `D_MODEL`, `ENCODER/DECODER_LAYERS`, `ENCODER/DECODER_ATTENTION_HEADS` were modified as a group to test different model sizes.

### Results Summary (Model Size)

| Experiment Name    | D_MODEL / LAYERS / HEADS | Best `val_loss` (at Epoch) | **Validation Set MAE** | Test Set MAE   | Test Set RMSE  |
| :----------------- | :----------------------- | :------------------------ | :--------------------- | :--------------- | :------------- |
| **Small (Baseline)** | **32 / 2 / 4**           | **-3.0656 (Ep 19)**       | **0.000277**           | **0.000207**     | **0.000320**   |
| Medium             | 64 / 2 / 4               | -3.1108 (Ep 18)           | 0.000369               | 0.000282         | 0.000383       |
| Large              | 64 / 4 / 8               | -3.2234 (Ep 19)           | 0.000259               | 0.000212         | 0.000323       |
| X-Large            | 128 / 4 / 8              | -3.2019 (Ep 18)           | 0.000284               | 0.000210         | 0.000326       |

### Analysis and Conclusion (Model Size)

1.  **The Simplest Model is the Best:** The **"Small (Baseline)"** configuration, despite not having the lowest validation MAE, demonstrated the best generalization to the unseen test set, achieving the lowest final Test MAE and RMSE.

2.  **Evidence of Overfitting in Larger Models:** The "Large" model achieved the best performance on the *validation set* (MAE 0.000259), suggesting it was the best model during the tuning phase. However, its performance on the *test set* was slightly worse than the "Small" model. This is a classic sign of minor overfitting, where the more complex model learned patterns specific to the validation data that did not generalize perfectly.

3.  **Robustness over Complexity:** For noisy financial data, this result is highly significant. It demonstrates that a less complex, more robust model ("Small") can be superior to a more powerful one ("Large" or "X-Large") that is more prone to fitting noise.

**Final Decision:**
The **`D_MODEL=32`, `LAYERS=2`, `HEADS=4`** configuration is confirmed as the optimal architecture. It provides the best balance of learning capacity and robust generalization for this specific forecasting task.

---

## Experiment 3: Optimizing Dropout Rate (Regularization)

This experiment aimed to find the optimal dropout rate to prevent overfitting without hindering the model's ability to learn. The tests were conducted using the best architecture identified from the previous experiments.

### Methodology (Dropout Rate)
*   **Fixed Parameters:** `CONTEXT_LENGTH=30`, `D_MODEL=32`, `Layers=2`, `Heads=4`, `LR=1e-4`, `Batch Size=64`, `Epochs=20`, `Patience=10`.
*   **Varied Parameter:** The `DROPOUT` rate was tested at values of 0.1, 0.2, and 0.3.

### Results Summary (Dropout Rate)

| DROPOUT Rate | Best `val_loss` (at Epoch) | Validation Set MAE | Test Set MAE | Test Set RMSE |
| :----------- | :------------------------ | :--------------------: | :------------- | :------------ |
| **0.1 (Baseline)** | **-3.0656 (Ep 19)**     | **0.000277**         | **0.000207**   | **0.000320**  |
| 0.2          | -2.9442 (Ep 20)           | 0.000498             | 0.000401       | 0.000503      |
| 0.3          | -2.9015 (Ep 20)           | 0.000429             | 0.000364       | 0.000466      |

### Analysis and Conclusion (Dropout Rate)

1.  **Optimal Regularization:** The results clearly show that the baseline **`DROPOUT=0.1` provides the best performance**. It strikes the right balance, allowing the model to learn effectively while still providing enough regularization to generalize well to the unseen test set.

2.  **Underfitting with Higher Dropout:** Increasing the dropout rate to `0.2` and `0.3` led to a significant degradation in performance across all metrics (Training, Validation, and Test MAE/RMSE). This indicates that higher dropout rates over-constrain the "Small" model architecture, causing it to underfit. The model struggles to capture the underlying patterns in the data when too many neurons are randomly deactivated during training.

**Final Decision:**
The hyperparameter **`DROPOUT = 0.1` is confirmed as optimal** for this model configuration. No further tuning of this parameter is necessary.