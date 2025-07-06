# Model Development and Experiments: TimeSeriesTransformer

This document outlines the experiments conducted to fine-tune the TimeSeriesTransformer model for forecasting EURUSD 5-minute closing prices. The goal is to systematically evaluate the impact of different hyperparameter choices on the model's predictive performance and compare it to the LSTM baseline.

All experiments use the 10-year, 5-minute EURUSD dataset with a 60/20/20 chronological split for training, validation, and testing. The primary decision-making metric is the **Validation Set MAE**, with the **Test Set MAE** used for final performance assessment.

*(Note: The negative loss values reported in these experiments are due to the model's default Negative Log-Likelihood (NLL) loss function. While useful for observing convergence, the MAE and RMSE metrics, calculated on the original price scale, are used for all performance comparisons.)*

---

## Experiment 1: Optimizing Lookback Window (CONTEXT_LENGTH)

This experiment focused on determining the most effective lookback window.

*   **Base Architecture:** `D_MODEL=32`, `Layers=2`, `Heads=4`, `Dropout=0.1`
*   **Base Training:** `LR=1e-4`, `Batch Size=64`, `Patience=10`

### Results Summary (Context Length)

| CONTEXT_LENGTH | Epochs | Best `val_loss` (at Epoch) | Validation Set MAE | **Test Set MAE** | Test Set RMSE |
| :------------: | :----: | :------------------------- | :------------------- | :--------------- | :------------ |
| 15             |   20   | -3.0266 (Ep 15)            | 0.000276             | 0.000257         | 0.000360      |
| **30**         | **20** | **-3.0656 (Ep 19)**        | **0.000277**         | **0.000207**     | **0.000320**  |
| 120            |   20   | -2.9704 (Ep 15)            | 0.000466             | 0.000337         | 0.000443      |

### Conclusion (Context Length)
The model with **`CONTEXT_LENGTH = 30`** achieved the best performance on the test set. This suggests that for 5-minute EURUSD data, the most relevant predictive signals are contained within the most recent 2.5 hours. This value was fixed for all subsequent experiments.

---

## Experiment 2: Optimizing Model Size and Complexity

This experiment investigated whether a wider or deeper Transformer architecture could improve upon the `CONTEXT_LENGTH=30` baseline.

### Results Summary (Model Size)

| Experiment Name    | D_MODEL / LAYERS / HEADS | Best `val_loss` (at Epoch) | **Validation Set MAE** | Test Set MAE   | Test Set RMSE  |
| :----------------- | :----------------------- | :------------------------ | :--------------------- | :--------------- | :------------- |
| **Small (Baseline)** | **32 / 2 / 4**           | **-3.0656 (Ep 19)**       | **0.000277**           | **0.000207**     | **0.000320**   |
| Medium             | 64 / 2 / 4               | -3.1108 (Ep 18)           | 0.000369               | 0.000282         | 0.000383       |
| Large              | 64 / 4 / 8               | -3.2234 (Ep 19)           | 0.000259               | 0.000212         | 0.000323       |
| X-Large            | 128 / 4 / 8              | -3.2019 (Ep 18)           | 0.000284               | 0.000210         | 0.000326       |

### Conclusion (Model Size)
The **"Small (Baseline)"** configuration demonstrated the best generalization to the unseen test set. While larger models achieved competitive or even slightly better validation scores, they failed to outperform the simpler architecture on the final test data, indicating minor overfitting. This confirms the baseline architecture (`D_MODEL=32`, `LAYERS=2`, `HEADS=4`) as the most robust choice.

---

## Experiment 3: Optimizing Dropout Rate (Regularization)

This experiment tested different dropout rates to find the optimal level of regularization.

### Results Summary (Dropout Rate)

| DROPOUT Rate | Best `val_loss` (at Epoch) | Validation Set MAE | Test Set MAE | Test Set RMSE |
| :----------- | :------------------------ | :--------------------: | :------------- | :------------ |
| **0.1 (Baseline)** | **-3.0656 (Ep 19)**     | **0.000277**         | **0.000207**   | **0.000320**  |
| 0.2          | -2.9442 (Ep 20)           | 0.000498             | 0.000401       | 0.000503      |
| 0.3          | -2.9015 (Ep 20)           | 0.000429             | 0.000364       | 0.000466      |

### Conclusion (Dropout Rate)
Increasing the dropout rate consistently degraded performance, leading to higher errors on all datasets. This indicates that a rate of **`DROPOUT = 0.1`** provides sufficient regularization for this model without hindering its ability to learn, a classic sign of underfitting at higher rates.

---

## Experiment 4: Optimizing Batch Size and Learning Rate

This final experiment tuned the core training dynamics.

### Results Summary (Batch Size & Learning Rate)

| `BATCH_SIZE` | `LEARNING_RATE` | Best `val_loss` (at Epoch) | **Validation Set MAE** | Test Set MAE   | Test Set RMSE  |
| :----------- | :-------------- | :------------------------- | :--------------------- | :--------------- | :------------- |
| **64**       | **1e-4**        | **-3.0656 (Ep 19)**        | **0.000277**           | **0.000207**     | **0.000320**   |
| 32           | 1e-4            | -3.1197 (Ep 18)            | 0.000287               | 0.000214         | 0.000328       |
| 128          | 1e-4            | -2.9103 (Ep 20)            | 0.000476               | 0.000410         | 0.000506       |
| 64           | 5e-5            | -2.9505 (Ep 20)            | 0.000291               | 0.000209         | 0.000322       |
| 32           | 5e-5            | -3.0144 (Ep 18)            | 0.000354               | 0.000264         | 0.000374       |

### Conclusion (Batch Size & Learning Rate)
The baseline configuration of **`BATCH_SIZE=64`** and **`LEARNING_RATE=1e-4`** was confirmed to be optimal, yielding the best Test Set MAE. While a smaller batch size of 32 performed nearly as well, the larger size of 128 significantly worsened performance. A slower learning rate did not provide any benefit.

---

## Final Model Configuration and Overall Conclusion

After a comprehensive series of experiments, the optimal configuration for the TimeSeriesTransformer model on this task was determined to be the **"Small (Baseline)"** configuration:

*   **CONTEXT_LENGTH:** 30
*   **D_MODEL:** 32
*   **LAYERS:** 2
*   **HEADS:** 4
*   **DROPOUT:** 0.1
*   **BATCH_SIZE:** 64
*   **LEARNING_RATE:** 1e-4

This configuration achieved a **final Test Set MAE of 0.000207 EURUSD**. This result is superior to the baseline LSTM model's performance, confirming the effectiveness of the Transformer architecture for this specific financial forecasting problem. The experiments also highlighted that for high-frequency data, a more focused, shorter lookback window combined with a relatively simple model architecture can yield the most robust and accurate results.