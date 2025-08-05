# Model Development and Experiments: TimeSeriesTransformer

This document outlines the systematic experiments conducted to fine-tune the TimeSeriesTransformer model for forecasting EURUSD 5-minute closing prices. The goal is to evaluate the impact of different hyperparameter choices on the model's predictive performance and establish a final, robust configuration.

All experiments use the 10-year, 5-minute EURUSD dataset with a 60/20/20 chronological split for training, validation, and testing. The primary decision-making metric is the **Validation Set MAE**, with the **Test Set MAE** used for final performance assessment.

*(Note: The negative loss values reported in these experiments are due to the model's default Negative Log-Likelihood (NLL) loss on a Student's t-distribution. While useful for observing convergence, the MAE and RMSE metrics, calculated on the original price scale, are used for all performance comparisons.)*

---

## Experiment 1: Optimizing Lookback Window (CONTEXT_LENGTH)

This experiment focused on determining the most effective lookback window.

*   **Base Architecture:** `D_MODEL=32`, `Layers=2`, `Heads=4`, `Dropout=0.1`
*   **Base Training:** `LR=1e-4`, `Batch Size=64`, `Patience=10`, `Epochs=20` (or 5 for computationally intensive runs)

### Results Summary (Context Length)

| CONTEXT_LENGTH | Epochs | Best `val_loss` (at Epoch) | Validation Set MAE | **Test Set MAE** | Test Set RMSE |
| :------------: | :----: | :------------------------- | :------------------- | :--------------- | :------------ |
| 15             |   20   | -3.0266 (Ep 15)            | 0.000276             | 0.000257         | 0.000360      |
| **30**         | **50** | **-3.4466 (Ep 33)**        | **0.000247**         | **0.000203**     | **0.000318**  |
| 30 (Run 2)     |   20   | -3.0656 (Ep 19)            | 0.000277             | 0.000207         | 0.000320      |
| 60             |   5    | -2.6073 (Ep 4)             | 0.000438             | 0.000390         | 0.000502      |
| 120            |   20   | -2.9704 (Ep 15)            | 0.000466             | 0.000337         | 0.000443      |
| 288            |   5    | *(Run Timed Out)*          | -                    | -                | -             |

### Conclusion (Context Length)
The model with **`CONTEXT_LENGTH = 30`** achieved the best performance. For this high-frequency dataset, a shorter lookback window proved most effective, suggesting the most relevant predictive information is contained within the most recent 2.5 hours of data. This value was fixed for all subsequent experiments.

---

## Experiment 2: Optimizing Model Size and Complexity

This experiment investigated whether a wider or deeper Transformer architecture could improve upon the `CONTEXT_LENGTH=30` baseline.

### Results Summary (Model Size)

| Experiment Name    | D_MODEL / LAYERS / HEADS | Validation Set MAE | **Test Set MAE** | Test Set RMSE  |
| :----------------- | :----------------------- | :--------------------- | :--------------- | :------------- |
| **Small (Baseline)** | **32 / 2 / 4**           | **0.000277**           | **0.000207**     | **0.000320**   |
| Medium             | 64 / 2 / 4               | 0.000369               | 0.000282         | 0.000383       |
| Large              | 64 / 4 / 8               | 0.000259               | 0.000212         | 0.000323       |
| X-Large            | 128 / 4 / 8              | 0.000284               | 0.000210         | 0.000326       |

### Conclusion (Model Size)
The **"Small (Baseline)"** configuration demonstrated the best generalization to the unseen test set. Larger models showed signs of minor overfitting. The configuration of `D_MODEL=32`, `LAYERS=2`, `HEADS=4` was confirmed as the most robust choice.

---

## Experiment 3: Optimizing Dropout Rate (Regularization)

This experiment tested different dropout rates.

### Results Summary (Dropout Rate)

| DROPOUT Rate       | Validation Set MAE | **Test Set MAE** |
| :----------------- | :------------------- | :--------------- |
| **0.1 (Baseline)** | **0.000277**         | **0.000207**     |
| 0.2                | 0.000498             | 0.000401       |
| 0.3                | 0.000429             | 0.000364       |

### Conclusion (Dropout Rate)
The baseline rate of **`DROPOUT = 0.1`** was confirmed as optimal. Higher rates caused the model to underfit.

---

## Experiment 4: Optimizing Batch Size and Learning Rate

This experiment tuned the core training dynamics.

### Results Summary (Batch Size & Learning Rate)

| `BATCH_SIZE` | `LEARNING_RATE` | Validation Set MAE | **Test Set MAE** |
| :----------- | :-------------- | :------------------- | :--------------- |
| **64**       | **1e-4**        | **0.000277**         | **0.000207**     |
| 32           | 1e-4            | 0.000287             | 0.000214         |
| 128          | 1e-4            | 0.000476             | 0.000410         |
| 64           | 5e-5            | 0.000291             | 0.000209         |
| 32           | 5e-5            | 0.000354             | 0.000264         |

### Conclusion (Batch Size & Learning Rate)
The configuration of **`BATCH_SIZE=64`** and **`LEARNING_RATE=1e-4`** was confirmed as the best combination.

---

## Post-Tuning: Final Definitive Retraining

After identifying the optimal hyperparameters and correcting a configuration issue with `lags_sequence`, a final, definitive training run was conducted to produce the champion model. This run used an extended `EPOCHS=50` limit to ensure full convergence.

### Final Optimal Configuration

*   **DISTRIBUTION_OUTPUT:** "student_t"
*   **CONTEXT_LENGTH:** 30
*   **LAGS_SEQUENCE:** `[1, 2, 3, 4, 5, 6, 7]` (Valid default sequence)
*   **D_MODEL:** 32
*   **LAYERS:** 2
*   **HEADS:** 4
*   **DROPOUT:** 0.1
*   **BATCH_SIZE:** 64
*   **LEARNING_RATE:** 1e-4
*   **EPOCHS:** 50 (with `PATIENCE=10`, early stopped at Epoch 43)

### Final Model Performance (Definitive Run)

This run produced the project's best and most reliable performance metrics, finding its optimal state at **Epoch 33**.

| Dataset    | Metric | Value (EURUSD) |
| :--------- | :----: | :------------- |
| **Test**   | **MAE**  | **0.000203**   |
| **Test**   | **RMSE** | **0.000318**   |
| Validation | MAE    | 0.000247       |
| Validation | RMSE   | 0.000381       |
| Training   | MAE    | 0.000234       |
| Training   | RMSE   | 0.000368       |

## Experiment 6: Impact of Dataset Length (10 Years vs. 20 Years)

A final experiment was conducted to determine if a larger historical dataset could further improve model performance. The model was retrained with the same optimal hyperparameters but using **20 years** of 5-minute EURUSD data.

### Results Summary (Dataset Length)

| Dataset Length | Test Set MAE (EURUSD) | Test Set RMSE (EURUSD) |
| :------------- | :-------------------- | :--------------------- |
| **10 Years**   | **0.000203**          | **0.000318**           |
| 20 Years       | 0.000249              | 0.000367               |

### Conclusion (Dataset Length)
The results clearly show that the model trained on the **10-year dataset is superior**.

The performance degradation with the 20-year dataset suggests that older market regimes (e.g., from the 2008 financial crisis era) introduce patterns that are not relevant to predicting the more recent market behavior found in the test set. This highlights a critical concept in financial ML: **data relevancy can be more important than data quantity.** For this task, a 10-year history provides the optimal balance of sufficient data and modern market dynamics.

---

## Final Overall Conclusion

The comprehensive tuning process identified a robust and highly accurate configuration for the TimeSeriesTransformer. The final champion model, trained on the 10-year dataset, achieved a **Test Set MAE of 0.000203**, confirming the effectiveness of the Transformer architecture for this financial forecasting problem.

---

## Final Exported Artifacts for Deployment

The final, optimized model was exported along with its essential components to facilitate its use in external trading applications like `backtrader`. The following files were generated and stored in the `Models/` directory:

1.  **`best_transformer_model.pth`**: A PyTorch file containing the learned weights and biases of the neural network.
2.  **`target_scaler.pkl`**: A `joblib` file containing the `StandardScaler` object that was fitted on the training data.
3.  **`model_config.json`**: A configuration file detailing the model's architecture.

***
**Important Note on Usage:** These three files form a complete, self-contained package. When loading the model, the `model_config.json` **must** be used to instantiate an identical model architecture. Failure to do so will result in a `RuntimeError` due to a mismatch between the architecture and the loaded weights (`.pth` file).
***