# Model Development and Experiments: TimeSeriesTransformer

This document outlines the experiments conducted to fine-tune the TimeSeriesTransformer model for forecasting EURUSD 5-minute closing prices. The goal is to systematically evaluate the impact of different hyperparameter choices on the model's predictive performance and compare it to the LSTM baseline.

All experiments use the 10-year, 5-minute EURUSD dataset with a 60/20/20 chronological split for training, validation, and testing.

## Experiment 1: Optimizing Lookback Window (CONTEXT_LENGTH)

The first set of experiments focuses on determining the most effective lookback window (`CONTEXT_LENGTH`), which is analogous to the `WINDOW` parameter in the LSTM model.

### Methodology (Context Length)
The core Transformer architecture and other training parameters were kept constant while varying `CONTEXT_LENGTH`.
*   **Base Model Architecture:**
    *   `D_MODEL`: 32
    *   `ENCODER_LAYERS`: 2, `DECODER_LAYERS`: 2
    *   `*_ATTENTION_HEADS`: 4
    *   `DROPOUT`: 0.1
*   **Constant Training Parameters:**
    *   `EPOCHS`: 20 (with EarlyStopping `PATIENCE=10`)
    *   `BATCH_SIZE`: 64
    *   `LEARNING_RATE`: 1e-4

### Results Summary (Context Length)

| CONTEXT_LENGTH | Val Set Loss (Best) | Test Set MAE (EURUSD) | Test Set RMSE (EURUSD) |
| :------------: | :------------------: | :--------------------: | :---------------------: |
|       **30**       | *[Fill from run]*   |   *[Fill from run]*    |   *[Fill from run]*    |
|       **60**       | *[Fill from run]*   |   *[Fill from run]*    |   *[Fill from run]*    |
|      **120**       | *[Fill from run]*   |   *[Fill from run]*    |   *[Fill from run]*    |
|      **288 (1 Day)** | *[Fill from run]*   |   *[Fill from run]*    |   *[Fill from run]*    |

*(Note: Run the notebook for each `CONTEXT_LENGTH` value, record the best validation loss achieved during training, and the final Test Set MAE/RMSE.)*

### Conclusion (Context Length)
*Based on these results, a `CONTEXT_LENGTH` of **[Your chosen value]** was selected as it provided the best performance on the unseen test data. This value will be used for subsequent hyperparameter tuning.*

---

## Experiment 2: Optimizing Model Size (d_model, layers, heads)
*(To be conducted after Experiment 1)*

---

## Experiment 3: Optimizing Dropout Rate
*(To be conducted after Experiment 2)*

---
