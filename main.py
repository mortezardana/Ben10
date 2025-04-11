# main.py

from pipeline.config import CONFIG
from pipeline.data_loader import load_data, normalize_features
from models.baseline_ml import train_baseline_ml
from utils.logger import get_logger
from models.predictor import load_model, predict
from pipeline.backtest import simple_strategy_backtest
from pipeline.evaluator import (
    plot_equity_curve,
    plot_predictions,
    plot_feature_importance,
    plot_confusion,
    plot_rolling_sharpe
)
from models.lstm_model import train_lstm_model, predict_lstm
from models.cnn_lstm import train_cnn_lstm_model, predict_cnn_lstm
from models.tcn_model import train_tcn_model, predict_tcn
from models.transformer_model import train_transformer_model, predict_transformer
from models.crnn_model import train_crnn_model, predict_crnn
from models.gru_model import train_gru_model, predict_gru
from models.tabnet_model import train_tabnet_model, predict_tabnet
import torch

logger = get_logger("main")


def main():
    logger.info("Starting Gold Trading AI pipeline")

    # Select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device.upper()}")

    # Load and preprocess data
    df = load_data(
        filepath=CONFIG['data_path'],
        target_horizon=CONFIG['target_horizon'],
        target_type=CONFIG['target_type']
    )
    df, scaler = normalize_features(df, exclude_cols=CONFIG['exclude_cols'])

    # Train baseline model
    model, metrics = train_baseline_ml(
        df,
        target_col='target',
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['random_state'],
        save_model=CONFIG['save_model']
    )

    logger.info(f"Training complete. Accuracy: {metrics['accuracy']:.4f}")

    # Predictor testing
    predictor(df)

    # Run backtest
    backtest_df = simple_strategy_backtest(df)
    backtest_df.to_csv("data/backtest_results.csv", index=False)
    logger.info("Backtest results saved to data/backtest_results.csv")

    # Plotting
    plot_equity_curve(backtest_df)
    plot_predictions(df)
    plot_confusion(df)
    plot_rolling_sharpe(backtest_df)

    feature_cols = df.drop(columns=['target', 'prediction', 'date'], errors='ignore').select_dtypes(
        include='number').columns.tolist()
    plot_feature_importance(model, feature_cols)

    # LSTM
    lstm_model, lstm_metrics = train_lstm_model(df, device=device)
    logger.info(f"LSTM Accuracy: {lstm_metrics['accuracy']:.4f}")
    lstm_pred_df = predict_lstm(lstm_model, df, device=device)
    lstm_backtest_df = simple_strategy_backtest(lstm_pred_df)
    lstm_pred_df.to_csv("data/lstm_predictions.csv", index=False)
    lstm_backtest_df.to_csv("data/lstm_backtest_results.csv", index=False)
    plot_predictions(lstm_pred_df)
    plot_confusion(lstm_pred_df)
    plot_equity_curve(lstm_backtest_df)

    # CNN-LSTM
    cnn_lstm_model, cnn_lstm_metrics = train_cnn_lstm_model(df, device=device)
    cnn_pred_df = predict_cnn_lstm(cnn_lstm_model, df, device=device)
    cnn_backtest_df = simple_strategy_backtest(cnn_pred_df)
    cnn_pred_df.to_csv("data/cnn_lstm_predictions.csv", index=False)
    cnn_backtest_df.to_csv("data/cnn_lstm_backtest_results.csv", index=False)
    plot_predictions(cnn_pred_df)
    plot_confusion(cnn_pred_df)
    plot_equity_curve(cnn_backtest_df)

    # CRNN
    crnn_model, crnn_metrics = train_crnn_model(df, device=device)
    logger.info(f"CRNN Accuracy: {crnn_metrics['accuracy']:.4f}")
    crnn_pred_df = predict_crnn(crnn_model, df, device=device)
    crnn_backtest_df = simple_strategy_backtest(crnn_pred_df)
    crnn_pred_df.to_csv("data/crnn_predictions.csv", index=False)
    crnn_backtest_df.to_csv("data/crnn_backtest_results.csv", index=False)
    plot_predictions(crnn_pred_df)
    plot_confusion(crnn_pred_df)
    plot_equity_curve(crnn_backtest_df)

    # GRU
    gru_model, gru_metrics = train_gru_model(df, device=device)
    logger.info(f"GRU Accuracy: {gru_metrics['accuracy']:.4f}")
    gru_pred_df = predict_gru(gru_model, df, device=device)
    gru_backtest_df = simple_strategy_backtest(gru_pred_df)
    gru_pred_df.to_csv("data/gru_predictions.csv", index=False)
    gru_backtest_df.to_csv("data/gru_backtest_results.csv", index=False)
    plot_predictions(gru_pred_df)
    plot_confusion(gru_pred_df)
    plot_equity_curve(gru_backtest_df)

    # TabNet
    tabnet_model, tabnet_metrics = train_tabnet_model(df, device_name=device)
    logger.info(f"TabNet Accuracy: {tabnet_metrics['accuracy']:.4f}")
    tabnet_pred_df = predict_tabnet(tabnet_model, df)
    tabnet_backtest_df = simple_strategy_backtest(tabnet_pred_df)
    tabnet_pred_df.to_csv("data/tabnet_predictions.csv", index=False)
    tabnet_backtest_df.to_csv("data/tabnet_backtest_results.csv", index=False)
    plot_predictions(tabnet_pred_df)
    plot_confusion(tabnet_pred_df)
    plot_equity_curve(tabnet_backtest_df)


def predictor(df):
    model_path = CONFIG['model_output_dir'] / "xgb_baseline.pkl"
    model = load_model(model_path)
    preds = predict(model, df)
    logger.info("Sample predictions:")
    logger.info(preds.head())
    df['prediction'] = preds
    df.to_csv("Data/predictions_with_features.csv", index=False)
    logger.info("Predictions saved to data/predictions_with_features.csv")


if __name__ == '__main__':
    main()
