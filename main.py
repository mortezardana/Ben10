# main.py

from pipeline.config import CONFIG
from pipeline.data_loader import load_data, normalize_features
from models.baseline_ml import train_baseline_ml
from models.lstm_model import train_lstm_model, predict_lstm
from models.cnn_lstm import train_cnn_lstm_model, predict_cnn_lstm
from models.tcn_model import train_tcn_model, predict_tcn
from models.transformer_model import train_transformer_model, predict_transformer
from models.predictor import load_model, predict
from pipeline.backtest import simple_strategy_backtest
from pipeline.evaluator import (
    plot_equity_curve,
    plot_predictions,
    plot_feature_importance,
    plot_confusion,
    plot_rolling_sharpe
)
from utils.logger import get_logger
from utils.plot_utils import create_plot_dir

logger = get_logger("main")


def main():
    logger.info("Starting Gold Trading AI pipeline")

    df = load_data(
        filepath=CONFIG['data_path'],
        target_horizon=CONFIG['target_horizon'],
        target_type=CONFIG['target_type']
    )
    df, scaler = normalize_features(df, exclude_cols=CONFIG['exclude_cols'])
    plot_dir = create_plot_dir()

    # XGBoost
    model, metrics = train_baseline_ml(df)
    logger.info(f"XGBoost Accuracy: {metrics['accuracy']:.4f}")
    df['prediction'] = predict(load_model(CONFIG['model_output_dir'] / "xgb_baseline.pkl"), df)
    df.to_csv("data/predictions_with_features.csv", index=False)
    backtest_df = simple_strategy_backtest(df)
    backtest_df.to_csv("data/backtest_results.csv", index=False)
    plot_equity_curve(backtest_df, plot_dir=plot_dir, model_name="xgboost")
    plot_predictions(df, plot_dir=plot_dir, model_name="xgboost")
    plot_confusion(df, plot_dir=plot_dir, model_name="xgboost")
    plot_rolling_sharpe(backtest_df, plot_dir=plot_dir, model_name="xgboost")
    feature_cols = df.drop(columns=['target', 'prediction', 'date'], errors='ignore').select_dtypes(
        include='number').columns.tolist()
    plot_feature_importance(model, feature_cols, plot_dir=plot_dir, model_name="xgboost")

    # LSTM
    lstm_model, lstm_metrics = train_lstm_model(df)
    logger.info(f"LSTM Accuracy: {lstm_metrics['accuracy']:.4f}")
    lstm_pred_df = predict_lstm(lstm_model, df)
    lstm_backtest_df = simple_strategy_backtest(lstm_pred_df)
    lstm_pred_df.to_csv("data/lstm_predictions.csv", index=False)
    lstm_backtest_df.to_csv("data/lstm_backtest_results.csv", index=False)
    plot_equity_curve(lstm_backtest_df, plot_dir=plot_dir, model_name="lstm")
    plot_predictions(lstm_pred_df, plot_dir=plot_dir, model_name="lstm")
    plot_confusion(lstm_pred_df, plot_dir=plot_dir, model_name="lstm")

    # CNN-LSTM
    cnn_lstm_model, cnn_lstm_metrics = train_cnn_lstm_model(df)
    logger.info(f"CNN-LSTM Accuracy: {cnn_lstm_metrics['accuracy']:.4f}")
    cnn_pred_df = predict_cnn_lstm(cnn_lstm_model, df)
    cnn_backtest_df = simple_strategy_backtest(cnn_pred_df)
    cnn_pred_df.to_csv("data/cnn_lstm_predictions.csv", index=False)
    cnn_backtest_df.to_csv("data/cnn_lstm_backtest_results.csv", index=False)
    plot_equity_curve(cnn_backtest_df, plot_dir=plot_dir, model_name="cnn_lstm")
    plot_predictions(cnn_pred_df, plot_dir=plot_dir, model_name="cnn_lstm")
    plot_confusion(cnn_pred_df, plot_dir=plot_dir, model_name="cnn_lstm")

    # TCN
    tcn_model, tcn_metrics = train_tcn_model(df)
    logger.info(f"TCN Accuracy: {tcn_metrics['accuracy']:.4f}")
    tcn_pred_df = predict_tcn(tcn_model, df)
    tcn_backtest_df = simple_strategy_backtest(tcn_pred_df)
    tcn_pred_df.to_csv("data/tcn_predictions.csv", index=False)
    tcn_backtest_df.to_csv("data/tcn_backtest_results.csv", index=False)
    plot_equity_curve(tcn_backtest_df, plot_dir=plot_dir, model_name="tcn")
    plot_predictions(tcn_pred_df, plot_dir=plot_dir, model_name="tcn")
    plot_confusion(tcn_pred_df, plot_dir=plot_dir, model_name="tcn")

    # Transformer
    transformer_model, transformer_metrics = train_transformer_model(df)
    logger.info(f"Transformer Accuracy: {transformer_metrics['accuracy']:.4f}")
    transformer_pred_df = predict_transformer(transformer_model, df)
    transformer_backtest_df = simple_strategy_backtest(transformer_pred_df)
    transformer_pred_df.to_csv("data/transformer_predictions.csv", index=False)
    transformer_backtest_df.to_csv("data/transformer_backtest_results.csv", index=False)
    plot_equity_curve(transformer_backtest_df, plot_dir=plot_dir, model_name="transformer")
    plot_predictions(transformer_pred_df, plot_dir=plot_dir, model_name="transformer")
    plot_confusion(transformer_pred_df, plot_dir=plot_dir, model_name="transformer")


if __name__ == '__main__':
    main()
