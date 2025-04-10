# main.py

from pipeline.config import CONFIG
from pipeline.data_loader import normalize_features
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
from utils.multi_timeframe_utils import merge_multi_timeframes

logger = get_logger("main")


def main():
    logger.info("Starting Gold Trading AI pipeline")

    files = {
        '1h': 'data/gold_1h.csv',
        '4h': 'data/gold_4h.csv',
        '1d': 'data/gold_1d.csv'
    }
    df = merge_multi_timeframes(files)

    # Fix: look for close_4h with case-insensitivity
    close_candidates = [col for col in df.columns if col.lower().endswith("close_4h")]
    if close_candidates:
        close_col = close_candidates[0]
        df['target'] = (df[close_col].shift(-1) > df[close_col]).astype(int)
        df.dropna(inplace=True)
    else:
        logger.error("Missing 'close_4h' column for target generation")
        logger.info(f"Available columns: {df.columns.tolist()}")
        return

    df, scaler = normalize_features(df, exclude_cols=CONFIG['exclude_cols'])

    plot_dir = create_plot_dir()

    # XGBoost
    model, metrics = train_baseline_ml(df)
    logger.info(f"XGBoost Accuracy: {metrics['accuracy']:.4f}")
    df['prediction'] = predict(load_model(CONFIG['model_output_dir'] / "xgb_baseline.pkl"), df)
    df.to_csv("data/predictions_with_features_mtf.csv", index=False)
    backtest_df = simple_strategy_backtest(df, price_col=close_col)
    backtest_df.to_csv("data/backtest_results_mtf.csv", index=False)
    plot_equity_curve(backtest_df, plot_dir=plot_dir, model_name="xgboost_mtf")
    plot_predictions(df, plot_dir=plot_dir, model_name="xgboost_mtf")
    plot_confusion(df, plot_dir=plot_dir, model_name="xgboost_mtf")
    plot_rolling_sharpe(backtest_df, plot_dir=plot_dir, model_name="xgboost_mtf")
    feature_cols = df.drop(columns=['target', 'prediction', 'date'], errors='ignore').select_dtypes(
        include='number').columns.tolist()
    plot_feature_importance(model, feature_cols, plot_dir=plot_dir, model_name="xgboost_mtf")

    # LSTM
    lstm_model, lstm_metrics = train_lstm_model(df)
    logger.info(f"LSTM Accuracy: {lstm_metrics['accuracy']:.4f}")
    lstm_pred_df = predict_lstm(lstm_model, df)
    lstm_backtest_df = simple_strategy_backtest(lstm_pred_df, price_col=close_col)
    lstm_pred_df.to_csv("data/lstm_mtf_predictions.csv", index=False)
    lstm_backtest_df.to_csv("data/lstm_mtf_backtest_results.csv", index=False)
    plot_equity_curve(lstm_backtest_df, plot_dir=plot_dir, model_name="lstm_mtf")
    plot_predictions(lstm_pred_df, plot_dir=plot_dir, model_name="lstm_mtf")
    plot_confusion(lstm_pred_df, plot_dir=plot_dir, model_name="lstm_mtf")

    # CNN-LSTM
    cnn_lstm_model, cnn_lstm_metrics = train_cnn_lstm_model(df)
    logger.info(f"CNN-LSTM Accuracy: {cnn_lstm_metrics['accuracy']:.4f}")
    cnn_pred_df = predict_cnn_lstm(cnn_lstm_model, df)
    cnn_backtest_df = simple_strategy_backtest(cnn_pred_df, price_col=close_col)
    cnn_pred_df.to_csv("data/cnn_lstm_mtf_predictions.csv", index=False)
    cnn_backtest_df.to_csv("data/cnn_lstm_mtf_backtest_results.csv", index=False)
    plot_equity_curve(cnn_backtest_df, plot_dir=plot_dir, model_name="cnn_lstm_mtf")
    plot_predictions(cnn_pred_df, plot_dir=plot_dir, model_name="cnn_lstm_mtf")
    plot_confusion(cnn_pred_df, plot_dir=plot_dir, model_name="cnn_lstm_mtf")

    # TCN
    tcn_model, tcn_metrics = train_tcn_model(df)
    logger.info(f"TCN Accuracy: {tcn_metrics['accuracy']:.4f}")
    tcn_pred_df = predict_tcn(tcn_model, df)
    tcn_backtest_df = simple_strategy_backtest(tcn_pred_df, price_col=close_col)
    tcn_pred_df.to_csv("data/tcn_mtf_predictions.csv", index=False)
    tcn_backtest_df.to_csv("data/tcn_mtf_backtest_results.csv", index=False)
    plot_equity_curve(tcn_backtest_df, plot_dir=plot_dir, model_name="tcn_mtf")
    plot_predictions(tcn_pred_df, plot_dir=plot_dir, model_name="tcn_mtf")
    plot_confusion(tcn_pred_df, plot_dir=plot_dir, model_name="tcn_mtf")

    # Transformer
    transformer_model, transformer_metrics = train_transformer_model(df)
    logger.info(f"Transformer Accuracy: {transformer_metrics['accuracy']:.4f}")
    transformer_pred_df = predict_transformer(transformer_model, df)
    transformer_backtest_df = simple_strategy_backtest(transformer_pred_df, price_col=close_col)
    transformer_pred_df.to_csv("data/transformer_mtf_predictions.csv", index=False)
    transformer_backtest_df.to_csv("data/transformer_mtf_backtest_results.csv", index=False)
    plot_equity_curve(transformer_backtest_df, plot_dir=plot_dir, model_name="transformer_mtf")
    plot_predictions(transformer_pred_df, plot_dir=plot_dir, model_name="transformer_mtf")
    plot_confusion(transformer_pred_df, plot_dir=plot_dir, model_name="transformer_mtf")


if __name__ == '__main__':
    main()
