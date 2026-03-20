# main.py — Orchestrator for Ben10 Gold Trading AI

import argparse
import json
from pathlib import Path

from utils.logger import get_logger
from utils.seed import set_all_seeds
from shared.data_loader import load_gold_data
from shared.config import load_config, AppConfig
from pipeline.backtest import BacktestEngine
from utils.benchmarks import buy_and_hold, sma_crossover

logger = get_logger("main")

# Registry of available pipelines
PIPELINE_REGISTRY = {
    'xgboost': ('pipelines.gradient_boosting.xgboost_pipeline', 'XGBoostPipeline'),
    'lightgbm': ('pipelines.gradient_boosting.lightgbm_pipeline', 'LightGBMPipeline'),
    'catboost': ('pipelines.gradient_boosting.catboost_pipeline', 'CatBoostPipeline'),
    'lstm': ('pipelines.deep_learning.lstm_pipeline', 'LSTMPipeline'),
    'gru': ('pipelines.deep_learning.gru_pipeline', 'GRUPipeline'),
    'tcn': ('pipelines.deep_learning.tcn_pipeline', 'TCNPipeline'),
    'transformer': ('pipelines.deep_learning.transformer_pipeline', 'TransformerPipeline'),
    'cnn_lstm': ('pipelines.deep_learning.cnn_lstm_pipeline', 'CNNLSTMPipeline'),
    'crnn': ('pipelines.deep_learning.crnn_pipeline', 'CRNNPipeline'),
    'tabnet': ('pipelines.deep_learning.tabnet_pipeline', 'TabNetPipeline'),
    'tft': ('pipelines.deep_learning.tft_pipeline', 'TFTPipeline'),
    'hmm_regime': ('pipelines.regime.hmm_pipeline', 'HMMRegimePipeline'),
}


def get_pipeline(name, config=None):
    """Dynamically import and instantiate a pipeline by name."""
    if name not in PIPELINE_REGISTRY:
        raise ValueError(f"Unknown pipeline: {name}. Available: {list(PIPELINE_REGISTRY.keys())}")
    module_path, class_name = PIPELINE_REGISTRY[name]
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config=config)


def run_pipeline(pipeline, data, backtest_engine=None):
    """Train, predict, evaluate, and backtest a single pipeline."""
    train_metrics = pipeline.train(data['train'], data.get('val'))
    logger.info(f"[{pipeline.name}] Train metrics: {train_metrics}")

    # Predict on test
    output = pipeline.predict(data['test'])
    eval_metrics = pipeline.evaluate(data['test'])
    logger.info(f"[{pipeline.name}] Test metrics: {eval_metrics}")

    # Backtest if prices available
    bt_result = None
    if backtest_engine and 'Close' in data['test'].columns:
        prices = data['test']['Close']
        bt_result = backtest_engine.run(prices, output.signals)
        logger.info(f"[{pipeline.name}] Backtest: {bt_result.metrics}")

    return {
        'name': pipeline.name,
        'train_metrics': train_metrics,
        'eval_metrics': eval_metrics,
        'output': output,
        'backtest': bt_result,
    }


def main():
    parser = argparse.ArgumentParser(description='Ben10 Gold Trading AI')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--pipeline', type=str, default=None,
                        help='Run a specific pipeline (e.g., xgboost, lstm)')
    parser.add_argument('--run-all', action='store_true',
                        help='Run all available pipelines')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Set seeds for reproducibility
    set_all_seeds(args.seed)

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.warning(f"Could not load config from {args.config}: {e}. Using defaults.")
        config = AppConfig()

    logger.info("Starting Ben10 Gold Trading AI pipeline")

    # Load data
    data = load_gold_data(config)
    logger.info(f"Data loaded: train={len(data['train'])}, val={len(data['val'])}, test={len(data['test'])}")

    # Backtesting engine
    bt_engine = BacktestEngine(initial_capital=100000, commission=0.0001, slippage=0.0001)

    # Determine which pipelines to run
    if args.pipeline:
        pipeline_names = [args.pipeline]
    elif args.run_all:
        pipeline_names = list(PIPELINE_REGISTRY.keys())
    else:
        # Default: just XGBoost
        pipeline_names = ['xgboost']

    # Run pipelines and collect results
    results = {}
    for name in pipeline_names:
        logger.info(f"{'='*60}")
        logger.info(f"Running pipeline: {name}")
        logger.info(f"{'='*60}")
        try:
            pipeline = get_pipeline(name)
            result = run_pipeline(pipeline, data, bt_engine)
            results[name] = result
        except Exception as e:
            logger.error(f"Pipeline {name} failed: {e}")
            import traceback
            traceback.print_exc()

    # Run benchmarks for comparison
    if 'Close' in data['test'].columns:
        logger.info(f"{'='*60}")
        logger.info("Running benchmarks...")
        prices = data['test']['Close']
        bh_equity = buy_and_hold(prices)
        bh_return = float(bh_equity.iloc[-1] - 1) if len(bh_equity) > 0 else 0
        logger.info(f"Buy & Hold return: {bh_return:.4f}")

        sma_signals = sma_crossover(prices, fast=50, slow=200)
        sma_bt = bt_engine.run(prices, sma_signals)
        logger.info(f"SMA Crossover metrics: {sma_bt.metrics}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    for name, result in results.items():
        eval_m = result['eval_metrics']
        bt_m = result['backtest'].metrics if result['backtest'] else {}
        logger.info(f"{name}: accuracy={eval_m.get('accuracy', 'N/A')}, "
                     f"sharpe={bt_m.get('sharpe_ratio', 'N/A')}, "
                     f"max_dd={bt_m.get('max_drawdown', 'N/A')}")

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    summary = {name: {
        'eval_metrics': r['eval_metrics'],
        'backtest_metrics': r['backtest'].metrics if r['backtest'] else {},
    } for name, r in results.items()}

    with open(output_dir / "pipeline_results.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Results saved to {output_dir / 'pipeline_results.json'}")


if __name__ == '__main__':
    main()
