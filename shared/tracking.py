"""Experiment tracking with MLflow."""
from utils.logger import get_logger

logger = get_logger("tracking")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Experiment tracking disabled.")


class ExperimentTracker:
    def __init__(self, tracking_uri="mlruns", experiment_name="ben10"):
        self.enabled = MLFLOW_AVAILABLE
        if self.enabled:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow tracking: uri={tracking_uri}, experiment={experiment_name}")

    def log_pipeline_run(self, pipeline_name, params, metrics, model=None):
        """Log a pipeline training run to MLflow."""
        if not self.enabled:
            logger.info(f"[NO-MLFLOW] {pipeline_name}: params={params}, metrics={metrics}")
            return

        with mlflow.start_run(run_name=pipeline_name):
            mlflow.set_tag("pipeline", pipeline_name)

            # Log params (flatten nested dicts)
            for k, v in (params or {}).items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        mlflow.log_param(f"{k}.{k2}", v2)
                else:
                    mlflow.log_param(k, v)

            # Log metrics
            for k, v in (metrics or {}).items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)

            # Log model artifact
            if model is not None:
                try:
                    mlflow.sklearn.log_model(model, "model")
                except Exception:
                    pass  # Not all models are sklearn-compatible

            logger.info(f"Logged run: {pipeline_name}")

    def log_combiner_run(self, combiner_type, pipeline_weights, combined_metrics):
        """Log a combiner run."""
        if not self.enabled:
            return

        with mlflow.start_run(run_name=f"combiner_{combiner_type}"):
            mlflow.set_tag("type", "combiner")
            mlflow.set_tag("combiner_type", combiner_type)

            for name, weight in (pipeline_weights or {}).items():
                mlflow.log_metric(f"weight_{name}", weight)

            for k, v in (combined_metrics or {}).items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)

    def get_best_run(self, pipeline_name, metric="accuracy"):
        """Get the best run for a pipeline by metric."""
        if not self.enabled:
            return None

        experiment = mlflow.get_experiment_by_name("ben10")
        if experiment is None:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.pipeline = '{pipeline_name}'",
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )

        if runs.empty:
            return None
        return runs.iloc[0].to_dict()

    def register_model(self, pipeline_name, run_id, stage="staging"):
        """Register model in MLflow Model Registry."""
        if not self.enabled:
            return

        model_uri = f"runs:/{run_id}/model"
        try:
            result = mlflow.register_model(model_uri, pipeline_name)
            if stage != "None":
                client = mlflow.tracking.MlflowClient()
                client.transition_model_version_stage(
                    name=pipeline_name,
                    version=result.version,
                    stage=stage,
                )
            logger.info(f"Registered model: {pipeline_name} v{result.version} ({stage})")
        except Exception as e:
            logger.warning(f"Failed to register model: {e}")
