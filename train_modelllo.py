import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pytorch_lightning as pl
from pytorch_lightning import loggers
from hydra.utils import to_absolute_path
import mlflow
import mlflow.sklearn
import joblib
import os
import dagshub

def setup_mlflow(tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
    return client


# Add Dagshub authentication
dagshub.auth.add_app_token(token="505fecb8c64c86e969ef3556e9a062da6d7d35ac")

# Hydra configuration
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize Dagshub repo with MLflow tracking
    dagshub.init(
        repo_owner=cfg["model"]["repo_owner"],
        repo_name=cfg["model"]["repo_name"],
        mlflow=cfg["model"]["use_mlflow"]
    )

    # Set up the tracking URI and MLflow client
    tracking_uri = cfg["model"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)

    # Load dataset
    df = pd.read_csv(to_absolute_path(cfg.data.file_path))

    # Features and target selection
    features = cfg.data.features
    target = cfg.data.target

    df = df[features + [target]].dropna()
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  

    X = df[features]
    y = df[target]

    # Scaling data
    if cfg.scaler.method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state)

    # Start MLflow experiment
    mlflow.set_experiment("titanic-mlflow-experiment")
    with mlflow.start_run():
        # ---------------- Logistic Regression ----------------
        log_reg = LogisticRegression(C=cfg.model.log_reg.C, max_iter=cfg.model.log_reg.max_iter)
        log_reg.fit(X_train, y_train)
        y_pred_lr = log_reg.predict(X_val)
        acc_lr = accuracy_score(y_val, y_pred_lr)

        print("\nLogistic Regression Results:")
        print("Accuracy:", acc_lr)
        print(classification_report(y_val, y_pred_lr))

        # ---------------- Random Forest ----------------
        rf = RandomForestClassifier(n_estimators=cfg.model.rf.n_estimators, random_state=cfg.model.rf.random_state)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_val)
        acc_rf = accuracy_score(y_val, y_pred_rf)

        print("\nRandom Forest Results:")
        print("Accuracy:", acc_rf)
        print(classification_report(y_val, y_pred_rf))

        # ---------------- MLflow Logging ----------------
        # Log hyperparameters
        mlflow.log_param("rf_n_estimators", cfg.model.rf.n_estimators)
        mlflow.log_param("rf_random_state", cfg.model.rf.random_state)
        mlflow.log_param("logreg_C", cfg.model.log_reg.C)
        mlflow.log_param("logreg_max_iter", cfg.model.log_reg.max_iter)

        # Log metrics
        mlflow.log_metric("accuracy_logistic_regression", acc_lr)
        mlflow.log_metric("accuracy_random_forest", acc_rf)

        # Log model (choose the best to log)
        best_model = rf if acc_rf > acc_lr else log_reg

        model_dir = to_absolute_path("models")
        if not os.path.exists(model_dir):
            print(f"Creating directory: {model_dir}")
            os.makedirs(model_dir)
        else:
            print(f"Model directory exists: {model_dir}")

        model_path = os.path.join(model_dir, "model.pkl")
        print(f"Saving model to: {model_path}")
        joblib.dump(best_model, model_path)
        signature = mlflow.models.infer_signature(X_val, y_pred_rf)
        # Log the best model with MLflow
        mlflow.sklearn.log_model(best_model, artifact_path= cfg.model.model_path ,registered_model_name=cfg.model.model_name,signature=signature)

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{cfg.model.model_path}"
        model_details = mlflow.register_model(
            model_uri=model_uri, 
            name=cfg.model.model_name
        )

    print("Experiment Logged")
    client = setup_mlflow(cfg.model.tracking_uri)
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage="production",
    )

    # Optional: PyTorch Lightning Logger (if you want to use it for later)
    logger = loggers.TensorBoardLogger("tb_logs", name="titanic_model")
    trainer = pl.Trainer(logger=logger)

if __name__ == "__main__":
    main()
