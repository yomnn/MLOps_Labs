import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# Hydra configuration
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Load dataset
    df = pd.read_csv(to_absolute_path(cfg.data.file_path))

    # Features and target selection
    features = cfg.data.features
    target = cfg.data.target

    # Preprocess data
    df = df[features + [target]].dropna()
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # male=1, female=0

    X = df[features]
    y = df[target]

    # Scaling data
    if cfg.scaler.method == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
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
        joblib.dump(best_model, "models/model.pkl")
        mlflow.sklearn.log_model(best_model, "best_model")

    # Optional: PyTorch Lightning Logger
    logger = loggers.TensorBoardLogger("tb_logs", name="titanic_model")
    trainer = pl.Trainer(logger=logger)

if __name__ == "__main__":
    main()
