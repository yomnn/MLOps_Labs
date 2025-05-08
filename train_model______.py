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
import mlflow
import os


os.environ["MLFLOW_TRACKING_USERNAME"] = "yomna.wael.mu"
os.environ["MLFLOW_TRACKING_PASSWOR"] = "2609ca7050da4117b26a3b50e3fe55f25f62db30en"



mlflow.set_tracking_uri("https://dagshub.com/yomna.wael.mu/MLOps_Labs.mlflow")
mlflow.set_experiment("titanic-mlflow-experiment") 


# Hydra configuration
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
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


        # Log the best model with MLflow
        mlflow.sklearn.log_model(best_model, "best_model")
    print("Experiment Logged")

    # Optional: PyTorch Lightning Logger (if you want to use it for later)
    # You can remove this part if you don't plan on using PyTorch Lightning here
    logger = loggers.TensorBoardLogger("tb_logs", name="titanic_model")
    trainer = pl.Trainer(logger=logger)

if __name__ == "__main__":
    main()
