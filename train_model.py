import hydra
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
from dotenv import load_dotenv

load_dotenv()  # This will load environment variables from the .env file

dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))



@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Authenticate with DagsHub
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))

    # Initialize DagsHub repo with MLflow tracking
    dagshub.init(
        repo_owner=cfg.model.repo_owner,
        repo_name=cfg.model.repo_name,
        mlflow=cfg.model.use_mlflow
    )

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(cfg.model.tracking_uri)
    client = mlflow.client.MlflowClient(tracking_uri=cfg.model.tracking_uri)

    # Load and prepare data
    df = pd.read_csv(to_absolute_path(cfg.data.file_path))
    df = df[cfg.data.features + [cfg.data.target]].dropna()
    X = df[cfg.data.features]
    y = df[cfg.data.target]

    # Preprocessing
    scaler = StandardScaler() if cfg.scaler.method == 'standard' else MinMaxScaler()
    categorical_features = ['Sex', 'Embarked', 'Pclass']
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']

    preprocessor = ColumnTransformer(transformers=[
        ('num', scaler, numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state)

    # Start experiment run
    mlflow.set_experiment("titanic-mlflow-experiment")
    with mlflow.start_run():
        # Logistic Regression
        log_reg_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(C=cfg.model.log_reg.C, max_iter=cfg.model.log_reg.max_iter))
        ])
        log_reg_pipeline.fit(X_train, y_train)
        y_pred_lr = log_reg_pipeline.predict(X_val)
        acc_lr = accuracy_score(y_val, y_pred_lr)

        print("\nLogistic Regression Results:")
        print("Accuracy:", acc_lr)
        print(classification_report(y_val, y_pred_lr))

        # Random Forest
        rf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=cfg.model.rf.n_estimators, random_state=cfg.model.rf.random_state))
        ])
        rf_pipeline.fit(X_train, y_train)
        y_pred_rf = rf_pipeline.predict(X_val)
        acc_rf = accuracy_score(y_val, y_pred_rf)

        print("\nRandom Forest Results:")
        print("Accuracy:", acc_rf)
        print(classification_report(y_val, y_pred_rf))

        # Log parameters and metrics
        mlflow.log_param("rf_n_estimators", cfg.model.rf.n_estimators)
        mlflow.log_param("rf_random_state", cfg.model.rf.random_state)
        mlflow.log_param("logreg_C", cfg.model.log_reg.C)
        mlflow.log_param("logreg_max_iter", cfg.model.log_reg.max_iter)
        mlflow.log_metric("accuracy_logistic_regression", acc_lr)
        mlflow.log_metric("accuracy_random_forest", acc_rf)

        # Choose best model
        if acc_rf > acc_lr:
            best_pipeline = rf_pipeline
            y_pred_best = y_pred_rf
        else:
            best_pipeline = log_reg_pipeline
            y_pred_best = y_pred_lr

        # Save model locally
        model_dir = to_absolute_path("models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(best_pipeline, model_path)
        print(f"Full pipeline saved to: {model_path}")

        # Infer signature from validation data and predictions
        signature = mlflow.models.infer_signature(X_val, y_pred_best)

        # Log and register model
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path=cfg.model.model_path,
            registered_model_name=cfg.model.model_name,
            signature=signature
        )

        print(f"Model logged and registered as: {cfg.model.model_name}")

        # Promote model to Production stage
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{cfg.model.model_path}"
        model_details = mlflow.register_model(
            model_uri=model_uri,
            name=cfg.model.model_name
        )

        client.transition_model_version_stage(
            name=model_details.name,
            version=model_details.version,
            stage="Production"
        )
        print(f"Model {model_details.name} version {model_details.version} promoted to Production.")

    # Optional: TensorBoard Logger
    logger = loggers.TensorBoardLogger("tb_logs", name="titanic_model")
    trainer = pl.Trainer(logger=logger)


if __name__ == "__main__":
    main()

