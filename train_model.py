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

def setup_mlflow(tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
    return client


dagshub.auth.add_app_token(token="505fecb8c64c86e969ef3556e9a062da6d7d35ac")

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
   
    dagshub.init(
        repo_owner=cfg["model"]["repo_owner"],
        repo_name=cfg["model"]["repo_name"],
        mlflow=cfg["model"]["use_mlflow"]
    )

    tracking_uri = cfg["model"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)

    
    df = pd.read_csv(to_absolute_path(cfg.data.file_path))
    df = df[cfg.data.features + [cfg.data.target]].dropna()

    
    X = df[cfg.data.features]
    y = df[cfg.data.target]

    
    scaler = StandardScaler() if cfg.scaler.method == 'standard' else MinMaxScaler()

    
    categorical_features = ['Sex', 'Embarked', 'Pclass']
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']

    
    preprocessor = ColumnTransformer(transformers=[
        ('num', scaler, numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state)

   
    mlflow.set_experiment("titanic-mlflow-experiment")
    with mlflow.start_run():
        
        log_reg_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(C=cfg.model.log_reg.C, max_iter=cfg.model.log_reg.max_iter))
        ])

        rf_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=cfg.model.rf.n_estimators, random_state=cfg.model.rf.random_state))
        ])

        
        log_reg_pipeline.fit(X_train, y_train)
        y_pred_lr = log_reg_pipeline.predict(X_val)
        acc_lr = accuracy_score(y_val, y_pred_lr)
        print("\nLogistic Regression Results:")
        print("Accuracy:", acc_lr)
        print(classification_report(y_val, y_pred_lr))

        
        rf_pipeline.fit(X_train, y_train)
        y_pred_rf = rf_pipeline.predict(X_val)
        acc_rf = accuracy_score(y_val, y_pred_rf)
        print("\nRandom Forest Results:")
        print("Accuracy:", acc_rf)
        print(classification_report(y_val, y_pred_rf))

        
        mlflow.log_param("rf_n_estimators", cfg.model.rf.n_estimators)
        mlflow.log_param("rf_random_state", cfg.model.rf.random_state)
        mlflow.log_param("logreg_C", cfg.model.log_reg.C)
        mlflow.log_param("logreg_max_iter", cfg.model.log_reg.max_iter)
        mlflow.log_metric("accuracy_logistic_regression", acc_lr)
        mlflow.log_metric("accuracy_random_forest", acc_rf)

        
        best_pipeline = rf_pipeline if acc_rf > acc_lr else log_reg_pipeline

        model_dir = to_absolute_path("models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(best_pipeline, model_path)
        print(f" Full pipeline saved to: {model_path}")

        
        signature = mlflow.models.infer_signature(X_val, y_pred_rf)
        mlflow.sklearn.log_model(best_pipeline, artifact_path=cfg.model.model_path,
                                 registered_model_name=cfg.model.model_name,
                                 signature=signature)

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{cfg.model.model_path}"
        model_details = mlflow.register_model(
            model_uri=model_uri, 
            name=cfg.model.model_name
        )

        client.transition_model_version_stage(
            name=model_details.name,
            version=model_details.version,
            stage="production",
        )

    # Optional: TensorBoard Logger
    logger = loggers.TensorBoardLogger("tb_logs", name="titanic_model")
    trainer = pl.Trainer(logger=logger)

if __name__ == "__main__":
    main()
