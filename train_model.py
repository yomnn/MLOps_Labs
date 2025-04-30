import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pytorch_lightning as pl
from pytorch_lightning import loggers
from hydra.utils import to_absolute_path


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
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state)

    # Logistic Regression
    log_reg = LogisticRegression(C=cfg.model.log_reg.C, max_iter=cfg.model.log_reg.max_iter)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_val)

    print("Logistic Regression Results:")
    print("Accuracy:", accuracy_score(y_val, y_pred_lr))
    print(classification_report(y_val, y_pred_lr))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=cfg.model.rf.n_estimators, random_state=cfg.model.rf.random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_val)

    print("Random Forest Results:")
    print("Accuracy:", accuracy_score(y_val, y_pred_rf))
    print(classification_report(y_val, y_pred_rf))

    # Use Lightning for logging (optional)
    logger = loggers.TensorBoardLogger("tb_logs", name="titanic_model")
    trainer = pl.Trainer(logger=logger)

if __name__ == "__main__":
    main()
