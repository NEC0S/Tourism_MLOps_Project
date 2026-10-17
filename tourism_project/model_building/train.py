# Model training with MLflow experiment tracking and model registration on Hugging Face
import os, joblib, mlflow
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_USER = os.getenv("HF_USER", "abhi261204")
token = os.getenv("HF_TOKEN")
data_repo = f"{HF_USER}/tourism-package-prediction"
model_repo = f"{HF_USER}/tourism-package-model"

# MLflow setup: MLflow server if provided, else a local SQLite store
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
mlflow.set_experiment("tourism-package-prediction")

# Load the processed splits (Hub when authenticated, otherwise local files)
base = f"hf://datasets/{data_repo}/" if token else "tourism_project/data/"
opts = {"token": token} if token else None
Xtrain = pd.read_csv(base + "Xtrain.csv", storage_options=opts)
Xtest = pd.read_csv(base + "Xtest.csv", storage_options=opts)
ytrain = pd.read_csv(base + "ytrain.csv", storage_options=opts).squeeze()
ytest = pd.read_csv(base + "ytest.csv", storage_options=opts).squeeze()

numeric = ["Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
           "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
           "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"]
categorical = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]

# Preprocessing: scale numeric, one-hot encode categorical
preprocessor = make_column_transformer(
    (StandardScaler(), numeric),
    (OneHotEncoder(handle_unknown="ignore"), categorical))

# Class imbalance ratio (~4:1) used by XGBoost
ratio = ytrain.value_counts()[0] / ytrain.value_counts()[1]

candidates = {
    "RandomForest": (RandomForestClassifier(random_state=42, class_weight="balanced"),
                     {"randomforestclassifier__n_estimators": [150, 300],
                      "randomforestclassifier__max_depth": [10, None],
                      "randomforestclassifier__min_samples_leaf": [1, 2]}),
    "XGBoost": (XGBClassifier(random_state=42, scale_pos_weight=ratio, eval_metric="logloss"),
                {"xgbclassifier__n_estimators": [150, 300],
                 "xgbclassifier__max_depth": [4, 6],
                 "xgbclassifier__learning_rate": [0.05, 0.1]}),
}

THRESHOLD = 0.45   # slightly below 0.5 to favour recall of potential buyers
best_f1, best_model, best_name = -1, None, None

for name, (clf, grid) in candidates.items():
    with mlflow.start_run(run_name=name):
        pipe = make_pipeline(preprocessor, clf)
        gs = GridSearchCV(pipe, grid, cv=3, scoring="f1", n_jobs=-1)
        gs.fit(Xtrain, ytrain)

        # Log every tried parameter combination as a nested run
        results = gs.cv_results_
        for i in range(len(results["params"])):
            with mlflow.start_run(nested=True, run_name=f"{name}_trial_{i}"):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])

        model = gs.best_estimator_
        mlflow.log_params(gs.best_params_)

        # Evaluate on train and test with the chosen threshold
        m = {}
        for split, X, y in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
            prob = model.predict_proba(X)[:, 1]
            pred = (prob >= THRESHOLD).astype(int)
            m.update({f"{split}_accuracy": accuracy_score(y, pred), f"{split}_precision": precision_score(y, pred),
                      f"{split}_recall": recall_score(y, pred), f"{split}_f1": f1_score(y, pred),
                      f"{split}_roc_auc": roc_auc_score(y, prob)})
        mlflow.log_metrics(m)
        print(name, {k: round(v, 3) for k, v in m.items() if k.startswith("test")})

        if m["test_f1"] > best_f1:
            best_f1, best_model, best_name = m["test_f1"], model, name

print(f"\nBest model: {best_name} (test F1 = {best_f1:.3f})")

# Save and register the best model
joblib.dump(best_model, "tourism_project/model_building/best_tourism_model_v1.joblib")
if token:
    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=model_repo, repo_type="model")
    except RepositoryNotFoundError:
        create_repo(repo_id=model_repo, repo_type="model", private=False, token=token)
    api.upload_file(path_or_fileobj="tourism_project/model_building/best_tourism_model_v1.joblib",
                    path_in_repo="best_tourism_model_v1.joblib", repo_id=model_repo, repo_type="model")
    print("Best model registered on Hugging Face Hub.")
