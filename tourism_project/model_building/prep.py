# Data preparation: load raw data, clean it, split into train/test and register the splits
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

HF_USER = os.getenv("HF_USER", "abhi261204")
repo_id = f"{HF_USER}/tourism-package-prediction"
token = os.getenv("HF_TOKEN")

# Load the raw data from the Hub when a token is available, otherwise use the local copy
DATA_PATH = f"hf://datasets/{repo_id}/tourism.csv" if token else "tourism_project/data/tourism.csv"
df = pd.read_csv(DATA_PATH, storage_options={"token": token} if token else None)
print("Raw data shape:", df.shape)

# 1. Drop the index column and the identifier (no predictive value)
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

# 2. Fix inconsistent category labels
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# 3. Remove duplicates
df = df.drop_duplicates()

# 4. Fill any missing values (median for numeric, mode for categorical)
for col in df.columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median() if df[col].dtype != "object" and str(df[col].dtype) != "str" else df[col].mode()[0])

# 5. Stratified train/test split so both sets keep the 19% purchase rate
target = "ProdTaken"
X, y = df.drop(columns=[target]), df[target]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

for name, obj in [("Xtrain", Xtrain), ("Xtest", Xtest), ("ytrain", ytrain), ("ytest", ytest)]:
    obj.to_csv(f"tourism_project/data/{name}.csv", index=False)
print("Train:", Xtrain.shape, "Test:", Xtest.shape)

# Register the processed splits on the Hub (only when authenticated)
if token:
    api = HfApi(token=token)
    for name in ["Xtrain", "Xtest", "ytrain", "ytest"]:
        api.upload_file(path_or_fileobj=f"tourism_project/data/{name}.csv", path_in_repo=f"{name}.csv",
                        repo_id=repo_id, repo_type="dataset")
    print("Processed splits uploaded to Hugging Face Hub.")
