import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from feature_extractor import extract_features_from_list, FEATURE_COLUMNS


# =========================
# CONFIG
# =========================
DATA_PATH = "./data/url_features_extracted1.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_url_model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")


# =========================
# HELPERS
# =========================
def standardize_label(value):
    if pd.isna(value):
        return None

    try:
        numeric_value = float(value)
        if numeric_value in [0.0, 1.0]:
            return int(numeric_value)
    except (ValueError, TypeError):
        pass

    value_str = str(value).strip().lower()
    if value_str in {"phishing", "malicious", "bad"}:
        return 1
    elif value_str in {"legitimate", "benign", "safe"}:
        return 0

    return None

def evaluate_model(name, model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        y_prob = None

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred, zero_division=0),
        "F1-score": f1_score(y_val, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_val, y_prob) if y_prob is not None else None,
    }

    return model, results, y_pred, y_prob


# =========================
# MAIN
# =========================
def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"Original shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    # Keep only rows with URL and label
    if "URL" not in df.columns or "ClassLabel" not in df.columns:
        raise ValueError("Dataset must contain 'URL' and 'ClassLabel' columns.")

    df = df[["URL", "ClassLabel"]].copy()

    # Clean labels
    df["label"] = df["ClassLabel"].apply(standardize_label)
    df = df.dropna(subset=["URL", "label"]).copy()
    df["label"] = df["label"].astype(int)

    print(f"Shape after cleaning: {df.shape}")
    print("Label distribution:")
    print(df["label"].value_counts(dropna=False))

    # Re-extract features from raw URLs
    print("Extracting features from URLs...")
    feature_df = extract_features_from_list(df["URL"].tolist())

    # Merge features with label
    model_df = feature_df.copy()
    model_df["label"] = df["label"].values

    # Final feature matrix
    X = model_df[FEATURE_COLUMNS]
    y = model_df["label"]

    print("Feature sample:")
    print(X.head())

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    # Candidate models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),
    }

    all_results = []
    trained_models = {}

    print("\nTraining models...\n")
    for name, model in models.items():
        trained_model, results, y_pred, y_prob = evaluate_model(
            name, model, X_train, X_val, y_train, y_val
        )

        trained_models[name] = trained_model
        all_results.append(results)

        print(f"=== {name} ===")
        print(results)
        print("Confusion Matrix:")
        print(confusion_matrix(y_val, y_pred))
        print("Classification Report:")
        print(classification_report(y_val, y_pred, zero_division=0))
        print()

    results_df = pd.DataFrame(all_results).sort_values(by="F1-score", ascending=False)
    print("Model comparison:")
    print(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print(f"\nBest model: {best_model_name}")

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(FEATURE_COLUMNS, FEATURES_PATH)

    print(f"Saved best model to: {MODEL_PATH}")
    print(f"Saved feature columns to: {FEATURES_PATH}")


if __name__ == "__main__":
    main()