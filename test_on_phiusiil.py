import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from feature_extractor import extract_features_from_list


PHIUSIIL_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH = "models/best_url_model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"


def standardize_phiusiil_label(value):
    """
    Adjust this only if PhiUSIIL uses reversed meaning.
    Current assumption:
        1 = phishing
        0 = legitimate
    """
    if pd.isna(value):
        return None

    try:
        numeric_value = float(value)
        if numeric_value in [0.0, 1.0]:
            return int(numeric_value)
    except (ValueError, TypeError):
        pass

    return None


def main():
    print("Loading PhiUSIIL dataset...")
    df = pd.read_csv(PHIUSIIL_PATH)

    print(f"Original shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())

    if "URL" not in df.columns or "label" not in df.columns:
        raise ValueError("PhiUSIIL dataset must contain 'URL' and 'label' columns.")

    print("Unique label values before cleaning:")
    print(df["label"].value_counts(dropna=False).head(20))

    df = df[["URL", "label"]].copy()
    df["label"] = df["label"].apply(standardize_phiusiil_label)
    df = df.dropna(subset=["URL", "label"]).copy()

    if df.empty:
        raise ValueError("No rows left after cleaning PhiUSIIL labels.")

    df["label"] = df["label"].astype(int)

    print(f"Shape after cleaning: {df.shape}")
    print("Label distribution:")
    print(df["label"].value_counts(dropna=False))

    print("Extracting URL-based features...")
    feature_df = extract_features_from_list(df["URL"].tolist())

    feature_columns = joblib.load(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)

    X_test = feature_df[feature_columns]
    y_test = df["label"]

    print(f"Test shape: {X_test.shape}")

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    print("\n=== External Test on PhiUSIIL ===")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
    print("F1-score :", f1_score(y_test, y_pred, zero_division=0))

    if y_prob is not None:
        print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()