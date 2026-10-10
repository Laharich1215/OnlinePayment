"""
Online Payment Fraud Detection
Expected CSV target column: isFraud
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate fraud detection models.")
    parser.add_argument("--data", default="onlinefraud.csv", help="CSV dataset path")
    parser.add_argument("--max-rows", type=int, default=500_000,
                        help="Maximum rows for local runs; use 0 for all rows.")
    parser.add_argument("--test-size", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.data):
        raise FileNotFoundError(
            f"Dataset not found: {args.data}. Put onlinefraud.csv beside this script "
            "or pass --data with the CSV path."
        )

    print(f"Loading dataset: {args.data}")
    data = pd.read_csv(args.data)
    print("Original dataset shape:", data.shape)
    target = "isFraud"
    if target not in data.columns:
        raise ValueError(
            f"Expected target column '{target}', but found: {list(data.columns)}. "
            "If your dataset uses 'Class', change target = 'isFraud' to target = 'Class'."
        )

    data = data.dropna(subset=[target]).copy()
    if args.max_rows > 0 and len(data) > args.max_rows:
        data, _ = train_test_split(
            data, train_size=args.max_rows, random_state=42, stratify=data[target]
        )
        print(f"Using stratified sample: {len(data):,} rows")

    y = data[target].astype(int)
    X = data.drop(columns=[target])
    id_columns = [c for c in X.columns if c.lower() in {
        "nameorig", "namedest", "transactionid", "id"
    }]
    if id_columns:
        X = X.drop(columns=id_columns)
        print("Dropped identifier columns:", id_columns)

    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ], remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    print(f"Training rows: {len(X_train):,}; test rows: {len(X_test):,}")
    print("Class proportions (0=legitimate, 1=fraud):")
    print(y.value_counts(normalize=True))

    rf = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
            class_weight="balanced_subsample"
        )),
    ])
    lr = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(
            max_iter=500, class_weight="balanced", solver="lbfgs"
        )),
    ])
    hybrid = StackingClassifier(
        estimators=[
            ("random_forest", Pipeline([
                ("preprocess", preprocessor),
                ("model", RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=-1,
                    class_weight="balanced_subsample"
                )),
            ])),
            ("logistic_regression", Pipeline([
                ("preprocess", preprocessor),
                ("model", LogisticRegression(
                    max_iter=500, class_weight="balanced", solver="lbfgs"
                )),
            ])),
        ],
        final_estimator=LogisticRegression(max_iter=500, class_weight="balanced"),
        stack_method="predict_proba",
        cv=3,
        n_jobs=1,
    )

    models = {
        "Random Forest": rf,
        "Logistic Regression": lr,
        "Hybrid (RF + LR)": hybrid,
    }
    results = {}
    os.makedirs("outputs", exist_ok=True)
    for name, model in models.items():
        print(f"\n{'=' * 55}\nTraining {name}\n{'=' * 55}")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
        }
        print("\n".join(f"{k}: {v:.4f}" for k, v in results[name].items()))
        print(classification_report(
            y_test, pred, target_names=["Legitimate (0)", "Fraud (1)"],
            zero_division=0
        ))
        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Legitimate", "Fraud"],
                    yticklabels=["Legitimate", "Fraud"])
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        plt.savefig(f"outputs/{safe_name}_confusion_matrix.png", dpi=150)
        plt.close()

    metrics = pd.DataFrame(results).T
    metrics.to_csv("outputs/model_metrics.csv", index_label="Model")
    print("\nModel comparison:\n", metrics.round(4))
    metrics["Accuracy"].plot(kind="bar", figsize=(8, 5), color="steelblue")
    plt.title("Comparison of Model Accuracies")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/model_accuracy_comparison.png", dpi=150)
    plt.close()
    print("\nBest model by F1-score:", metrics["F1-score"].idxmax())
    print("Results saved in the outputs/ folder.")
    print("For imbalanced fraud data, check fraud-class precision, recall, and F1-score—not accuracy alone.")


if __name__ == "__main__":
    main()
