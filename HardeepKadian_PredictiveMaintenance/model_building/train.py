# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np

from datasets import load_dataset

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from huggingface_hub import HfApi

# ==========================================
# LOAD TRAIN AND TEST DATASETS
# ==========================================

print("Loading train dataset...")

train_dataset = load_dataset(
    "Kaddy4613/Predictive-Maintenance-Train"
)

train_df = train_dataset["train"].to_pandas()

print("Loading test dataset...")

test_dataset = load_dataset(
    "Kaddy4613/Predictive-Maintenance-Test"
)

test_df = test_dataset["train"].to_pandas()

print("Datasets loaded successfully.")

# ==========================================
# SPLIT FEATURES AND TARGET
# ==========================================

X_train = train_df.drop("Engine Condition", axis=1)

y_train = train_df["Engine Condition"]

X_test = test_df.drop("Engine Condition", axis=1)

y_test = test_df["Engine Condition"]

# ==========================================
# DEFINE MODELS
# ==========================================

models = {

    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "classifier__max_depth": [3, 5, 10],
            "classifier__min_samples_split": [2, 5]
        }
    },

    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [5, 10]
        }
    },

    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [50, 100],
            "classifier__learning_rate": [0.01, 0.1]
        }
    },

    "XGBoost": {
        "model": XGBClassifier(
            random_state=42,
            eval_metric='logloss'
        ),
        "params": {
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.01, 0.1]
        }
    }
}

# ==========================================
# TRACK BEST MODEL
# ==========================================

best_recall = 0

best_f1 = 0

best_model = None

best_model_name = None

# ==========================================
# TRAIN MODELS
# ==========================================

for model_name, config in models.items():

    print(f"\nTraining {model_name}...")

    # Create pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", config["model"])
    ])

    # GridSearch
    grid_search = GridSearchCV(
        pipeline,
        config["params"],
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    # ======================================
    # START MLFLOW RUN
    # ======================================

    with mlflow.start_run(run_name=model_name):

        # Train
        grid_search.fit(X_train, y_train)

        # Best estimator
        model = grid_search.best_estimator_

        # Predictions
        y_pred = model.predict(X_test)

        # Prediction probabilities
        y_prob = model.predict_proba(X_test)[:,1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(y_test, y_pred)

        recall = recall_score(y_test, y_pred)

        f1 = f1_score(y_test, y_pred)

        roc_auc = roc_auc_score(y_test, y_prob)

        # ==================================
        # LOG PARAMETERS
        # ==================================

        mlflow.log_params(
            grid_search.best_params_
        )

        # ==================================
        # LOG METRICS
        # ==================================

        mlflow.log_metric("accuracy", accuracy)

        mlflow.log_metric("precision", precision)

        mlflow.log_metric("recall", recall)

        mlflow.log_metric("f1_score", f1)

        mlflow.log_metric("roc_auc", roc_auc)

        # ==================================
        # LOG MODEL
        # ==================================

        mlflow.sklearn.log_model(
            model,
            model_name
        )

        # ==================================
        # PRINT RESULTS
        # ==================================

        print(f"Best Parameters: {grid_search.best_params_}")

        print(f"Accuracy: {accuracy:.4f}")

        print(f"Precision: {precision:.4f}")

        print(f"Recall: {recall:.4f}")

        print(f"F1 Score: {f1:.4f}")

        print(f"ROC-AUC Score: {roc_auc:.4f}")

        # ==================================
        # CONFUSION MATRIX
        # ==================================

        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6,5))

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues'
        )

        plt.xlabel("Predicted")

        plt.ylabel("Actual")

        plt.title(f"Confusion Matrix - {model_name}")

        plt.show()

        # ==================================
        # FEATURE IMPORTANCE
        # ==================================

        # Only for tree-based models

        if model_name in [
            "DecisionTree",
            "RandomForest",
            "GradientBoosting",
            "XGBoost"
        ]:

            importance = model.named_steps[
                'classifier'
            ].feature_importances_

            feature_names = X_train.columns

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            plt.figure(figsize=(10,6))

            sns.barplot(
                data=importance_df,
                x="Importance",
                y="Feature"
            )

            plt.title(
                f"Feature Importance - {model_name}"
            )

            plt.show()

        # ==================================
        # TRACK BEST MODEL
        # ==================================

        # Priority:
        # 1. Higher Recall
        # 2. Higher F1 Score

        if (
            recall > best_recall
            or
            (
                recall == best_recall
                and f1 > best_f1
            )
        ):

            best_recall = recall

            best_f1 = f1

            best_model = model

            best_model_name = model_name

# ==========================================
# SAVE BEST MODEL
# ==========================================

print(f"\nBest Model: {best_model_name}")

joblib.dump(
    best_model,
    "HardeepKadian_PredictiveMaintenance/model_building/best_model.pkl"
)

print("Best model saved successfully.")

# ==========================================
# UPLOAD MODEL TO HUGGING FACE
# ==========================================

api = HfApi()

repo_id = "Kaddy4613/Predictive-Maintenance-Model"

api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True
)

api.upload_file(
    path_or_fileobj="HardeepKadian_PredictiveMaintenance/model_building/best_model.pkl",
    path_in_repo="best_model.pkl",
    repo_id=repo_id,
    repo_type="model"
)

print("Best model uploaded to Hugging Face.")
