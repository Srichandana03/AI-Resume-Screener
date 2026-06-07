import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt

# Features used for the prediction models
FEATURE_COLUMNS = ['similarity_score', 'matched_skills_count', 'missing_skills_count', 'experience_years']

def train_and_evaluate(data_path: str, models_dir: str = "models") -> dict:
    """
    Loads candidate features, trains Logistic Regression, Random Forest, and XGBoost.
    Evaluates them and saves models and metrics.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at {data_path}")
        
    os.makedirs(models_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    X = df[FEATURE_COLUMNS]
    y = df['shortlisted']
    
    # Stratified split to keep label proportions same
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save training datasets for SHAP references
    X_train.to_csv(os.path.join(models_dir, "X_train.csv"), index=False)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6),
        "XGBoost": XGBClassifier(random_state=42, n_estimators=100, max_depth=4, eval_metric='logloss')
    }
    
    metrics_report = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        # Predict labels and probabilities
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred
            
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.5
            
        metrics_report[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        }
        
        trained_models[name] = model
        
        # Save individual model
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(models_dir, f"{safe_name}_model.joblib"))
        
    # Find the best model based on F1 Score
    best_name = max(metrics_report.keys(), key=lambda k: metrics_report[k]["F1 Score"])
    best_model = trained_models[best_name]
    
    # Save best model references
    joblib.dump(best_model, os.path.join(models_dir, "best_model.joblib"))
    with open(os.path.join(models_dir, "best_model_name.txt"), "w") as f:
        f.write(best_name)
        
    # Save the metrics to a CSV for easily reading inside app
    metrics_df = pd.DataFrame(metrics_report).T
    metrics_df.to_csv(os.path.join(models_dir, "model_comparison.csv"))
    
    return {
        "metrics": metrics_report,
        "best_model_name": best_name,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

def load_best_model(models_dir: str = "models"):
    """
    Loads the best trained model and model name from disk.
    """
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    name_path = os.path.join(models_dir, "best_model_name.txt")
    
    if not os.path.exists(best_model_path):
        return None, "None"
        
    model = joblib.load(best_model_path)
    name = "Best Model"
    if os.path.exists(name_path):
        with open(name_path, "r") as f:
            name = f.read().strip()
            
    return model, name

def get_shap_waterfall_plot(model, X_train, sample_features: list):
    """
    Generates a SHAP waterfall (or bar) plot for a single prediction and returns the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    try:
        sample_df = pd.DataFrame([sample_features], columns=FEATURE_COLUMNS)
        
        # Use TreeExplainer for Tree-based models, Explainer for others
        model_type = str(type(model))
        if "RandomForest" in model_type or "XGB" in model_type:
            explainer = shap.TreeExplainer(model)
            # TreeExplainer might yield shap_values of shape (n_samples, n_features, n_classes) or (n_samples, n_features)
            shap_values = explainer(sample_df)
            
            # For RF multiclass/binary we might need to select class 1
            if len(shap_values.shape) == 3: # shape is (samples, features, classes)
                # Select positive class
                shap_val_single = shap.Explanation(
                    values=shap_values.values[0, :, 1],
                    base_values=shap_values.base_values[0, 1],
                    data=shap_values.data[0],
                    feature_names=FEATURE_COLUMNS
                )
            else:
                shap_val_single = shap_values[0]
        else:
            # Linear / Logistic Regression
            # Train explainer on background X_train
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(sample_df)
            shap_val_single = shap_values[0]
            
        shap.plots.waterfall(shap_val_single, show=False)
        plt.tight_layout()
    except Exception as e:
        plt.clf()
        plt.text(0.5, 0.5, f"SHAP Visualization Unavailable:\n{e}", 
                 ha='center', va='center', fontsize=10, color='red', wrap=True)
        plt.axis('off')
        
    return fig
