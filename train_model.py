# ========================================
# 🧠 Train and Save Conversion Model
# ========================================

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
import datetime

# Error handling for data loading
if not os.path.exists("user_conversion.csv"):
    raise FileNotFoundError("❌ user_conversion.csv not found!")

try:
    df = pd.read_csv("user_conversion.csv")
except Exception as e:
    raise Exception(f"❌ Error loading data: {e}")

# Separate features and target
X = df.drop(["user_id", "is_premium_user"], axis=1)
y = df["is_premium_user"]

# Validate data quality
print(f"📊 Dataset shape: {df.shape}")
print(f"🎯 Class distribution: {y.value_counts().to_dict()}")
print(f"❓ Missing values: {X.isnull().sum().sum()}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define preprocessing
numeric_features = ["days_active", "avg_session_duration", "num_logins",
                    "features_used", "email_open_rate", "ad_clicks"]
categorical_features = ["region", "device_type"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5, 
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ))
])

# Train model
print("🚀 Training model...")
model.fit(X_train, y_train)
print("✅ Model training complete!")

# Evaluate model performance
print("📈 Evaluating model performance...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

test_accuracy = model.score(X_test, y_test)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"🎯 Test Accuracy: {test_accuracy:.3f}")
print(f"📊 ROC-AUC Score: {roc_auc:.3f}")

# Cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"🔄 CV ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model with metadata
model_info = {
    'model': model,
    'feature_names': numeric_features + categorical_features,
    'training_date': datetime.datetime.now().isoformat(),
    'test_accuracy': test_accuracy,
    'roc_auc': roc_auc,
    'cv_score_mean': cv_scores.mean()
}
joblib.dump(model_info, "conversion_model_new.pkl")
print("💾 Model saved with metadata as conversion_model_new.pkl")
