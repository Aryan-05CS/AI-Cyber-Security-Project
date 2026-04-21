from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
import pandas as pd
import joblib
import os


def train_model(X_train, y_train):
    print("🤖 Training model...")

    import os
    import joblib
    os.makedirs("models", exist_ok=True)
    
    # 🔥 SAVE COLUMNS (THIS IS MISSING
    joblib.dump(list(X_train.columns), "models/columns.pkl")

    # Ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # 🔥 Step 1: Save training column names (IMPORTANT)
    joblib.dump(list(X_train.columns), "models/columns.pkl")

    # 🔥 Step 2: Base model for feature selection
    base_model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=42
    )
    base_model.fit(X_train, y_train)

    # 🔥 Step 3: Feature selection
    selector = SelectFromModel(base_model, threshold="median")
    X_train_selected = selector.transform(X_train)

    print(f"Original features: {X_train.shape[1]}")
    print(f"Selected features: {X_train_selected.shape[1]}")

    # 🔥 Step 4: Final model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train_selected, y_train)

    # 🔥 Step 5: Evaluation
    train_acc = model.score(X_train_selected, y_train)
    print(f"📊 Train Accuracy: {train_acc:.4f}")

    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, n_jobs=-1)
    print(f"📊 Cross-validation Accuracy: {cv_scores.mean():.4f}")

    # 🔥 Step 6: Feature importance (optional but good)
    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': base_model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print("\n🔝 Top 10 Features:")
    print(feature_importance_df.head(10))

    # 🔥 Step 7: Save everything
    joblib.dump(model, "models/model.pkl")
    joblib.dump(selector, "models/selector.pkl")

    print("💾 Model + selector + columns saved")

    return model, selector