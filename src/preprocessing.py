import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    print("📂 Loading dataset...")

    df = pd.read_csv(path)
    print(f"✅ Rows: {len(df)}, Columns: {len(df.columns)}")

    # 🔥 Optional: sample for speed
    df = df.sample(50000, random_state=42)

    return df


def preprocess(df):
    print("🧹 Cleaning data...")

    # Remove spaces in column names
    df.columns = df.columns.str.strip()

    print("🔍 Columns found:")
    print(df.columns.tolist())

    # 🔥 Detect label column automatically
    label_col = None
    for col in df.columns:
        if col.lower() == "label":
            label_col = col
            break

    if label_col is None:
        raise Exception("❌ No label column found")

    # Rename to standard name
    df.rename(columns={label_col: "label"}, inplace=True)

    # 🔥 Remove duplicates (important for CICIDS)
    df.drop_duplicates(inplace=True)

    # 🔥 Handle infinity values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop missing values
    df.dropna(inplace=True)

    # 🔥 Encode categorical features
    for col in df.select_dtypes(include='object').columns:
        if col != "label":
            df[col] = LabelEncoder().fit_transform(df[col])

    # 🔥 Encode labels
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])

    # 🔥 Save label encoder (for Flask)
    joblib.dump(label_encoder, "models/label_encoder.pkl")

    # 🔍 Sanity check (class distribution)
    print("\n📊 Class Distribution:")
    print(df["label"].value_counts(normalize=True))

    # Split features and target
    X = df.drop("label", axis=1)
    y = df["label"]

    print("📊 Splitting data...")

    # 🔥 Stratified split (important)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return (X_train, X_test, y_train, y_test), label_encoder