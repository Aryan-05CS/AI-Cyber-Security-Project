from src.preprocessing import load_data, preprocess
from src.model import train_model
from src.detect import detect

print("🚀 STARTING PROJECT....")

# Load dataset
print("STEP 1")
data = load_data("data/cicids.csv")

# Preprocess
print("STEP 2")
(X_train, X_test, y_train, y_test), label_encoder = preprocess(data)

# Train model
print("STEP 3")
model = train_model(X_train, y_train)

# Detect
print("STEP 4")
detect(model, X_test, y_test, label_encoder)

print("STEP 5")
print("✅ Done!")