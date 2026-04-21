from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
from src.alerts import generate_alert

def detect(model, X_test, y_test, label_encoder, selector):
    print("\n🔍 Running predictions...")

    X_test = selector.transform(X_test)

    print("\n🚨 Real Predictions:")
    for i in range(min(5, len(preds))):
        pred_label = label_encoder.inverse_transform([preds[i]])[0]
        actual_label = label_encoder.inverse_transform([y_test.iloc[i]])[0]
        
        print(f"Actual: {actual_label} | Predicted: {pred_label}")
        
        # ✅ FIXED ALERT
        if pred_label != "BENIGN":
            print(f"🚨 ALERT: {pred_label} attack detected!")
        else:
            print("✅ Normal traffic")

    print("\n📊 RESULTS:")
    print("Accuracy:", accuracy_score(y_test, preds))

    print("\n📄 Classification Report:")
    print(classification_report(y_test, preds))

    print("\n🚨 Alerts:")
    for i in range(min(5, len(preds))):
        print(f"Traffic {i}: ", end="")
        generate_alert(pred_label)

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    plt.show()