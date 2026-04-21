def generate_alert(label):
    if label != "BENIGN":
        print(f"🚨 ALERT: {label} attack detected!")
    else:
        print("✅ Normal traffic")