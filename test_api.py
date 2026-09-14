import time
import random
import requests
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

API_URL = "http://127.0.0.1:8000/predict"

def run_tests():
    # 1. Check if API is running
    try:
        requests.get("http://127.0.0.1:8000/health")
    except requests.exceptions.ConnectionError:
        print("Error: FastAPI is not running. Start it with 'uvicorn main:app --reload'")
        return

    # 2. Isolate the exact same test partition the model NEVER saw during training
    data = load_breast_cancer()
    _, X_test, _, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    print("--- Live API Benchmark: Unseen Test Data ---\n")
    
    # Pick 7 random samples purely from the unseen test partition
    random.seed(99)
    test_indices = random.sample(range(len(X_test)), 7)

    correct = 0
    for i, idx in enumerate(test_indices, 1):
        sample_features = X_test[idx].tolist()
        expected_label = data.target_names[y_test[idx]]

        # Query the FastAPI service
        start_time = time.perf_counter()
        res = requests.post(API_URL, json={"features": sample_features})
        latency = (time.perf_counter() - start_time) * 1000
        
        if res.status_code == 200:
            payload = res.json()
            predicted = payload["prediction"]
            confidence = payload["confidence_pct"]
            
            # Check accuracy
            is_match = predicted == expected_label
            if is_match:
                correct += 1
                
            icon = "✅ Match" if is_match else "❌ Error"
            
            print(f"Sample #{i} | True Label: {expected_label.upper():<9} | API Says: {predicted.upper():<9} | {icon:<7} | Latency: {latency:.1f}ms")
        else:
            print(f"Sample #{i} | API Request Failed: HTTP {res.status_code}")

    print(f"\nTotal Batch Accuracy: {correct}/7 correct ({(correct/7)*100:.1f}%)")

if __name__ == "__main__":
    run_tests()