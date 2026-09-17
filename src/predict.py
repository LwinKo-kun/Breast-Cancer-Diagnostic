import random
import requests
from sklearn.datasets import load_breast_cancer

API_URL = "http://127.0.0.1:8000/predict"


def get_prediction(features: list, true_label: str = None):
    try:
        res = requests.post(API_URL, json={"features": features})
        if res.status_code == 200:
            data = res.json()
            print(f"\nAPI Prediction: {data['prediction'].upper()}")
            print(f"Confidence: {data['confidence_pct']}%")
            if true_label:
                is_correct = data["prediction"].lower() == true_label.lower()
                print(
                    f"Ground Truth: {true_label.upper()} -> Match: {'✅' if is_correct else '❌'}"
                )
            print(
                f"Nearest Neighbors Distances: {data['nearest_neighbors_distances'][:3]}\n"
            )
        else:
            print(f"Error HTTP {res.status_code}: {res.json()}")
    except requests.exceptions.ConnectionError:
        print(
            "Error: Could not connect to FastAPI. Ensure 'uvicorn main:app --reload' is running."
        )


def main():
    print("Select prediction source:")
    print("  [1] Random Patient from Dataset")
    print("  [2] Custom Feature Vector (30 inputs)")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        data = load_breast_cancer()
        idx = random.randint(0, len(data.data) - 1)
        sample = data.data[idx].tolist()
        label = data.target_names[data.target[idx]]
        print(f"\n--- Testing Specimen #{idx} ---")
        get_prediction(sample, true_label=label)

    elif choice == "2":
        print(
            "\nEnter 30 comma-separated values (or press Enter for malignant benchmark):"
        )
        user_input = input("Features: ").strip()
        if not user_input:
            sample = [
                17.99,
                10.38,
                122.8,
                1001.0,
                0.1184,
                0.2776,
                0.3001,
                0.1471,
                0.2419,
                0.07871,
                1.095,
                0.9053,
                8.589,
                153.4,
                0.006399,
                0.04904,
                0.05373,
                0.01587,
                0.03003,
                0.006193,
                25.38,
                17.33,
                184.6,
                2019.0,
                0.1622,
                0.6656,
                0.7119,
                0.2654,
                0.4601,
                0.1189,
            ]
        else:
            try:
                sample = [float(x.strip()) for x in user_input.split(",")]
            except ValueError:
                print("Invalid input format. Values must be numbers.")
                return

        print(f"\n--- Testing Vector ({len(sample)} features) ---")
        get_prediction(sample)
    else:
        print("Invalid option selected.")


if __name__ == "__main__":
    main()