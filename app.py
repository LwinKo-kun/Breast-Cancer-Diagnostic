import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

from src.project_paths import MODEL_PATH


class BreastCancerDiagnosticGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Breast Cancer Diagnostic Studio (v2.0 Desktop)")
        self.root.geometry("980x720")
        self.root.minsize(900, 650)

        # Load Model & Dataset
        self.load_resources()

        # Build UI Components
        self.setup_styles()
        self.build_layout()

        # Pre-fill with first specimen
        self.load_specimen_by_index(0)

    def load_resources(self):
        if not MODEL_PATH.exists():
            messagebox.showerror(
                "Model Not Found",
                f"Artifact missing at '{MODEL_PATH}'.\nRun 'python -m src.train 400' first.",
            )
            self.root.destroy()
            return

        self.artifact = joblib.load(MODEL_PATH)
        self.pipeline = self.artifact["pipeline"]
        self.scaler = self.pipeline.named_steps["scaler"]
        self.knn = self.pipeline.named_steps["knn"]
        self.feature_names = self.artifact["feature_names"]
        self.target_names = self.artifact["target_names"]

        self.dataset = load_breast_cancer(as_frame=True)
        self.X_data = self.dataset.data
        self.y_data = self.dataset.target

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.style.configure("Header.TLabel", font=("Helvetica", 14, "bold"))
        self.style.configure("SubHeader.TLabel", font=("Helvetica", 11, "bold"))
        self.style.configure("Result.TLabel", font=("Helvetica", 15, "bold"))
        self.style.configure("Meta.TLabel", font=("Helvetica", 10))

    def build_layout(self):
        # 1. Top Header Banner
        header_frame = ttk.Frame(self.root, padding="12 10")
        header_frame.pack(fill=tk.X)

        title = ttk.Label(
            header_frame,
            text="🔬 Histological Diagnostic Studio",
            style="Header.TLabel",
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            header_frame,
            text="Fine-Needle Aspirate (FNA) Morphological Analysis & KNN Nearest Neighbors",
            style="Meta.TLabel",
        )
        subtitle.pack(anchor="w")

        ttk.Separator(self.root, orient="horizontal").pack(fill=tk.X, padx=10, pady=5)

        # 2. Main Content Split (Left: Inputs, Right: Predictions & Neighbors)
        content_frame = ttk.Frame(self.root, padding="10")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left Panel (Controls & Sliders)
        left_panel = ttk.LabelFrame(content_frame, text=" Specimen Inputs ", padding="10")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # Quick Loader Controls
        quick_frame = ttk.Frame(left_panel)
        quick_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(quick_frame, text="Specimen Index (0-568):").pack(side=tk.LEFT, padx=(0, 5))
        self.sample_idx_var = tk.IntVar(value=14)
        spinbox = ttk.Spinbox(
            quick_frame,
            from_=0,
            to=len(self.X_data) - 1,
            textvariable=self.sample_idx_var,
            width=6,
            command=self.on_spinbox_change,
        )
        spinbox.pack(side=tk.LEFT, padx=(0, 8))

        btn_load = ttk.Button(quick_frame, text="Load Specimen", command=self.on_spinbox_change)
        btn_load.pack(side=tk.LEFT, padx=(0, 5))

        btn_random = ttk.Button(quick_frame, text="🎲 Random", command=self.load_random_specimen)
        btn_random.pack(side=tk.LEFT)

        # Scrollable Feature List for Interactive Adjustment
        canvas = tk.Canvas(left_panel, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=canvas.yview)
        self.scroll_window = ttk.Frame(canvas)

        self.scroll_window.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scroll_window, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.input_vars = {}
        for feat in self.feature_names:
            row_frame = ttk.Frame(self.scroll_window, padding="2")
            row_frame.pack(fill=tk.X, expand=True)

            lbl = ttk.Label(row_frame, text=feat.title(), width=24, anchor="w")
            lbl.pack(side=tk.LEFT)

            var = tk.DoubleVar()
            self.input_vars[feat] = var

            ent = ttk.Entry(row_frame, textvariable=var, width=12)
            ent.pack(side=tk.RIGHT, padx=5)

        # Right Panel (Diagnosis, Confidences, and Nearest Neighbors)
        right_panel = ttk.LabelFrame(content_frame, text=" Diagnostic Outcome ", padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(6, 0))

        # Diagnosis Result Card
        self.card_frame = tk.Frame(right_panel, bg="#f0f0f0", bd=2, relief=tk.GROOVE, padx=12, pady=12)
        self.card_frame.pack(fill=tk.X, pady=(0, 10))

        self.lbl_diagnosis = tk.Label(
            self.card_frame, text="DIAGNOSIS: ---", font=("Helvetica", 14, "bold"), bg="#f0f0f0"
        )
        self.lbl_diagnosis.pack(anchor="w")

        self.lbl_confidence = tk.Label(
            self.card_frame, text="Confidence: ---", font=("Helvetica", 11), bg="#f0f0f0"
        )
        self.lbl_confidence.pack(anchor="w", pady=(4, 0))

        self.lbl_ground_truth = tk.Label(
            self.card_frame, text="Ground Truth: ---", font=("Helvetica", 10, "italic"), bg="#f0f0f0"
        )
        self.lbl_ground_truth.pack(anchor="w", pady=(2, 0))

        self.conf_bar = ttk.Progressbar(right_panel, orient="horizontal", mode="determinate")
        self.conf_bar.pack(fill=tk.X, pady=(0, 15))

        # Predict Button
        btn_predict = tk.Button(
            right_panel,
            text="Run Prediction",
            bg="#2563eb",
            fg="white",
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT,
            padx=10,
            pady=6,
            cursor="hand2",
            command=self.run_prediction,
        )
        btn_predict.pack(fill=tk.X, pady=(0, 15))

        # Nearest Neighbors Table
        ttk.Label(right_panel, text="Nearest Diagnostic Neighbors (k-NN):", style="SubHeader.TLabel").pack(
            anchor="w", pady=(0, 4)
        )

        cols = ("rank", "index", "diagnosis", "distance")
        self.tree = ttk.Treeview(right_panel, columns=cols, show="headings", height=8)
        self.tree.heading("rank", text="Rank")
        self.tree.heading("index", text="Sample #")
        self.tree.heading("diagnosis", text="Class")
        self.tree.heading("distance", text="Euclidean Dist")

        self.tree.column("rank", width=50, anchor="center")
        self.tree.column("index", width=80, anchor="center")
        self.tree.column("diagnosis", width=110, anchor="center")
        self.tree.column("distance", width=110, anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True)

    def on_spinbox_change(self):
        idx = self.sample_idx_var.get()
        self.load_specimen_by_index(idx)

    def load_random_specimen(self):
        rand_idx = int(np.random.randint(0, len(self.X_data)))
        self.sample_idx_var.set(rand_idx)
        self.load_specimen_by_index(rand_idx)

    def load_specimen_by_index(self, idx: int):
        if idx < 0 or idx >= len(self.X_data):
            return
        row = self.X_data.iloc[idx]
        for feat in self.feature_names:
            self.input_vars[feat].set(round(float(row[feat]), 4))

        actual_class = self.target_names[self.y_data.iloc[idx]].upper()
        self.current_ground_truth = actual_class
        self.run_prediction()

    def run_prediction(self):
        # 1. Collect inputs
        try:
            vec = [float(self.input_vars[f].get()) for f in self.feature_names]
        except ValueError:
            messagebox.showerror("Input Error", "All 30 feature inputs must be valid floating numbers.")
            return

        input_arr = np.array(vec, dtype=float).reshape(1, -1)

        # 2. Model Inference
        pred_idx = int(self.pipeline.predict(input_arr)[0])
        pred_label = self.target_names[pred_idx].upper()
        probabilities = self.pipeline.predict_proba(input_arr)[0]
        confidence = probabilities[pred_idx] * 100

        # 3. Nearest Neighbors Extraction
        scaled_input = self.scaler.transform(input_arr)
        distances, indices = self.knn.kneighbors(scaled_input)

        # 4. Update UI Components
        is_malignant = pred_label == "MALIGNANT"
        bg_color = "#fee2e2" if is_malignant else "#dcfce7"
        text_color = "#991b1b" if is_malignant else "#166534"

        self.card_frame.configure(bg=bg_color)
        self.lbl_diagnosis.configure(
            text=f"DIAGNOSIS: {pred_label}", bg=bg_color, fg=text_color
        )
        self.lbl_confidence.configure(
            text=f"Confidence: {confidence:.2f}% ({probabilities[0]*100:.1f}% Malignant | {probabilities[1]*100:.1f}% Benign)",
            bg=bg_color,
            fg=text_color,
        )

        truth_text = f"Dataset Ground Truth: {self.current_ground_truth}"
        if self.current_ground_truth:
            match = pred_label == self.current_ground_truth
            truth_text += f" -> {'MATCH ✅' if match else 'MISMATCH ❌'}"
        self.lbl_ground_truth.configure(text=truth_text, bg=bg_color, fg=text_color)

        self.conf_bar["value"] = confidence

        # Populate Neighbors Table
        for row in self.tree.get_children():
            self.tree.delete(row)

        for rank, (d, idx) in enumerate(zip(distances[0], indices[0]), start=1):
            n_class = self.target_names[self.knn._y[idx]].upper()
            self.tree.insert("", tk.END, values=(f"#{rank}", int(idx), n_class, f"{d:.4f}"))


def main():
    root = tk.Tk()
    app = BreastCancerDiagnosticGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()