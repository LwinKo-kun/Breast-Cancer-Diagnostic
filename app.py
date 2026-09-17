import csv
from datetime import datetime
import os
from tkinter import filedialog, messagebox

import customtkinter as ctk
import requests

from src.project_paths import get_data_path

# =========================================================
# CONFIG
# =========================================================

API_URL = "http://127.0.0.1:8000"
HISTORY_FILE = str(get_data_path("prediction_history.csv"))
APP_TITLE = "Breast Cancer Diagnostic Classifier"

# =========================================================
# HIGH-DPI SCALING & THEME
# =========================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Scale all widgets and fonts by 1.4x for high clarity
ctk.set_widget_scaling(1.40)
ctk.set_window_scaling(1.40)

# Colors
BG_COLOR = ("#F1F5F9", "#0B0F19")
CARD_COLOR = ("#FFFFFF", "#1E293B")
CARD_ALT_COLOR = ("#F8FAFC", "#0F172A")
BORDER_COLOR = ("#CBD5E1", "#334155")

TEXT_PRIMARY = ("#0F172A", "#F8FAFC")
TEXT_MUTED = ("#64748B", "#94A3B8")

ACCENT_BLUE = ("#2563EB", "#3B82F6")
COLOR_MALIGNANT = "#F43F5E"
COLOR_BENIGN = "#10B981"
COLOR_WARN = "#F59E0B"

# =========================================================
# FEATURE DEFINITIONS
# =========================================================

FEATURE_GROUPS = {
    "Mean Morphometry": [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "mean compactness",
        "mean concavity",
        "mean concave points",
        "mean symmetry",
        "mean fractal dimension",
    ],
    "Standard Error": [
        "radius error",
        "texture error",
        "perimeter error",
        "area error",
        "smoothness error",
        "compactness error",
        "concavity error",
        "concave points error",
        "symmetry error",
        "fractal dimension error",
    ],
    "Worst / Extreme": [
        "worst radius",
        "worst texture",
        "worst perimeter",
        "worst area",
        "worst smoothness",
        "worst compactness",
        "worst concavity",
        "worst concave points",
        "worst symmetry",
        "worst fractal dimension",
    ],
}

FEATURE_NAMES = []
for group in FEATURE_GROUPS.values():
    FEATURE_NAMES.extend(group)

DEFAULT_SAMPLE = [
    17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
    25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

# =========================================================
# ROOT WINDOW
# =========================================================

app = ctk.CTk()
app.title(APP_TITLE)
app.geometry("1480x960")
app.minsize(1200, 820)
app.configure(fg_color=BG_COLOR)

try:
    app.state("zoomed")
except Exception:
    pass

app.grid_columnconfigure(0, weight=1)
app.grid_rowconfigure(0, weight=0)  # Top Bar
app.grid_rowconfigure(1, weight=0)  # Patient Info
app.grid_rowconfigure(2, weight=1)  # Core Workspace
app.grid_rowconfigure(3, weight=0)  # History Box
app.grid_rowconfigure(4, weight=0)  # Toolbar

# =========================================================
# ROW 0: TOP HEADER BAR
# =========================================================

header = ctk.CTkFrame(
    app, fg_color=CARD_COLOR, corner_radius=0, border_width=1, border_color=BORDER_COLOR
)
header.grid(row=0, column=0, sticky="ew")
header.grid_columnconfigure(0, weight=1)
header.grid_columnconfigure(1, weight=0)
header.grid_columnconfigure(2, weight=0)

left_header = ctk.CTkFrame(header, fg_color="transparent")
left_header.grid(row=0, column=0, padx=25, pady=14, sticky="w")

title_lbl = ctk.CTkLabel(
    left_header,
    text="🔬 Breast Cancer Diagnostic Studio",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(size=26, weight="bold"),
)
title_lbl.pack(anchor="w")

sub_lbl = ctk.CTkLabel(
    left_header,
    text="FNA Morphological Biopsy Classification & Decision Space Analysis",
    text_color=TEXT_MUTED,
    font=ctk.CTkFont(size=14),
)
sub_lbl.pack(anchor="w", pady=(3, 0))

status_pill = ctk.CTkLabel(
    header,
    text="● API Checking...",
    text_color=TEXT_MUTED,
    fg_color=CARD_ALT_COLOR,
    corner_radius=8,
    padx=16,
    pady=8,
    font=ctk.CTkFont(size=13, weight="bold"),
)
status_pill.grid(row=0, column=1, padx=12, pady=12)


def toggle_theme():
    if ctk.get_appearance_mode() == "Dark":
        ctk.set_appearance_mode("Light")
        theme_btn.configure(text="🌙 Dark Mode")
    else:
        ctk.set_appearance_mode("Dark")
        theme_btn.configure(text="☀️ Light Mode")


theme_btn = ctk.CTkButton(
    header,
    text="☀️ Light Mode",
    width=130,
    height=38,
    fg_color=CARD_ALT_COLOR,
    hover_color=BORDER_COLOR,
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(size=13, weight="bold"),
    command=toggle_theme,
)
theme_btn.grid(row=0, column=2, padx=(0, 25), pady=12)

# =========================================================
# ROW 1: PATIENT META STRIP
# =========================================================

meta_frame = ctk.CTkFrame(
    app, fg_color=CARD_COLOR, corner_radius=10, border_width=1, border_color=BORDER_COLOR
)
meta_frame.grid(row=1, column=0, sticky="ew", padx=25, pady=(12, 10))
meta_frame.grid_columnconfigure(1, weight=1)
meta_frame.grid_columnconfigure(3, weight=2)

lbl_p_id = ctk.CTkLabel(
    meta_frame,
    text="Patient Identifier:",
    text_color=TEXT_MUTED,
    font=ctk.CTkFont(size=14, weight="bold"),
)
lbl_p_id.grid(row=0, column=0, padx=(20, 10), pady=12, sticky="w")

ent_patient_id = ctk.CTkEntry(
    meta_frame,
    placeholder_text="e.g. PT-BENCH-01",
    height=40,
    fg_color=CARD_ALT_COLOR,
    border_color=BORDER_COLOR,
    font=ctk.CTkFont(size=14),
)
ent_patient_id.grid(row=0, column=1, padx=(0, 20), pady=12, sticky="ew")

lbl_p_name = ctk.CTkLabel(
    meta_frame,
    text="Patient Name:",
    text_color=TEXT_MUTED,
    font=ctk.CTkFont(size=14, weight="bold"),
)
lbl_p_name.grid(row=0, column=2, padx=(10, 10), pady=12, sticky="w")

ent_patient_name = ctk.CTkEntry(
    meta_frame,
    placeholder_text="Enter full patient reference name",
    height=40,
    fg_color=CARD_ALT_COLOR,
    border_color=BORDER_COLOR,
    font=ctk.CTkFont(size=14),
)
ent_patient_name.grid(row=0, column=3, padx=(0, 20), pady=12, sticky="ew")

# =========================================================
# ROW 2: CORE WORKSPACE
# =========================================================

workspace = ctk.CTkFrame(app, fg_color="transparent")
workspace.grid(row=2, column=0, sticky="nsew", padx=25, pady=0)
workspace.grid_columnconfigure(0, weight=3)
workspace.grid_columnconfigure(1, weight=2)
workspace.grid_rowconfigure(0, weight=1)

# Left Card: Features
input_card = ctk.CTkFrame(
    workspace, fg_color=CARD_COLOR, corner_radius=10, border_width=1, border_color=BORDER_COLOR
)
input_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
input_card.grid_columnconfigure(0, weight=1)
input_card.grid_rowconfigure(1, weight=1)

input_header = ctk.CTkFrame(input_card, fg_color="transparent")
input_header.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 8))
input_header.grid_columnconfigure(0, weight=1)

card_title = ctk.CTkLabel(
    input_header,
    text="Cell Nuclei Measurements",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(size=19, weight="bold"),
)
card_title.grid(row=0, column=0, sticky="w")

badge_30 = ctk.CTkLabel(
    input_header,
    text="30 FEATURES",
    text_color=ACCENT_BLUE[1],
    fg_color=CARD_ALT_COLOR,
    corner_radius=6,
    font=ctk.CTkFont(size=12, weight="bold"),
    padx=12,
    pady=5,
)
badge_30.grid(row=0, column=1, sticky="e")

# Scrollable Feature List
scroll_features = ctk.CTkScrollableFrame(
    input_card, fg_color=CARD_ALT_COLOR, corner_radius=8
)
scroll_features.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
scroll_features.grid_columnconfigure(0, weight=1)

feature_entries = []
entry_idx = 0

for g_name, g_feats in FEATURE_GROUPS.items():
    g_box = ctk.CTkFrame(
        scroll_features, fg_color=CARD_COLOR, corner_radius=8, border_width=1, border_color=BORDER_COLOR
    )
    g_box.grid(row=entry_idx, column=0, sticky="ew", padx=6, pady=(8, 6))
    g_box.grid_columnconfigure(0, weight=1)
    entry_idx += 1

    g_title = ctk.CTkLabel(
        g_box,
        text=g_name.upper(),
        text_color=ACCENT_BLUE[1],
        font=ctk.CTkFont(size=13, weight="bold"),
    )
    g_title.grid(row=0, column=0, padx=14, pady=(10, 6), sticky="w")

    for f_name in g_feats:
        row_bar = ctk.CTkFrame(g_box, fg_color="transparent")
        row_bar.grid(row=entry_idx, column=0, sticky="ew", padx=14, pady=3)
        row_bar.grid_columnconfigure(0, weight=1)

        f_label = ctk.CTkLabel(
            row_bar,
            text=f"{len(feature_entries)+1:02d}  {f_name}",
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=14),
        )
        f_label.grid(row=0, column=0, sticky="w")

        f_entry = ctk.CTkEntry(
            row_bar,
            width=165,
            height=38,
            fg_color=CARD_ALT_COLOR,
            border_color=BORDER_COLOR,
            font=ctk.CTkFont(size=14),
        )
        f_entry.grid(row=0, column=1, padx=(10, 0))

        feature_entries.append(f_entry)
        entry_idx += 1

# Right Card: Diagnostics
result_card = ctk.CTkFrame(
    workspace, fg_color=CARD_COLOR, corner_radius=10, border_width=1, border_color=BORDER_COLOR
)
result_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
result_card.grid_columnconfigure(0, weight=1)

res_header = ctk.CTkFrame(result_card, fg_color="transparent")
res_header.grid(row=0, column=0, sticky="ew", padx=22, pady=(14, 8))

res_title = ctk.CTkLabel(
    res_header,
    text="Diagnostic Assessment",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(size=19, weight="bold"),
)
res_title.pack(anchor="w")

# Prediction Card
pred_box = ctk.CTkFrame(result_card, fg_color=CARD_ALT_COLOR, corner_radius=10)
pred_box.grid(row=1, column=0, sticky="ew", padx=22, pady=8)
pred_box.grid_columnconfigure(0, weight=1)

pred_tag = ctk.CTkLabel(
    pred_box,
    text="PREDICTION RESULT",
    text_color=TEXT_MUTED,
    font=ctk.CTkFont(size=12, weight="bold"),
)
pred_tag.grid(row=0, column=0, pady=(12, 2))

lbl_prediction = ctk.CTkLabel(
    pred_box,
    text="AWAITING INPUT",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(size=36, weight="bold"),
)
lbl_prediction.grid(row=1, column=0, pady=(0, 14))

# Confidence
lbl_confidence = ctk.CTkLabel(
    result_card,
    text="Model Confidence: —",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(size=16, weight="bold"),
)
lbl_confidence.grid(row=2, column=0, padx=22, pady=(12, 6), sticky="w")

# Probabilities
prob_frame = ctk.CTkFrame(result_card, fg_color="transparent")
prob_frame.grid(row=3, column=0, sticky="ew", padx=22, pady=(6, 12))
prob_frame.grid_columnconfigure(0, weight=1)

lbl_malig_stat = ctk.CTkLabel(
    prob_frame,
    text="Malignant Risk: —",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(size=14, weight="bold"),
)
lbl_malig_stat.grid(row=0, column=0, sticky="w")
bar_malig = ctk.CTkProgressBar(
    prob_frame, height=12, corner_radius=6, progress_color=COLOR_MALIGNANT
)
bar_malig.grid(row=1, column=0, sticky="ew", pady=(3, 10))
bar_malig.set(0)

lbl_benign_stat = ctk.CTkLabel(
    prob_frame,
    text="Benign Likelihood: —",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(size=14, weight="bold"),
)
lbl_benign_stat.grid(row=2, column=0, sticky="w")
bar_benign = ctk.CTkProgressBar(
    prob_frame, height=12, corner_radius=6, progress_color=COLOR_BENIGN
)
bar_benign.grid(row=3, column=0, sticky="ew", pady=(3, 0))
bar_benign.set(0)

# Nearest Neighbors Box
knn_box = ctk.CTkFrame(result_card, fg_color=CARD_ALT_COLOR, corner_radius=8)
knn_box.grid(row=4, column=0, sticky="ew", padx=22, pady=(8, 10))
knn_box.grid_columnconfigure(0, weight=1)

lbl_knn_tag = ctk.CTkLabel(
    knn_box,
    text="NEAREST REFERENCE BIOPSIES (k-NN)",
    text_color=TEXT_MUTED,
    font=ctk.CTkFont(size=12, weight="bold"),
)
lbl_knn_tag.grid(row=0, column=0, padx=14, pady=(10, 4), sticky="w")

lbl_knn_val = ctk.CTkLabel(
    knn_box,
    text="No specimen processed.",
    text_color=TEXT_PRIMARY,
    wraplength=480,
    justify="left",
    font=ctk.CTkFont(size=14),
)
lbl_knn_val.grid(row=1, column=0, padx=14, pady=(0, 12), sticky="w")

disclaimer = ctk.CTkLabel(
    result_card,
    text="⚠ Experimental ML Prototype for Research Support Only.",
    text_color=COLOR_WARN,
    font=ctk.CTkFont(size=13, slant="italic"),
)
disclaimer.grid(row=5, column=0, pady=(6, 12))

# =========================================================
# ROW 3: RECENT PREDICTIONS BOX
# =========================================================

history_strip = ctk.CTkFrame(
    app, fg_color=CARD_COLOR, corner_radius=10, border_width=1, border_color=BORDER_COLOR
)
history_strip.grid(row=3, column=0, sticky="ew", padx=25, pady=(12, 10))
history_strip.grid_columnconfigure(1, weight=1)

h_title_box = ctk.CTkFrame(history_strip, fg_color="transparent")
h_title_box.grid(row=0, column=0, padx=18, pady=12, sticky="nw")

ctk.CTkLabel(
    h_title_box,
    text="Recent Sessions",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(size=15, weight="bold"),
).pack(anchor="w")

ctk.CTkLabel(
    h_title_box,
    text="Last 3 entries",
    text_color=TEXT_MUTED,
    font=ctk.CTkFont(size=12),
).pack(anchor="w", pady=(2, 0))

txt_history = ctk.CTkTextbox(
    history_strip,
    height=95,
    corner_radius=8,
    fg_color=CARD_ALT_COLOR,
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(family="Courier", size=13),
    wrap="none",
)
txt_history.grid(row=0, column=1, padx=(0, 18), pady=10, sticky="ew")
txt_history.configure(state="disabled")

# =========================================================
# ROW 4: ACTION TOOLBAR
# =========================================================

toolbar = ctk.CTkFrame(
    app, fg_color=CARD_COLOR, corner_radius=0, border_width=1, border_color=BORDER_COLOR
)
toolbar.grid(row=4, column=0, sticky="ew", padx=0, pady=0)

for i in range(8):
    toolbar.grid_columnconfigure(i, weight=1)

# =========================================================
# LOGIC & HANDLERS
# =========================================================

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            status_pill.configure(
                text="● API Connected (Online)", text_color=COLOR_BENIGN
            )
        else:
            status_pill.configure(
                text=f"API Error {r.status_code}", text_color=COLOR_MALIGNANT
            )
    except requests.exceptions.ConnectionError:
        status_pill.configure(
            text="● API Offline (Run uvicorn)", text_color=COLOR_MALIGNANT
        )
    except Exception:
        status_pill.configure(text="API Error", text_color=COLOR_MALIGNANT)


def load_sample():
    ent_patient_id.delete(0, "end")
    ent_patient_id.insert(0, "PT-BENCH-01")

    ent_patient_name.delete(0, "end")
    ent_patient_name.insert(0, "Benchmark Diagnostic Case")

    for ent, val in zip(feature_entries, DEFAULT_SAMPLE):
        ent.delete(0, "end")
        ent.insert(0, str(val))


def clear_all():
    ent_patient_id.delete(0, "end")
    ent_patient_name.delete(0, "end")

    for ent in feature_entries:
        ent.delete(0, "end")

    lbl_prediction.configure(text="AWAITING INPUT", text_color=TEXT_PRIMARY)
    lbl_confidence.configure(text="Model Confidence: —")
    lbl_malig_stat.configure(text="Malignant Risk: —")
    lbl_benign_stat.configure(text="Benign Likelihood: —")
    bar_malig.set(0)
    bar_benign.set(0)
    lbl_knn_val.configure(text="No specimen processed.")


def get_feature_vector():
    vals = []
    for idx, ent in enumerate(feature_entries, 1):
        v = ent.get().strip()
        if not v:
            raise ValueError(f"Missing feature #{idx} ({FEATURE_NAMES[idx-1]}).")
        try:
            vals.append(float(v))
        except ValueError:
            raise ValueError(f"Feature #{idx} contains invalid number: '{v}'")
    return vals


def save_record(pid, pname, pred, conf):
    exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["Timestamp", "Patient ID", "Patient Name", "Prediction", "Confidence"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pid, pname, pred, conf])


def read_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def refresh_history():
    rows = read_history()
    txt_history.configure(state="normal")
    txt_history.delete("1.0", "end")

    if not rows:
        txt_history.insert("end", "No previous diagnosis logs recorded.")
    else:
        hdr = f"{'TIMESTAMP':<21} | {'ID':<12} | {'PATIENT NAME':<22} | {'PREDICTION':<12} | {'CONFIDENCE':<10}\n"
        div = "-" * 85 + "\n"
        txt_history.insert("end", hdr + div)

        for r in rows[-3:]:
            ts = r.get("Timestamp", "")[:19]
            pid = r.get("Patient ID", "N/A")[:10]
            pnm = r.get("Patient Name", "Unknown")[:20]
            prd = r.get("Prediction", "").upper()[:10]
            cnf = f"{r.get('Confidence', '')}%"
            txt_history.insert("end", f"{ts:<21} | {pid:<12} | {pnm:<22} | {prd:<12} | {cnf:<10}\n")

    txt_history.configure(state="disabled")


def run_prediction():
    try:
        features = get_feature_vector()
        pid = ent_patient_id.get().strip() or "ANONYMOUS"
        pname = ent_patient_name.get().strip() or "Unnamed Specimen"

        lbl_prediction.configure(text="INFERRING...", text_color=ACCENT_BLUE[1])
        app.update_idletasks()

        res = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=8)
        if res.status_code != 200:
            messagebox.showerror("Inference Failed", f"HTTP {res.status_code}\n{res.text}")
            lbl_prediction.configure(text="ERROR", text_color=COLOR_MALIGNANT)
            return

        data = res.json()
        pred = data["prediction"].upper()
        conf = data["confidence_pct"]
        probs = data["class_probabilities"]
        dists = data["nearest_neighbors_distances"]

        if pred == "MALIGNANT":
            lbl_prediction.configure(text=f"● {pred}", text_color=COLOR_MALIGNANT)
        else:
            lbl_prediction.configure(text=f"● {pred}", text_color=COLOR_BENIGN)

        lbl_confidence.configure(text=f"Model Confidence: {conf}%")

        m_prob = float(probs.get("malignant", 0))
        b_prob = float(probs.get("benign", 0))

        lbl_malig_stat.configure(text=f"Malignant Risk: {m_prob*100:.2f}%")
        bar_malig.set(m_prob)

        lbl_benign_stat.configure(text=f"Benign Likelihood: {b_prob*100:.2f}%")
        bar_benign.set(b_prob)

        dist_str = ", ".join(f"{d:.4f}" for d in dists[:3])
        lbl_knn_val.configure(text=f"Distances to 3 closest cases: [ {dist_str} ]")

        save_record(pid, pname, pred, conf)
        refresh_history()

    except ValueError as e:
        messagebox.showwarning("Validation Warning", str(e))
    except requests.exceptions.ConnectionError:
        messagebox.showerror("Connection Error", "FastAPI backend unreachable.\nRun 'uvicorn main:app --reload'.")
    except Exception as e:
        messagebox.showerror("Runtime Error", str(e))


def view_full_history():
    rows = read_history()
    win = ctk.CTkToplevel(app)
    win.title("Diagnosis Audit History")
    win.geometry("980x560")
    win.configure(fg_color=BG_COLOR)
    win.transient(app)
    win.grab_set()

    win.grid_columnconfigure(0, weight=1)
    win.grid_rowconfigure(1, weight=1)

    top = ctk.CTkFrame(win, fg_color=CARD_COLOR, corner_radius=0)
    top.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
    ctk.CTkLabel(
        top,
        text="Clinical Prediction Audit Log",
        text_color=TEXT_PRIMARY,
        font=ctk.CTkFont(size=20, weight="bold"),
    ).pack(anchor="w", padx=20, pady=14)

    scroll = ctk.CTkScrollableFrame(win, fg_color=CARD_COLOR, corner_radius=8)
    scroll.grid(row=1, column=0, sticky="nsew", padx=20, pady=12)

    headers = ["Timestamp", "Patient ID", "Name", "Result", "Confidence"]
    widths = [180, 110, 200, 120, 110]

    for c, (h, w) in enumerate(zip(headers, widths)):
        scroll.grid_columnconfigure(c, minsize=w, weight=1)
        ctk.CTkLabel(
            scroll, text=h, text_color=TEXT_MUTED, font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=c, pady=8)

    for r_idx, r in enumerate(rows, 1):
        items = [
            r.get("Timestamp", ""),
            r.get("Patient ID", ""),
            r.get("Patient Name", ""),
            r.get("Prediction", "").upper(),
            f"{r.get('Confidence', '')}%",
        ]
        for c_idx, val in enumerate(items):
            color = (
                COLOR_MALIGNANT
                if val == "MALIGNANT"
                else (COLOR_BENIGN if val == "BENIGN" else TEXT_PRIMARY)
            )
            ctk.CTkLabel(
                scroll, text=val, text_color=color, font=ctk.CTkFont(size=13)
            ).grid(row=r_idx, column=c_idx, pady=5)

    ctk.CTkButton(
        win, text="Dismiss", width=120, height=36, command=win.destroy
    ).grid(row=2, column=0, pady=14)


def clear_history():
    if not os.path.exists(HISTORY_FILE):
        return
    if messagebox.askyesno("Clear History", "Delete all local prediction logs?"):
        os.remove(HISTORY_FILE)
        refresh_history()


def export_history():
    if not os.path.exists(HISTORY_FILE):
        messagebox.showinfo("Export", "No log records found.")
        return
    path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
    if path:
        with open(HISTORY_FILE, "r", encoding="utf-8") as s, open(path, "w", encoding="utf-8") as d:
            d.write(s.read())
        messagebox.showinfo("Export", f"Exported successfully to:\n{path}")


def choose_csv_row(rows):
    """Displays a modal picker dialog when a CSV has multiple records."""
    modal = ctk.CTkToplevel(app)
    modal.title("Select Patient Specimen")
    modal.geometry("820x540")
    modal.configure(fg_color=BG_COLOR)
    modal.transient(app)
    modal.grab_set()

    modal.grid_columnconfigure(0, weight=1)
    modal.grid_rowconfigure(1, weight=1)

    top_banner = ctk.CTkFrame(modal, fg_color=CARD_COLOR, corner_radius=0)
    top_banner.grid(row=0, column=0, sticky="ew")

    ctk.CTkLabel(
        top_banner,
        text="Multiple Specimens Detected",
        text_color=TEXT_PRIMARY,
        font=ctk.CTkFont(size=20, weight="bold"),
    ).pack(anchor="w", padx=24, pady=(16, 2))

    ctk.CTkLabel(
        top_banner,
        text=f"Found {len(rows)} records in CSV. Select which patient to load into the diagnostic studio:",
        text_color=TEXT_MUTED,
        font=ctk.CTkFont(size=13),
    ).pack(anchor="w", padx=24, pady=(0, 14))

    scroll_area = ctk.CTkScrollableFrame(
        modal, fg_color=CARD_ALT_COLOR, corner_radius=8
    )
    scroll_area.grid(row=1, column=0, sticky="nsew", padx=20, pady=12)

    selected = {"row": None}

    for idx, row in enumerate(rows, start=1):
        pid = row.get("patient_id", f"Row #{idx}")
        pname = row.get("patient_name", "Unknown Name")

        btn_txt = f"#{idx:02d}   |   Patient ID: {pid:<15}   |   Name: {pname}"

        def make_select_handler(r=row):
            def select_item():
                selected["row"] = r
                modal.destroy()
            return select_item

        ctk.CTkButton(
            scroll_area,
            text=btn_txt,
            command=make_select_handler(row),
            height=44,
            corner_radius=6,
            fg_color=CARD_COLOR,
            hover_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            anchor="w",
            font=ctk.CTkFont(family="monospace", size=13),
        ).pack(fill="x", padx=8, pady=4)

    def cancel():
        modal.destroy()

    ctk.CTkButton(
        modal,
        text="Cancel",
        width=120,
        height=38,
        fg_color=CARD_ALT_COLOR,
        text_color=TEXT_PRIMARY,
        command=cancel,
    ).grid(row=2, column=0, pady=(0, 16))

    app.wait_window(modal)
    return selected["row"]


def load_csv():
    fpath = filedialog.askopenfilename(
        title="Select Patient CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not fpath:
        return

    try:
        with open(fpath, "r", encoding="utf-8-sig") as f:
            reader = list(csv.DictReader(f))

        if not reader:
            messagebox.showwarning("Empty CSV", "The selected file has no records.")
            return

        # 1. Select the row (auto-picks if 1 row, opens picker if > 1 rows)
        if len(reader) == 1:
            chosen_row = reader[0]
        else:
            chosen_row = choose_csv_row(reader)

        if not chosen_row:
            return  # User canceled the picker

        # 2. Populate Patient Identifier & Name
        ent_patient_id.delete(0, "end")
        ent_patient_id.insert(0, chosen_row.get("patient_id", "P-CSV"))

        ent_patient_name.delete(0, "end")
        ent_patient_name.insert(0, chosen_row.get("patient_name", "Imported Patient"))

        # 3. Populate all 30 feature entries
        matched_count = 0
        for ent, fn in zip(feature_entries, FEATURE_NAMES):
            if fn in chosen_row and chosen_row[fn].strip() != "":
                ent.delete(0, "end")
                ent.insert(0, chosen_row[fn].strip())
                matched_count += 1

        if matched_count == len(FEATURE_NAMES):
            messagebox.showinfo(
                "Patient Loaded",
                f"Successfully loaded 30 cell measurements for '{ent_patient_name.get()}'.\n\nClick 'DIAGNOSE NOW' to run inference.",
            )
        else:
            messagebox.showwarning(
                "Partial Import",
                f"Matched {matched_count}/{len(FEATURE_NAMES)} features from CSV.\nPlease check and fill missing values before running diagnosis.",
            )

    except UnicodeDecodeError:
        messagebox.showerror(
            "Encoding Error", "Could not decode file. Please save your CSV as standard UTF-8."
        )
    except Exception as e:
        messagebox.showerror("CSV Error", str(e))


# =========================================================
# ATTACH TOOLBAR BUTTONS
# =========================================================

btn_config = [
    ("Check Health", check_api, CARD_ALT_COLOR, TEXT_PRIMARY),
    ("Benchmark Specimen", load_sample, CARD_ALT_COLOR, TEXT_PRIMARY),
    ("Import CSV", load_csv, CARD_ALT_COLOR, TEXT_PRIMARY),
    ("DIAGNOSE NOW", run_prediction, ACCENT_BLUE, "white"),
    ("Reset Inputs", clear_all, CARD_ALT_COLOR, TEXT_PRIMARY),
    ("Audit Logs", view_full_history, CARD_ALT_COLOR, TEXT_PRIMARY),
    ("Purge Logs", clear_history, CARD_ALT_COLOR, COLOR_MALIGNANT),
    ("Export CSV", export_history, CARD_ALT_COLOR, TEXT_PRIMARY),
]

for col, (txt, cmd, bg, fg) in enumerate(btn_config):
    ctk.CTkButton(
        toolbar,
        text=txt,
        command=cmd,
        height=48,
        corner_radius=6,
        fg_color=bg,
        text_color=fg,
        font=ctk.CTkFont(size=14, weight="bold"),
    ).grid(row=0, column=col, padx=4, pady=10, sticky="ew")

# =========================================================
# BOOT
# =========================================================

refresh_history()
app.after(300, check_api)

if __name__ == "__main__":
    app.mainloop()