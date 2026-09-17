import customtkinter as ctk
import requests
import csv
import os

from datetime import datetime
from tkinter import messagebox, filedialog


# =========================================================
# CONFIG
# =========================================================

API_URL = "http://127.0.0.1:8000"

HISTORY_FILE = "prediction_history.csv"

APP_TITLE = "Breast Cancer Diagnostic Classifier"


# =========================================================
# COLORS - DESIGN ONLY
# =========================================================

BG_COLOR = "#F4F7FB"
CARD_COLOR = "#FFFFFF"
BORDER_COLOR = "#E3E8EF"

TEXT_PRIMARY = "#1F2937"
TEXT_SECONDARY = "#6B7280"

ACCENT_COLOR = "#2563EB"
ACCENT_HOVER = "#1D4ED8"

SUCCESS_COLOR = "#16A34A"
WARNING_COLOR = "#D97706"
DANGER_COLOR = "#DC2626"

SOFT_BLUE = "#EFF6FF"
SOFT_GREEN = "#ECFDF3"
SOFT_RED = "#FEF2F2"
SOFT_GRAY = "#F8FAFC"


# =========================================================
# FEATURE NAMES
# =========================================================

FEATURE_GROUPS = {

    "Mean Features": [
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

    "Error Features": [
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

    "Worst Features": [
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

for group_features in FEATURE_GROUPS.values():
    FEATURE_NAMES.extend(group_features)


# =========================================================
# DEFAULT SAMPLE
# =========================================================

DEFAULT_SAMPLE = [

    # Mean
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

    # Error
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

    # Worst
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


# =========================================================
# CUSTOMTKINTER SETUP
# =========================================================

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


# =========================================================
# MAIN WINDOW
# =========================================================

app = ctk.CTk()

app.title(APP_TITLE)

app.geometry("1250x800")

app.minsize(
    1050,
    720
)

app.configure(
    fg_color=BG_COLOR
)


# Try maximize on Windows
try:
    app.state("zoomed")
except Exception:
    pass


# =========================================================
# ROOT GRID
# =========================================================

app.grid_columnconfigure(
    0,
    weight=1
)

app.grid_rowconfigure(
    0,
    weight=0
)

app.grid_rowconfigure(
    1,
    weight=0
)

app.grid_rowconfigure(
    2,
    weight=1
)

app.grid_rowconfigure(
    3,
    weight=0
)

app.grid_rowconfigure(
    4,
    weight=0
)


# =========================================================
# HEADER
# =========================================================

header_frame = ctk.CTkFrame(
    app,
    fg_color=CARD_COLOR,
    corner_radius=0,
    border_width=0
)

header_frame.grid(
    row=0,
    column=0,
    sticky="ew"
)

header_frame.grid_columnconfigure(
    0,
    weight=1
)

header_frame.grid_columnconfigure(
    1,
    weight=0
)


# ---------------------------------------------------------
# Header left
# ---------------------------------------------------------

header_left = ctk.CTkFrame(
    header_frame,
    fg_color="transparent"
)

header_left.grid(
    row=0,
    column=0,
    padx=(30, 10),
    pady=20,
    sticky="w"
)


title_label = ctk.CTkLabel(
    header_left,
    text="Breast Cancer Diagnostic Classifier",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(
        size=25,
        weight="bold"
    )
)

title_label.pack(
    anchor="w"
)


subtitle_label = ctk.CTkLabel(
    header_left,
    text="Machine Learning Prediction Prototype",
    text_color=TEXT_SECONDARY,
    font=ctk.CTkFont(
        size=13
    )
)

subtitle_label.pack(
    anchor="w",
    pady=(4, 0)
)


# ---------------------------------------------------------
# Header right
# ---------------------------------------------------------

header_badge = ctk.CTkLabel(
    header_frame,
    text="ML PROTOTYPE",
    text_color=ACCENT_COLOR,
    fg_color=SOFT_BLUE,
    corner_radius=8,
    font=ctk.CTkFont(
        size=11,
        weight="bold"
    ),
    padx=14,
    pady=7
)

header_badge.grid(
    row=0,
    column=1,
    padx=(10, 30),
    pady=20
)


# =========================================================
# STATUS BAR
# =========================================================

status_frame = ctk.CTkFrame(
    app,
    fg_color=BG_COLOR,
    corner_radius=0
)

status_frame.grid(
    row=1,
    column=0,
    sticky="ew",
    padx=25,
    pady=(15, 10)
)

status_frame.grid_columnconfigure(
    1,
    weight=1
)


status_title = ctk.CTkLabel(
    status_frame,
    text="API STATUS",
    text_color=TEXT_SECONDARY,
    font=ctk.CTkFont(
        size=11,
        weight="bold"
    )
)

status_title.grid(
    row=0,
    column=0,
    padx=(5, 10),
    pady=5
)


status_label = ctk.CTkLabel(
    status_frame,
    text="Checking connection...",
    text_color=TEXT_SECONDARY,
    fg_color=CARD_COLOR,
    corner_radius=8,
    anchor="w",
    padx=15,
    pady=8,
    font=ctk.CTkFont(
        size=12
    )
)

status_label.grid(
    row=0,
    column=1,
    sticky="ew",
    padx=(0, 5),
    pady=2
)


# =========================================================
# PATIENT INFORMATION CARD
# =========================================================

patient_frame = ctk.CTkFrame(
    app,
    fg_color=CARD_COLOR,
    corner_radius=12,
    border_width=1,
    border_color=BORDER_COLOR
)

patient_frame.grid(
    row=1,
    column=0,
    sticky="ew",
    padx=25,
    pady=(0, 12)
)

patient_frame.grid_columnconfigure(
    1,
    weight=1
)

patient_frame.grid_columnconfigure(
    3,
    weight=2
)


patient_section_title = ctk.CTkLabel(
    patient_frame,
    text="Patient Information",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(
        size=14,
        weight="bold"
    )
)

patient_section_title.grid(
    row=0,
    column=0,
    columnspan=4,
    padx=18,
    pady=(12, 3),
    sticky="w"
)


patient_id_label = ctk.CTkLabel(
    patient_frame,
    text="Patient ID",
    text_color=TEXT_SECONDARY,
    font=ctk.CTkFont(
        size=12,
        weight="bold"
    )
)

patient_id_label.grid(
    row=1,
    column=0,
    padx=(18, 8),
    pady=(5, 14)
)


patient_id_entry = ctk.CTkEntry(
    patient_frame,
    placeholder_text="e.g. P001",
    height=36,
    corner_radius=8,
    border_color=BORDER_COLOR
)

patient_id_entry.grid(
    row=1,
    column=1,
    padx=(0, 20),
    pady=(5, 14),
    sticky="ew"
)


patient_name_label = ctk.CTkLabel(
    patient_frame,
    text="Patient Name",
    text_color=TEXT_SECONDARY,
    font=ctk.CTkFont(
        size=12,
        weight="bold"
    )
)

patient_name_label.grid(
    row=1,
    column=2,
    padx=(10, 8),
    pady=(5, 14)
)


patient_name_entry = ctk.CTkEntry(
    patient_frame,
    placeholder_text="Enter patient name",
    height=36,
    corner_radius=8,
    border_color=BORDER_COLOR
)

patient_name_entry.grid(
    row=1,
    column=3,
    padx=(0, 18),
    pady=(5, 14),
    sticky="ew"
)


# =========================================================
# MAIN CONTENT
# =========================================================

content_frame = ctk.CTkFrame(
    app,
    fg_color="transparent"
)

content_frame.grid(
    row=2,
    column=0,
    sticky="nsew",
    padx=25,
    pady=(0, 12)
)

content_frame.grid_columnconfigure(
    0,
    weight=3
)

content_frame.grid_columnconfigure(
    1,
    weight=2
)

content_frame.grid_rowconfigure(
    0,
    weight=1
)


# =========================================================
# LEFT INPUT CARD
# =========================================================

input_frame = ctk.CTkFrame(
    content_frame,
    fg_color=CARD_COLOR,
    corner_radius=12,
    border_width=1,
    border_color=BORDER_COLOR
)

input_frame.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0, 7)
)

input_frame.grid_columnconfigure(
    0,
    weight=1
)

input_frame.grid_rowconfigure(
    1,
    weight=1
)


input_header = ctk.CTkFrame(
    input_frame,
    fg_color="transparent"
)

input_header.grid(
    row=0,
    column=0,
    sticky="ew",
    padx=18,
    pady=(15, 8)
)

input_header.grid_columnconfigure(
    0,
    weight=1
)


input_title = ctk.CTkLabel(
    input_header,
    text="Cell Measurements",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(
        size=18,
        weight="bold"
    )
)

input_title.grid(
    row=0,
    column=0,
    sticky="w"
)


input_subtitle = ctk.CTkLabel(
    input_header,
    text="30 features required for prediction",
    text_color=TEXT_SECONDARY,
    font=ctk.CTkFont(
        size=11
    )
)

input_subtitle.grid(
    row=1,
    column=0,
    pady=(3, 0),
    sticky="w"
)


feature_count_badge = ctk.CTkLabel(
    input_header,
    text="30 FEATURES",
    text_color=ACCENT_COLOR,
    fg_color=SOFT_BLUE,
    corner_radius=7,
    font=ctk.CTkFont(
        size=10,
        weight="bold"
    ),
    padx=10,
    pady=5
)

feature_count_badge.grid(
    row=0,
    column=1,
    rowspan=2,
    padx=(10, 0)
)


# =========================================================
# SCROLLABLE FEATURE AREA
# =========================================================

feature_scroll = ctk.CTkScrollableFrame(
    input_frame,
    fg_color=SOFT_GRAY,
    corner_radius=10
)

feature_scroll.grid(
    row=1,
    column=0,
    sticky="nsew",
    padx=12,
    pady=(0, 12)
)

feature_scroll.grid_columnconfigure(
    0,
    weight=1
)


feature_entries = []


feature_number = 0


for group_name, features in FEATURE_GROUPS.items():

    # -----------------------------------------------------
    # Group card
    # -----------------------------------------------------

    group_frame = ctk.CTkFrame(
        feature_scroll,
        fg_color=CARD_COLOR,
        corner_radius=9,
        border_width=1,
        border_color=BORDER_COLOR
    )

    group_frame.grid(
        row=feature_number,
        column=0,
        sticky="ew",
        padx=5,
        pady=(8, 5)
    )

    group_frame.grid_columnconfigure(
        0,
        weight=1
    )

    group_label = ctk.CTkLabel(
        group_frame,
        text=group_name,
        text_color=ACCENT_COLOR,
        font=ctk.CTkFont(
            size=13,
            weight="bold"
        )
    )

    group_label.grid(
        row=0,
        column=0,
        padx=12,
        pady=(8, 7),
        sticky="w"
    )

    feature_number += 1

    for feature_name in features:

        row_frame = ctk.CTkFrame(
            group_frame,
            fg_color="transparent"
        )

        row_frame.grid(
            row=feature_number,
            column=0,
            sticky="ew",
            padx=8,
            pady=2
        )

        row_frame.grid_columnconfigure(
            0,
            weight=1
        )

        label = ctk.CTkLabel(
            row_frame,
            text=f"{len(feature_entries) + 1:02d}  {feature_name}",
            text_color=TEXT_PRIMARY,
            anchor="w",
            font=ctk.CTkFont(
                size=11
            )
        )

        label.grid(
            row=0,
            column=0,
            sticky="w",
            padx=(5, 10)
        )

        entry = ctk.CTkEntry(
            row_frame,
            width=125,
            height=31,
            corner_radius=7,
            border_color=BORDER_COLOR
        )

        entry.grid(
            row=0,
            column=1,
            padx=5,
            pady=2
        )

        feature_entries.append(
            entry
        )

        feature_number += 1


# =========================================================
# RIGHT RESULT CARD
# =========================================================

result_frame = ctk.CTkFrame(
    content_frame,
    fg_color=CARD_COLOR,
    corner_radius=12,
    border_width=1,
    border_color=BORDER_COLOR
)

result_frame.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(7, 0)
)

result_frame.grid_columnconfigure(
    0,
    weight=1
)


# =========================================================
# RESULT HEADER
# =========================================================

result_header = ctk.CTkFrame(
    result_frame,
    fg_color="transparent"
)

result_header.grid(
    row=0,
    column=0,
    sticky="ew",
    padx=20,
    pady=(18, 8)
)

result_header.grid_columnconfigure(
    0,
    weight=1
)


result_title = ctk.CTkLabel(
    result_header,
    text="Prediction Result",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(
        size=19,
        weight="bold"
    )
)

result_title.grid(
    row=0,
    column=0,
    sticky="w"
)


result_subtitle = ctk.CTkLabel(
    result_header,
    text="Model output",
    text_color=TEXT_SECONDARY,
    font=ctk.CTkFont(
        size=11
    )
)

result_subtitle.grid(
    row=1,
    column=0,
    pady=(2, 0),
    sticky="w"
)


# =========================================================
# PREDICTION RESULT BOX
# =========================================================

prediction_box = ctk.CTkFrame(
    result_frame,
    fg_color=SOFT_BLUE,
    corner_radius=10
)

prediction_box.grid(
    row=1,
    column=0,
    sticky="ew",
    padx=20,
    pady=(5, 10)
)

prediction_box.grid_columnconfigure(
    0,
    weight=1
)


prediction_caption = ctk.CTkLabel(
    prediction_box,
    text="PREDICTION",
    text_color=TEXT_SECONDARY,
    font=ctk.CTkFont(
        size=10,
        weight="bold"
    )
)

prediction_caption.grid(
    row=0,
    column=0,
    pady=(12, 2)
)


prediction_label = ctk.CTkLabel(
    prediction_box,
    text="—",
    text_color=ACCENT_COLOR,
    font=ctk.CTkFont(
        size=28,
        weight="bold"
    )
)

prediction_label.grid(
    row=1,
    column=0,
    pady=(0, 13)
)


# =========================================================
# CONFIDENCE
# =========================================================

confidence_card = ctk.CTkFrame(
    result_frame,
    fg_color=SOFT_GRAY,
    corner_radius=10
)

confidence_card.grid(
    row=2,
    column=0,
    sticky="ew",
    padx=20,
    pady=5
)

confidence_card.grid_columnconfigure(
    0,
    weight=1
)


confidence_caption = ctk.CTkLabel(
    confidence_card,
    text="CONFIDENCE",
    text_color=TEXT_SECONDARY,
    font=ctk.CTkFont(
        size=10,
        weight="bold"
    )
)

confidence_caption.grid(
    row=0,
    column=0,
    pady=(10, 0)
)


confidence_label = ctk.CTkLabel(
    confidence_card,
    text="Confidence: —",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(
        size=17,
        weight="bold"
    )
)

confidence_label.grid(
    row=1,
    column=0,
    pady=(2, 10)
)


# =========================================================
# PROBABILITY SECTION
# =========================================================

probability_title = ctk.CTkLabel(
    result_frame,
    text="Class Probabilities",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(
        size=14,
        weight="bold"
    )
)

probability_title.grid(
    row=3,
    column=0,
    padx=20,
    pady=(18, 8),
    sticky="w"
)


# ---------------------------------------------------------
# Malignant
# ---------------------------------------------------------

malignant_label = ctk.CTkLabel(
    result_frame,
    text="Malignant: —",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(
        size=12
    )
)

malignant_label.grid(
    row=4,
    column=0,
    padx=20,
    pady=(3, 3),
    sticky="w"
)


malignant_bar = ctk.CTkProgressBar(
    result_frame,
    height=10,
    corner_radius=5
)

malignant_bar.grid(
    row=5,
    column=0,
    padx=20,
    pady=(0, 10),
    sticky="ew"
)

malignant_bar.set(
    0
)


# ---------------------------------------------------------
# Benign
# ---------------------------------------------------------

benign_label = ctk.CTkLabel(
    result_frame,
    text="Benign: —",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(
        size=12
    )
)

benign_label.grid(
    row=6,
    column=0,
    padx=20,
    pady=(3, 3),
    sticky="w"
)


benign_bar = ctk.CTkProgressBar(
    result_frame,
    height=10,
    corner_radius=5
)

benign_bar.grid(
    row=7,
    column=0,
    padx=20,
    pady=(0, 10),
    sticky="ew"
)

benign_bar.set(
    0
)


# =========================================================
# NEAREST NEIGHBORS CARD
# =========================================================

neighbors_card = ctk.CTkFrame(
    result_frame,
    fg_color=SOFT_GRAY,
    corner_radius=10
)

neighbors_card.grid(
    row=8,
    column=0,
    sticky="ew",
    padx=20,
    pady=(10, 8)
)

neighbors_card.grid_columnconfigure(
    0,
    weight=1
)


neighbors_caption = ctk.CTkLabel(
    neighbors_card,
    text="NEAREST NEIGHBOR DISTANCES",
    text_color=TEXT_SECONDARY,
    font=ctk.CTkFont(
        size=10,
        weight="bold"
    )
)

neighbors_caption.grid(
    row=0,
    column=0,
    padx=12,
    pady=(10, 3),
    sticky="w"
)


neighbors_label = ctk.CTkLabel(
    neighbors_card,
    text="—",
    text_color=TEXT_PRIMARY,
    wraplength=400,
    justify="left",
    font=ctk.CTkFont(
        size=11
    )
)

neighbors_label.grid(
    row=1,
    column=0,
    padx=12,
    pady=(0, 10),
    sticky="w"
)


# =========================================================
# DISCLAIMER
# =========================================================

disclaimer_label = ctk.CTkLabel(
    result_frame,
    text=(
        "⚠ Experimental ML prediction only.\n"
        "This is NOT a medical diagnosis."
    ),
    text_color=WARNING_COLOR,
    font=ctk.CTkFont(
        size=11,
        slant="italic"
    ),
    justify="center"
)

disclaimer_label.grid(
    row=9,
    column=0,
    padx=20,
    pady=(10, 15)
)


# =========================================================
# HISTORY PREVIEW
# =========================================================

history_frame = ctk.CTkFrame(
    app,
    fg_color=CARD_COLOR,
    corner_radius=12,
    border_width=1,
    border_color=BORDER_COLOR
)

history_frame.grid(
    row=3,
    column=0,
    sticky="ew",
    padx=25,
    pady=(0, 10)
)

history_frame.grid_columnconfigure(
    1,
    weight=1
)


history_title = ctk.CTkLabel(
    history_frame,
    text="Recent Predictions",
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(
        size=13,
        weight="bold"
    )
)

history_title.grid(
    row=0,
    column=0,
    padx=(15, 10),
    pady=10
)


history_text = ctk.CTkTextbox(
    history_frame,
    height=55,
    corner_radius=8,
    border_width=0,
    fg_color=SOFT_GRAY,
    text_color=TEXT_PRIMARY,
    font=ctk.CTkFont(
        size=11
    )
)

history_text.grid(
    row=0,
    column=1,
    padx=(0, 15),
    pady=8,
    sticky="ew"
)

history_text.configure(
    state="disabled"
)


# =========================================================
# BOTTOM TOOLBAR
# =========================================================

button_frame = ctk.CTkFrame(
    app,
    fg_color=CARD_COLOR,
    corner_radius=0
)

button_frame.grid(
    row=4,
    column=0,
    sticky="ew",
    padx=25,
    pady=(0, 15)
)

for column in range(8):

    button_frame.grid_columnconfigure(
        column,
        weight=1
    )


# =========================================================
# FUNCTIONS
# =========================================================

def check_api():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )

        if response.status_code == 200:

            data = response.json()

            status_label.configure(
                text=(
                    "● Connected     "
                    f"Model Loaded: {data.get('model_loaded')}     "
                    f"Features: {data.get('features_expected')}"
                ),
                text_color=SUCCESS_COLOR
            )

        else:

            status_label.configure(
                text=(
                    f"API Error: "
                    f"HTTP {response.status_code}"
                ),
                text_color=DANGER_COLOR
            )

    except requests.exceptions.ConnectionError:

        status_label.configure(
            text=(
                "● Disconnected     "
                "Start FastAPI server first."
            ),
            text_color=DANGER_COLOR
        )

    except Exception as e:

        status_label.configure(
            text=f"Error: {str(e)}",
            text_color=DANGER_COLOR
        )


# =========================================================
# LOAD SAMPLE
# =========================================================

def load_sample():

    patient_id_entry.delete(
        0,
        "end"
    )

    patient_id_entry.insert(
        0,
        "P001"
    )

    patient_name_entry.delete(
        0,
        "end"
    )

    patient_name_entry.insert(
        0,
        "Sample Patient"
    )

    for entry, value in zip(
        feature_entries,
        DEFAULT_SAMPLE
    ):

        entry.delete(
            0,
            "end"
        )

        entry.insert(
            0,
            str(value)
        )


# =========================================================
# CLEAR
# =========================================================

def clear_all():

    patient_id_entry.delete(
        0,
        "end"
    )

    patient_name_entry.delete(
        0,
        "end"
    )

    for entry in feature_entries:

        entry.delete(
            0,
            "end"
        )

    prediction_label.configure(
        text="—"
    )

    confidence_label.configure(
        text="Confidence: —"
    )

    malignant_label.configure(
        text="Malignant: —"
    )

    benign_label.configure(
        text="Benign: —"
    )

    malignant_bar.set(
        0
    )

    benign_bar.set(
        0
    )

    neighbors_label.configure(
        text="—"
    )


# =========================================================
# GET FEATURES
# =========================================================

def get_features():

    values = []

    for entry in feature_entries:

        value = entry.get().strip()

        if value == "":

            raise ValueError(
                "Please fill all 30 feature values."
            )

        try:

            number = float(
                value
            )

        except ValueError:

            raise ValueError(
                f"Invalid number: {value}"
            )

        values.append(
            number
        )

    if len(values) != 30:

        raise ValueError(
            "Exactly 30 features are required."
        )

    return values


# =========================================================
# SAVE HISTORY
# =========================================================

def save_history(
    patient_id,
    patient_name,
    prediction,
    confidence
):

    file_exists = os.path.exists(
        HISTORY_FILE
    )

    with open(
        HISTORY_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(
            file
        )

        if not file_exists:

            writer.writerow([
                "Timestamp",
                "Patient ID",
                "Patient Name",
                "Prediction",
                "Confidence"
            ])

        writer.writerow([
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            patient_id,
            patient_name,
            prediction,
            confidence
        ])


# =========================================================
# READ HISTORY
# =========================================================

def read_history():

    if not os.path.exists(
        HISTORY_FILE
    ):

        return []

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(
                file
            )

            return list(reader)

    except Exception as e:

        print(
            "History reading error:",
            e
        )

        return []


# =========================================================
# LOAD HISTORY
# =========================================================

def load_history():

    rows = read_history()

    history_text.configure(
        state="normal"
    )

    history_text.delete(
        "1.0",
        "end"
    )

    if not rows:

        history_text.insert(
            "end",
            "No prediction history yet."
        )

    else:

        for row in rows[-5:]:

            history_text.insert(
                "end",

                (
                    f"{row.get('Timestamp', '')}    |    "
                    f"{row.get('Patient ID', '')}    |    "
                    f"{row.get('Patient Name', '')}    |    "
                    f"{row.get('Prediction', '').upper()}    |    "
                    f"{row.get('Confidence', '')}%\n"
                )
            )

    history_text.configure(
        state="disabled"
    )


# =========================================================
# VIEW FULL HISTORY
# =========================================================

def view_full_history():

    rows = read_history()

    window = ctk.CTkToplevel(
        app
    )

    window.title(
        "Prediction History"
    )

    window.geometry(
        "1000x600"
    )

    window.minsize(
        800,
        500
    )

    window.configure(
        fg_color=BG_COLOR
    )

    window.transient(
        app
    )

    window.grab_set()

    window.grid_columnconfigure(
        0,
        weight=1
    )

    window.grid_rowconfigure(
        1,
        weight=1
    )

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    header = ctk.CTkFrame(
        window,
        fg_color=CARD_COLOR,
        corner_radius=0
    )

    header.grid(
        row=0,
        column=0,
        sticky="ew"
    )

    title = ctk.CTkLabel(
        header,
        text="Prediction History",
        text_color=TEXT_PRIMARY,
        font=ctk.CTkFont(
            size=22,
            weight="bold"
        )
    )

    title.pack(
        anchor="w",
        padx=25,
        pady=(18, 2)
    )

    subtitle = ctk.CTkLabel(
        header,
        text="Previous model prediction records",
        text_color=TEXT_SECONDARY,
        font=ctk.CTkFont(
            size=12
        )
    )

    subtitle.pack(
        anchor="w",
        padx=25,
        pady=(0, 15)
    )

    # -----------------------------------------------------
    # Table
    # -----------------------------------------------------

    table_frame = ctk.CTkScrollableFrame(
        window,
        fg_color=CARD_COLOR,
        corner_radius=12
    )

    table_frame.grid(
        row=1,
        column=0,
        sticky="nsew",
        padx=25,
        pady=15
    )

    headers = [
        "Timestamp",
        "Patient ID",
        "Patient Name",
        "Prediction",
        "Confidence"
    ]

    widths = [
        180,
        120,
        220,
        140,
        120
    ]

    for column, width in enumerate(
        widths
    ):

        table_frame.grid_columnconfigure(
            column,
            minsize=width,
            weight=1
        )

    # Header row

    for column, header_text in enumerate(
        headers
    ):

        label = ctk.CTkLabel(
            table_frame,
            text=header_text,
            text_color=TEXT_SECONDARY,
            fg_color=SOFT_GRAY,
            corner_radius=6,
            font=ctk.CTkFont(
                size=11,
                weight="bold"
            ),
            anchor="center"
        )

        label.grid(
            row=0,
            column=column,
            padx=3,
            pady=5,
            sticky="ew"
        )

    # No history

    if not rows:

        empty_label = ctk.CTkLabel(
            table_frame,
            text="No prediction history available.",
            text_color=TEXT_SECONDARY
        )

        empty_label.grid(
            row=1,
            column=0,
            columnspan=5,
            pady=40
        )

    else:

        for row_index, row in enumerate(
            rows,
            start=1
        ):

            values = [

                row.get(
                    "Timestamp",
                    ""
                ),

                row.get(
                    "Patient ID",
                    ""
                ),

                row.get(
                    "Patient Name",
                    ""
                ),

                row.get(
                    "Prediction",
                    ""
                ).upper(),

                f"{row.get('Confidence', '')}%"
            ]

            for column, value in enumerate(
                values
            ):

                label = ctk.CTkLabel(
                    table_frame,
                    text=value,
                    text_color=TEXT_PRIMARY,
                    anchor="center",
                    font=ctk.CTkFont(
                        size=11
                    )
                )

                label.grid(
                    row=row_index,
                    column=column,
                    padx=5,
                    pady=7,
                    sticky="ew"
                )

    # -----------------------------------------------------
    # Close
    # -----------------------------------------------------

    close_button = ctk.CTkButton(
        window,
        text="Close",
        height=38,
        width=120,
        corner_radius=8,
        command=window.destroy
    )

    close_button.grid(
        row=2,
        column=0,
        pady=(0, 18)
    )


# =========================================================
# CLEAR HISTORY
# =========================================================

def clear_history():

    if not os.path.exists(
        HISTORY_FILE
    ):

        messagebox.showinfo(
            "History",
            "There is no prediction history to clear."
        )

        return

    confirm = messagebox.askyesno(
        "Clear History",
        (
            "Are you sure you want to delete "
            "all prediction history?"
        )
    )

    if not confirm:

        return

    try:

        os.remove(
            HISTORY_FILE
        )

        load_history()

        messagebox.showinfo(
            "History Cleared",
            "All prediction history has been deleted."
        )

    except Exception as e:

        messagebox.showerror(
            "Error",
            str(e)
        )


# =========================================================
# EXPORT HISTORY
# =========================================================

def export_history():

    if not os.path.exists(
        HISTORY_FILE
    ):

        messagebox.showwarning(
            "No History",
            "There is no prediction history to export."
        )

        return

    save_path = filedialog.asksaveasfilename(

        title="Export Prediction History",

        defaultextension=".csv",

        filetypes=[
            (
                "CSV files",
                "*.csv"
            ),
            (
                "All files",
                "*.*"
            )
        ]
    )

    if not save_path:

        return

    try:

        with open(
            HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as source:

            data = source.read()

        with open(
            save_path,
            "w",
            encoding="utf-8",
            newline=""
        ) as destination:

            destination.write(
                data
            )

        messagebox.showinfo(
            "Export Successful",
            (
                "Prediction history exported successfully.\n\n"
                f"File:\n{save_path}"
            )
        )

    except Exception as e:

        messagebox.showerror(
            "Export Error",
            str(e)
        )


# =========================================================
# UPDATE RESULT
# =========================================================

def update_result(data):

    prediction = data[
        "prediction"
    ]

    confidence = data[
        "confidence_pct"
    ]

    probabilities = data[
        "class_probabilities"
    ]

    distances = data[
        "nearest_neighbors_distances"
    ]

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction_label.configure(
        text=prediction.upper()
    )

    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    confidence_label.configure(
        text=f"Confidence: {confidence}%"
    )

    # -----------------------------------------------------
    # Probabilities
    # -----------------------------------------------------

    malignant_probability = float(
        probabilities.get(
            "malignant",
            0
        )
    )

    benign_probability = float(
        probabilities.get(
            "benign",
            0
        )
    )

    malignant_label.configure(
        text=(
            f"Malignant: "
            f"{malignant_probability * 100:.2f}%"
        )
    )

    benign_label.configure(
        text=(
            f"Benign: "
            f"{benign_probability * 100:.2f}%"
        )
    )

    malignant_bar.set(
        malignant_probability
    )

    benign_bar.set(
        benign_probability
    )

    # -----------------------------------------------------
    # Nearest neighbors
    # -----------------------------------------------------

    first_distances = distances[:3]

    neighbors_label.configure(
        text=(
            ", ".join(
                f"{d:.4f}"
                for d in first_distances
            )
        )
    )


# =========================================================
# PREDICT
# =========================================================

def predict():

    try:

        # -----------------------------------------------
        # Get 30 features
        # -----------------------------------------------

        features = get_features()

        # -----------------------------------------------
        # Patient information
        # -----------------------------------------------

        patient_id = (
            patient_id_entry
            .get()
            .strip()
        )

        patient_name = (
            patient_name_entry
            .get()
            .strip()
        )

        if patient_id == "":
            patient_id = "N/A"

        if patient_name == "":
            patient_name = "Unknown"

        # -----------------------------------------------
        # Loading
        # -----------------------------------------------

        prediction_label.configure(
            text="Predicting..."
        )

        app.update_idletasks()

        # -----------------------------------------------
        # POST request
        # -----------------------------------------------

        response = requests.post(
            f"{API_URL}/predict",

            json={
                "features": features
            },

            timeout=10
        )

        # -----------------------------------------------
        # Error
        # -----------------------------------------------

        if response.status_code != 200:

            try:

                error_data = response.json()

            except Exception:

                error_data = response.text

            messagebox.showerror(
                "Prediction Error",

                (
                    f"HTTP {response.status_code}\n\n"
                    f"{error_data}"
                )
            )

            prediction_label.configure(
                text="—"
            )

            return

        # -----------------------------------------------
        # JSON result
        # -----------------------------------------------

        data = response.json()

        # -----------------------------------------------
        # Display
        # -----------------------------------------------

        update_result(
            data
        )

        # -----------------------------------------------
        # Save history
        # -----------------------------------------------

        save_history(
            patient_id,
            patient_name,
            data["prediction"],
            data["confidence_pct"]
        )

        # -----------------------------------------------
        # Refresh history
        # -----------------------------------------------

        load_history()

    except ValueError as e:

        messagebox.showwarning(
            "Invalid Input",
            str(e)
        )

    except requests.exceptions.ConnectionError:

        messagebox.showerror(
            "API Connection Error",

            (
                "Cannot connect to FastAPI.\n\n"
                "Start the API server first:\n\n"
                "uvicorn main:app --reload"
            )
        )

    except requests.exceptions.Timeout:

        messagebox.showerror(
            "Timeout",
            "The API request took too long."
        )

    except Exception as e:

        messagebox.showerror(
            "Error",
            str(e)
        )

        print(
            "Prediction error:",
            e
        )


# =========================================================
# CSV LOADER
# =========================================================

def load_csv():

    file_path = filedialog.askopenfilename(

        title="Select Patient CSV File",

        filetypes=[
            (
                "CSV files",
                "*.csv"
            ),
            (
                "All files",
                "*.*"
            )
        ]
    )

    if not file_path:

        return

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8-sig"
        ) as file:

            reader = csv.DictReader(
                file
            )

            rows = list(
                reader
            )

            columns = reader.fieldnames

        if not rows:

            messagebox.showwarning(
                "Empty CSV",
                "The selected CSV file has no data."
            )

            return

        if columns is None:

            messagebox.showerror(
                "CSV Error",
                "Could not read CSV column names."
            )

            return

        missing_features = [

            feature

            for feature in FEATURE_NAMES

            if feature not in columns
        ]

        if missing_features:

            messagebox.showerror(

                "Missing CSV Columns",

                (
                    "The CSV file is missing "
                    "some required feature columns.\n\n"

                    "Missing columns:\n"

                    +
                    "\n".join(
                        missing_features
                    )
                )
            )

            return

        has_patient_id = (
            "patient_id"
            in columns
        )

        has_patient_name = (
            "patient_name"
            in columns
        )

        if len(rows) == 1:

            selected_row = rows[0]

        else:

            selected_row = choose_csv_row(
                rows
            )

        if selected_row is None:

            return

        patient_id_entry.delete(
            0,
            "end"
        )

        if has_patient_id:

            patient_id_entry.insert(
                0,
                selected_row.get(
                    "patient_id",
                    ""
                )
            )

        patient_name_entry.delete(
            0,
            "end"
        )

        if has_patient_name:

            patient_name_entry.insert(
                0,
                selected_row.get(
                    "patient_name",
                    ""
                )
            )

        for entry, feature_name in zip(
            feature_entries,
            FEATURE_NAMES
        ):

            entry.delete(
                0,
                "end"
            )

            value = selected_row.get(
                feature_name,
                ""
            )

            entry.insert(
                0,
                value
            )

        status_label.configure(
            text=(
                "● CSV Loaded     "
                f"{os.path.basename(file_path)}"
            ),
            text_color=SUCCESS_COLOR
        )

        messagebox.showinfo(
            "CSV Loaded",
            (
                "Patient data loaded successfully.\n\n"
                "You can now click PREDICT."
            )
        )

    except UnicodeDecodeError:

        messagebox.showerror(
            "CSV Encoding Error",
            (
                "Could not read this CSV file.\n\n"
                "Please save the CSV as UTF-8."
            )
        )

    except Exception as e:

        messagebox.showerror(
            "CSV Error",
            str(e)
        )


# =========================================================
# CSV ROW SELECTOR
# =========================================================

def choose_csv_row(rows):

    window = ctk.CTkToplevel(
        app
    )

    window.title(
        "Select Patient"
    )

    window.geometry(
        "700x500"
    )

    window.configure(
        fg_color=BG_COLOR
    )

    window.transient(
        app
    )

    window.grab_set()

    window.grid_columnconfigure(
        0,
        weight=1
    )

    window.grid_rowconfigure(
        1,
        weight=1
    )

    title = ctk.CTkLabel(
        window,
        text="Select Patient",
        text_color=TEXT_PRIMARY,
        font=ctk.CTkFont(
            size=20,
            weight="bold"
        )
    )

    title.grid(
        row=0,
        column=0,
        padx=20,
        pady=(20, 5)
    )

    subtitle = ctk.CTkLabel(
        window,
        text="Choose a patient from the selected CSV file",
        text_color=TEXT_SECONDARY,
        font=ctk.CTkFont(
            size=12
        )
    )

    subtitle.grid(
        row=0,
        column=0,
        padx=20,
        pady=(55, 10)
    )

    scroll = ctk.CTkScrollableFrame(
        window,
        fg_color=CARD_COLOR,
        corner_radius=10
    )

    scroll.grid(
        row=1,
        column=0,
        sticky="nsew",
        padx=20,
        pady=10
    )

    selected_row = {
        "value": None
    }

    for index, row in enumerate(rows):

        patient_id = row.get(
            "patient_id",
            f"Row {index + 1}"
        )

        patient_name = row.get(
            "patient_name",
            ""
        )

        button_text = (
            f"{index + 1:02d}     "
            f"{patient_id}     "
            f"{patient_name}"
        )

        def select_row(
            current_row=row
        ):

            selected_row[
                "value"
            ] = current_row

            window.destroy()

        button = ctk.CTkButton(
            scroll,
            text=button_text,
            command=select_row,
            height=42,
            corner_radius=8,
            fg_color=SOFT_BLUE,
            hover_color="#DBEAFE",
            text_color=TEXT_PRIMARY
        )

        button.pack(
            fill="x",
            padx=10,
            pady=5
        )

    def cancel():

        selected_row[
            "value"
        ] = None

        window.destroy()

    cancel_button = ctk.CTkButton(
        window,
        text="Cancel",
        command=cancel,
        height=38,
        width=120
    )

    cancel_button.grid(
        row=2,
        column=0,
        padx=20,
        pady=15
    )

    app.wait_window(
        window
    )

    return selected_row[
        "value"
    ]


# =========================================================
# BUTTONS
# =========================================================

check_button = ctk.CTkButton(
    button_frame,
    text="Check API",
    height=40,
    corner_radius=8,
    fg_color="#E5E7EB",
    hover_color="#D1D5DB",
    text_color=TEXT_PRIMARY,
    command=check_api
)

check_button.grid(
    row=0,
    column=0,
    padx=3,
    pady=5,
    sticky="ew"
)


sample_button = ctk.CTkButton(
    button_frame,
    text="Load Sample",
    height=40,
    corner_radius=8,
    fg_color="#E5E7EB",
    hover_color="#D1D5DB",
    text_color=TEXT_PRIMARY,
    command=load_sample
)

sample_button.grid(
    row=0,
    column=1,
    padx=3,
    pady=5,
    sticky="ew"
)


csv_button = ctk.CTkButton(
    button_frame,
    text="Load CSV",
    height=40,
    corner_radius=8,
    fg_color="#E5E7EB",
    hover_color="#D1D5DB",
    text_color=TEXT_PRIMARY,
    command=load_csv
)

csv_button.grid(
    row=0,
    column=2,
    padx=3,
    pady=5,
    sticky="ew"
)


predict_button = ctk.CTkButton(
    button_frame,
    text="PREDICT",
    height=40,
    corner_radius=8,
    fg_color=ACCENT_COLOR,
    hover_color=ACCENT_HOVER,
    text_color="white",
    font=ctk.CTkFont(
        size=13,
        weight="bold"
    ),
    command=predict
)

predict_button.grid(
    row=0,
    column=3,
    padx=3,
    pady=5,
    sticky="ew"
)


clear_button = ctk.CTkButton(
    button_frame,
    text="Clear",
    height=40,
    corner_radius=8,
    fg_color="#E5E7EB",
    hover_color="#D1D5DB",
    text_color=TEXT_PRIMARY,
    command=clear_all
)

clear_button.grid(
    row=0,
    column=4,
    padx=3,
    pady=5,
    sticky="ew"
)


view_history_button = ctk.CTkButton(
    button_frame,
    text="View History",
    height=40,
    corner_radius=8,
    fg_color="#E5E7EB",
    hover_color="#D1D5DB",
    text_color=TEXT_PRIMARY,
    command=view_full_history
)

view_history_button.grid(
    row=0,
    column=5,
    padx=3,
    pady=5,
    sticky="ew"
)


clear_history_button = ctk.CTkButton(
    button_frame,
    text="Clear History",
    height=40,
    corner_radius=8,
    fg_color="#FEE2E2",
    hover_color="#FECACA",
    text_color=DANGER_COLOR,
    command=clear_history
)

clear_history_button.grid(
    row=0,
    column=6,
    padx=3,
    pady=5,
    sticky="ew"
)


export_history_button = ctk.CTkButton(
    button_frame,
    text="Export CSV",
    height=40,
    corner_radius=8,
    fg_color="#E5E7EB",
    hover_color="#D1D5DB",
    text_color=TEXT_PRIMARY,
    command=export_history
)

export_history_button.grid(
    row=0,
    column=7,
    padx=3,
    pady=5,
    sticky="ew"
)


# =========================================================
# STARTUP
# =========================================================

load_history()


app.after(
    500,
    check_api
)


# =========================================================
# RUN APP
# =========================================================

app.mainloop()
