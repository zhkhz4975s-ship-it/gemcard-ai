

import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(
    page_title="Gemcard AI",
    page_icon="💎",
    layout="wide"
)

st.title("💎 GEMCARD AI")
st.subheader("Professional TCG Vision & Centering Engine")
st.caption("Version 0.4 – Integrated Experimental Engine")

st.write(
    "Sistema sperimentale per l'analisi visiva di "
    "Pokémon, One Piece e altri Trading Card Games."
)

# --------------------------------------------------
# IMAGE INPUT
# --------------------------------------------------

card_name = st.text_input(
    "Nome della carta",
    placeholder="Es. Charizard EX"
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 FRONTE")
    front = st.file_uploader(
        "Carica fronte carta",
        type=["jpg", "jpeg", "png"],
        key="front"
    )

with col2:
    st.subheader("📸 RETRO")
    back = st.file_uploader(
        "Carica retro carta",
        type=["jpg", "jpeg", "png"],
        key="back"
    )

# --------------------------------------------------
# IMAGE PREPARATION
# --------------------------------------------------

def load_image(upload):
    image = Image.open(upload)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def order_points(points):
    points = np.asarray(points, dtype=np.float32)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).flatten()

    return np.array([
        points[np.argmin(sums)],
        points[np.argmin(diffs)],
        points[np.argmax(sums)],
        points[np.argmax(diffs)]
    ], dtype=np.float32)


def card_geometry(points):
    p = order_points(points)

    top = np.linalg.norm(p[1] - p[0])
    bottom = np.linalg.norm(p[2] - p[3])
    left = np.linalg.norm(p[3] - p[0])
    right = np.linalg.norm(p[2] - p[1])

    width = (top + bottom) / 2
    height = (left + right) / 2

    if min(width, height) < 50:
        return False

    ratio = min(width, height) / max(width, height)

    return 0.60 <= ratio <= 0.83


# --------------------------------------------------
# AUTOMATIC CARD DETECTION
# --------------------------------------------------

def detect_card(image):

    original = np.array(image)
    h, w = original.shape[:2]

    scale = min(1.0, 1400 / max(h, w))

    small = cv2.resize(
        original,
        None,
        fx=scale,
        fy=scale
    )

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_RGB2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    hsv = cv2.cvtColor(
        small,
        cv2.COLOR_RGB2HSV
    )

    kernel = np.ones((5, 5), np.uint8)

    masks = []

    # Canny with multiple thresholds
    for low, high in [(20, 80), (40, 120), (60, 180)]:

        edges = cv2.Canny(
            blurred,
            low,
            high
        )

        edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        masks.append(edges)

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        8
    )

    masks.append(adaptive)

    # Background difference
    border = np.concatenate([
        small[0, :, :],
        small[-1, :, :],
        small[:, 0, :],
        small[:, -1, :]
    ])

    background = np.median(
        border.astype(np.float32),
        axis=0
    )

    distance = np.linalg.norm(
        small.astype(np.float32) - background,
        axis=2
    )

    for threshold in [12, 20, 30, 40]:

        foreground = (
            distance > threshold
        ).astype(np.uint8) * 255

        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=3
        )

        masks.append(foreground)

    # Find possible card contours
    candidates = []
    image_area = small.shape[0] * small.shape[1]

    for mask in masks:

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < image_area * 0.06:
                continue

            if area > image_area * 0.98:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            approx = cv2.approxPolyDP(
                contour,
                0.015 * perimeter,
                True
            )

            if len(approx) != 4:
                continue

            points = approx.reshape(4, 2)

            if not cv2.isContourConvex(approx):
                continue

            if not card_geometry(points):
                continue

            rect_area = cv2.contourArea(approx)

            rectangularity = min(
                1.0,
                rect_area / (area + 1e-6)
            )

            score = (
                (area / image_area) *
                rectangularity
            )

            candidates.append(
                (score, points)
            )

    if not candidates:
        return original, None, False

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    points = order_points(
        candidates[0][1]
    ) / scale

    return original, points, True


# --------------------------------------------------
# PERSPECTIVE CORRECTION
# --------------------------------------------------

def rectify_card(array, points):

    points = order_points(points)

    width = 744
    height = 1038

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(
        points,
        destination
    )

    warped = cv2.warpPerspective(
        array,
        matrix,
        (width, height)
    )

    return warped


# --------------------------------------------------
# CENTERING ENGINE
# --------------------------------------------------

def analyze_centering(card):

    gray = cv2.cvtColor(
        card,
        cv2.COLOR_RGB2GRAY
    )

    h, w = gray.shape

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # Search for candidate internal boundaries
    vertical_profile = np.mean(
        np.abs(gx[int(h * 0.20):int(h * 0.80), :]),
        axis=0
    )

    horizontal_profile = np.mean(
        np.abs(gy[:, int(w * 0.20):int(w * 0.80)]),
        axis=1
    )

    def find_margin_lines(profile, length):

        profile = cv2.GaussianBlur(
            profile.reshape(1, -1),
            (9, 1),
            0
        ).flatten()

        low = int(length * 0.03)
        high = int(length * 0.30)

        left_zone = profile[low:high]
        right_zone = profile[length-high:length-low]

        if len(left_zone) == 0 or len(right_zone) == 0:
            return None

        left = low + int(np.argmax(left_zone))

        right = (
            length - high +
            int(np.argmax(right_zone))
        )

        if right <= left:
            return None

        baseline = float(np.median(profile))
        noise = float(np.std(profile)) + 1e-6

        left_strength = float(profile[left])
        right_strength = float(profile[right])

        left_conf = max(
            0,
            (left_strength - baseline) / noise
        )

        right_conf = max(
            0,
            (right_strength - baseline) / noise
        )

        confidence = min(
            100,
            (min(left_conf, right_conf) / 5) * 100
        )

        if confidence < 25:
            return None

        first_margin = left
        second_margin = length - right

        total = first_margin + second_margin

        if total <= 0:
            return None

        first_percent = (
            first_margin / total
        ) * 100

        second_percent = 100 - first_percent

        return {
            "first": round(first_margin, 1),
            "second": round(second_margin, 1),
            "first_percent": round(first_percent, 1),
            "second_percent": round(second_percent, 1),
            "confidence": round(confidence, 1)
        }

    horizontal = find_margin_lines(
        vertical_profile,
        w
    )

    vertical = find_margin_lines(
        horizontal_profile,
        h
    )

    return {
        "left_right": horizontal,
        "top_bottom": vertical
    }


# --------------------------------------------------
# CORNER, EDGE AND SURFACE ANALYSIS
# --------------------------------------------------

def analyze_card(card):

    gray = cv2.cvtColor(
        card,
        cv2.COLOR_RGB2GRAY
    )

    hsv = cv2.cvtColor(
        card,
        cv2.COLOR_RGB2HSV
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    brightness = float(
        np.mean(gray)
    )

    margin = 30
    size = 90

    border_pixels = np.concatenate([
        hsv[:margin].reshape(-1, 3),
        hsv[-margin:].reshape(-1, 3),
        hsv[:, :margin].reshape(-1, 3),
        hsv[:, -margin:].reshape(-1, 3)
    ])

    white = (
        (border_pixels[:, 2] > 205) &
        (border_pixels[:, 1] < 75)
    )

    border_indicator = float(
        np.mean(white) * 100
    )

    corners = [
        hsv[:size, :size],
        hsv[:size, -size:],
        hsv[-size:, :size],
        hsv[-size:, -size:]
    ]

    corner_results = []

    for corner in corners:

        white = (
            (corner[:, :, 2] > 205) &
            (corner[:, :, 1] < 75)
        )

        corner_results.append(
            round(float(np.mean(white) * 100), 2)
        )

    smooth = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    residual = cv2.absdiff(
        gray,
        smooth
    )

    surface_indicator = float(
        np.mean(residual > 35) * 100
    )

    centering = analyze_centering(card)

    return {
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "border_indicator": round(border_indicator, 2),
        "corners": corner_results,
        "surface_indicator": round(surface_indicator, 2),
        "centering": centering
    }


# --------------------------------------------------
# MANUAL FALLBACK
# --------------------------------------------------

with st.expander("Impostazioni di ritaglio di emergenza"):

    st.write(
        "Usa questa funzione soltanto se il riconoscimento "
        "automatico non individua la carta."
    )

    manual_mode = st.checkbox(
        "Abilita ritaglio manuale"
    )

    crop_x = st.slider(
        "Posizione X (%)",
        0, 70, 10
    )

    crop_y = st.slider(
        "Posizione Y (%)",
        0, 70, 10
    )

    crop_w = st.slider(
        "Larghezza (%)",
        20, 100, 80
    )

    crop_h = st.slider(
        "Altezza (%)",
        20, 100, 80
    )


def manual_crop(image):

    array = np.array(image)

    h, w = array.shape[:2]

    x1 = int(w * crop_x / 100)
    y1 = int(h * crop_y / 100)

    x2 = min(
        w,
        int(w * (crop_x + crop_w) / 100)
    )

    y2 = min(
        h,
        int(h * (crop_y + crop_h) / 100)
    )

    if x2 <= x1 or y2 <= y1:
        return None

    crop = array[y1:y2, x1:x2]

    return cv2.resize(
        crop,
        (744, 1038)
    )


# --------------------------------------------------
# PROCESSING
# --------------------------------------------------

def process_image(upload):

    image = load_image(upload)

    array, points, detected = detect_card(image)

    if detected:

        card = rectify_card(
            array,
            points
        )

        method = "Automatic Detection"

    elif manual_mode:

        card = manual_crop(image)

        if card is None:
            return {
                "detected": False,
                "original": image
            }

        method = "Manual Crop"

    else:

        return {
            "detected": False,
            "original": image
        }

    metrics = analyze_card(card)

    return {
        "detected": True,
        "original": image,
        "card": card,
        "method": method,
        "metrics": metrics
    }


# --------------------------------------------------
# REPORT
# --------------------------------------------------

def render_result(title, result):

    st.header(title)

    if not result["detected"]:

        st.error(
            "Carta non riconosciuta. "
            "Prova una fotografia con maggiore contrasto "
            "oppure abilita il ritaglio manuale."
        )

        st.image(
            result["original"],
            use_container_width=True
        )

        return

    data = result["metrics"]

    st.success(
        "Immagine elaborata: " + result["method"]
    )

    c1, c2 = st.columns(2)

    with c1:
        st.image(
            result["original"],
            caption="Original Image",
            use_container_width=True
        )

    with c2:
        st.image(
            result["card"],
            caption="Processed Card",
            use_container_width=True
        )

    st.subheader("📊 Analisi visiva")

    a, b, c = st.columns(3)

    a.metric(
        "Nitidezza",
        data["sharpness"]
    )

    b.metric(
        "Luminosità",
        data["brightness"]
    )

    c.metric(
        "Indicatore pixel chiari sui bordi",
        f"{data['border_indicator']}%"
    )

    st.subheader("🔍 Analisi angoli")

    labels = [
        "Alto sinistro",
        "Alto destro",
        "Basso sinistro",
        "Basso destro"
    ]

    for label, value in zip(
        labels,
        data["corners"]
    ):

        st.write(
            f"**{label}:** {value}%"
        )

    st.metric(
        "Indicatore anomalie superficiali",
        f"{data['surface_indicator']}%"
    )

    # Centering
    st.divider()

    st.subheader("🎯 AUTOMATIC CENTERING ENGINE")

    centering = data["centering"]

    lr = centering["left_right"]
    tb = centering["top_bottom"]

    if lr:

        st.write("**Margini sinistro / destro**")

        st.metric(
            "Rapporto candidato",
            f"{lr['first_percent']} / {lr['second_percent']}"
        )

        st.caption(
            f"Affidabilità del rilevamento: "
            f"{lr['confidence']}%"
        )

    else:

        st.warning(
            "Margini sinistro e destro non identificati "
            "con sufficiente affidabilità."
        )

    if tb:

        st.write("**Margini superiore / inferiore**")

        st.metric(
            "Rapporto candidato",
            f"{tb['first_percent']} / {tb['second_percent']}"
        )

        st.caption(
            f"Affidabilità del rilevamento: "
            f"{tb['confidence']}%"
        )

    else:

        st.warning(
            "Margini superiore e inferiore non identificati "
            "con sufficiente affidabilità."
        )


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

if st.button(
    "🔍 ANALIZZA CARTA",
    type="primary",
    use_container_width=True
):

    if not front or not back:

        st.error(
            "Carica sia il fronte che il retro."
        )

    else:

        try:

            with st.spinner(
                "Gemcard AI sta analizzando fronte e retro..."
            ):

                front_result = process_image(front)
                back_result = process_image(back)

            st.success(
                "Elaborazione completata."
            )

            if card_name:
                st.write("**Carta:**", card_name)

            render_result(
                "📸 ANALISI FRONTE",
                front_result
            )

            st.divider()

            render_result(
                "📸 ANALISI RETRO",
                back_result
            )

            st.divider()

            st.header("💎 GEMCARD GRADING ENGINE")

            st.warning(
                "PSA e BGS: voti non ancora disponibili. "
                "Il motore richiede calibrazione e validazione "
                "su carte già gradate."
            )

            st.info(
                "Le misure di centratura sono margini candidati "
                "rilevati tramite gradienti. Non sono ancora "
                "misure certificate dei margini di stampa."
            )

        except Exception as error:

            st.error(
                f"Errore durante l'analisi: {error}"
            )


st.divider()

st.caption(
    "GEMCARD AI © 2026 – Gemcard Infinity Collection"
)
