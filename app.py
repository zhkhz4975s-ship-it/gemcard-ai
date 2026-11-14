
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO

st.set_page_config(
    page_title="Gemcard AI",
    page_icon="💎",
    layout="wide"
)

st.title("💎 GEMCARD AI")
st.subheader("TCG Vision Engine – Prototype 0.2")

st.write(
    "Analisi preliminare di carte collezionabili "
    "con riconoscimento geometrico e controllo visivo."
)


def load_image(upload):
    image = Image.open(upload)
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image


def order_points(points):
    points = np.array(points, dtype="float32")
    result = np.zeros((4, 2), dtype="float32")

    s = points.sum(axis=1)
    d = np.diff(points, axis=1).flatten()

    result[0] = points[np.argmin(s)]  # top-left
    result[2] = points[np.argmax(s)]  # bottom-right
    result[1] = points[np.argmin(d)]  # top-right
    result[3] = points[np.argmax(d)]  # bottom-left

    return result


def detect_card(image):
    array = np.array(image)
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = image.width * image.height
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < image_area * 0.08:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(
            contour, 0.02 * perimeter, True
        )

        if len(approx) != 4:
            continue

        points = order_points(approx.reshape(4, 2))

        width = (
            np.linalg.norm(points[0] - points[1])
            + np.linalg.norm(points[2] - points[3])
        ) / 2

        height = (
            np.linalg.norm(points[0] - points[3])
            + np.linalg.norm(points[1] - points[2])
        ) / 2

        if width == 0 or height == 0:
            continue

        ratio = min(width, height) / max(width, height)

        # Rapporto indicativo delle carte TCG standard
        if 0.58 <= ratio <= 0.85:
            candidates.append((area, points))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    points = candidates[0][1]

    return array, points


def rectify_card(array, points):
    points = order_points(points)

    width = 744
    height = 1038

    destination = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(
        points, destination
    )

    warped = cv2.warpPerspective(
        array, matrix, (width, height)
    )

    return warped


def analyze_card(warped):
    gray = cv2.cvtColor(
        warped, cv2.COLOR_RGB2GRAY
    )

    hsv = cv2.cvtColor(
        warped, cv2.COLOR_RGB2HSV
    )

    sharpness = float(
        cv2.Laplacian(
            gray, cv2.CV_64F
        ).var()
    )

    brightness = float(np.mean(gray))

    # Analisi preliminare dei bordi esterni
    margin = 35

    top = hsv[:margin, :, :]
    bottom = hsv[-margin:, :, :]
    left = hsv[:, :margin, :]
    right = hsv[:, -margin:, :]

    border = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ])

    # Pixel molto chiari e poco saturi:
    # possibile whitening, non prova certa di danno
    white_pixels = (
        (border[:, 2] > 205) &
        (border[:, 1] < 75)
    )

    whitening = float(
        np.mean(white_pixels) * 100
    )

    # Analisi delle quattro zone d'angolo
    corner_size = 90

    corners = [
        hsv[:corner_size, :corner_size],
        hsv[:corner_size, -corner_size:],
        hsv[-corner_size:, :corner_size],
        hsv[-corner_size:, -corner_size:]
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

    # Ricerca preliminare di anomalie superficiali
    smooth = cv2.GaussianBlur(
        gray, (5, 5), 0
    )

    residual = cv2.absdiff(gray, smooth)

    anomaly_ratio = float(
        np.mean(residual > 35) * 100
    )

    return {
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "whitening": round(whitening, 2),
        "corners": corner_results,
        "surface_anomaly": round(anomaly_ratio, 2)
    }


st.divider()

card_name = st.text_input(
    "Nome della carta",
    placeholder="Es. Charizard EX"
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Fronte")
    front = st.file_uploader(
        "Carica fotografia fronte",
        type=["jpg", "jpeg", "png"],
        key="front"
    )

with col2:
    st.subheader("📸 Retro")
    back = st.file_uploader(
        "Carica fotografia retro",
        type=["jpg", "jpeg", "png"],
        key="back"
    )


if st.button(
    "🔍 ANALIZZA CARTA",
    type="primary",
    use_container_width=True
):

    if not front or not back:
        st.error("Carica fronte e retro.")
        st.stop()

    try:
        with st.spinner("Gemcard AI sta analizzando la carta..."):

            results = []

            for upload in [front, back]:
                image = load_image(upload)
                array, points = detect_card(image)

                if points is None:
                    results.append({
                        "detected": False,
                        "image": image
                    })
                    continue

                warped = rectify_card(array, points)

                metrics = analyze_card(warped)

                results.append({
                    "detected": True,
                    "image": image,
                    "warped": warped,
                    "metrics": metrics
                })

        st.header("💎 Risultati Gemcard Vision")

        if card_name:
            st.write("Carta:", card_name)

        for title, result in zip(
            ["FRONTE", "RETRO"], results
        ):

            st.subheader(title)

            if not result["detected"]:
                st.warning(
                    "Carta non riconosciuta con sufficiente "
                    "affidabilità. Prova una fotografia più "
                    "nitida, con tutti e quattro gli angoli visibili."
                )
                st.image(
                    result["image"],
                    use_container_width=True
                )
                continue

            c1, c2 = st.columns(2)

            with c1:
                st.image(
                    result["image"],
                    caption="Fotografia originale",
                    use_container_width=True
                )

            with c2:
                st.image(
                    result["warped"],
                    caption="Carta raddrizzata da Gemcard AI",
                    use_container_width=True
                )

            data = result["metrics"]

            a, b, c = st.columns(3)

            a.metric("Nitidezza", data["sharpness"])
            b.metric("Luminosità", data["brightness"])
            c.metric(
                "Pixel chiari sul bordo",
                f"{data['whitening']}%"
            )

            st.write("**Analisi dei quattro angoli**")

            labels = [
                "Alto sinistro",
                "Alto destro",
                "Basso sinistro",
                "Basso destro"
            ]

            for label, value in zip(
                labels, data["corners"]
            ):
                st.write(f"{label}: {value}%")

            st.metric(
                "Anomalie di superficie rilevate",
                f"{data['surface_anomaly']}%"
            )

        st.divider()

        st.header("🎯 Centering e Grading")

        st.info(
            "Il riconoscimento geometrico è preliminare. "
            "La centratura della stampa deve essere misurata "
            "riconoscendo i margini grafici specifici della carta."
        )

        st.warning(
            "PSA e BGS: voto non ancora disponibile. "
            "Le misure attuali non sono sufficienti per "
            "assegnare un voto attendibile."
        )

        st.caption(
            "Gemcard AI 0.2 – Experimental Vision Engine. "
            "Non è un servizio ufficiale PSA o Beckett."
        )

    except Exception as error:
        st.error(f"Errore durante l'analisi: {error}")


st.divider()

st.caption(
    "GEMCARD AI © 2026 – Gemcard Infinity Collection"
)
