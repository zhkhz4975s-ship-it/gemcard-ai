
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Gemcard AI",
    page_icon="💎",
    layout="wide"
)

st.title("💎 GEMCARD AI")
st.subheader("TCG Vision Engine – Prototype 0.1")

st.write(
    "Motore proprietario Gemcard per l'analisi "
    "visiva delle carte collezionabili."
)

st.divider()

card_name = st.text_input(
    "Nome della carta",
    placeholder="Es. Charizard EX"
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Fronte carta")
    front = st.file_uploader(
        "Carica il fronte",
        type=["jpg", "jpeg", "png"],
        key="front"
    )

with col2:
    st.subheader("📸 Retro carta")
    back = st.file_uploader(
        "Carica il retro",
        type=["jpg", "jpeg", "png"],
        key="back"
    )


def analyze_image(upload):
    image = Image.open(upload).convert("RGB")
    array = np.array(image)

    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

    sharpness = cv2.Laplacian(
        gray, cv2.CV_64F
    ).var()

    brightness = float(np.mean(gray))

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0) * 100)

    return {
        "image": image,
        "width": image.width,
        "height": image.height,
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "edge_density": round(edge_density, 2)
    }


if st.button("🔍 ANALIZZA CARTA", type="primary"):

    if not front or not back:
        st.error("Carica sia il fronte che il retro.")
    else:
        with st.spinner("Analisi delle immagini in corso..."):

            front_data = analyze_image(front)
            back_data = analyze_image(back)

            st.success("Acquisizione e analisi completate!")

            st.header("Risultati Gemcard Vision")

            if card_name:
                st.write("Carta:", card_name)

            c1, c2 = st.columns(2)

            for col, title, data in [
                (c1, "FRONTE", front_data),
                (c2, "RETRO", back_data)
            ]:
                with col:
                    st.subheader(title)
                    st.image(data["image"], use_container_width=True)

                    st.write(
                        f"Dimensioni: {data['width']} × "
                        f"{data['height']} px"
                    )
                    st.metric(
                        "Nitidezza immagine",
                        data["sharpness"]
                    )
                    st.metric(
                        "Luminosità",
                        data["brightness"]
                    )
                    st.metric(
                        "Densità dei bordi",
                        f"{data['edge_density']}%"
                    )

            st.info(
                "Questa versione analizza la qualità delle "
                "immagini. Non assegna ancora voti PSA o BGS."
            )

st.divider()

st.caption(
    "GEMCARD AI © 2026 – Proprietary TCG Vision Engine"
)
