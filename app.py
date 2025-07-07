# app.py reorganizado com inputs abaixo da curva

import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_pdf import PdfPages

# --- Funções auxiliares ---
def logmar_from_snellen(snellen_str):
    try:
        base, value = snellen_str.strip().split('/')
        base = int(base)
        value = float(value)
        return round(np.log10(value / base), 2)
    except:
        return None

def snellen_from_logmar(logmar):
    if logmar is None:
        return ""
    try:
        acuity = 10 ** (-logmar)
        snellen = round(20 / acuity)
        return f"20/{snellen}" if snellen > 10 else "20/10"
    except:
        return "Inválido"

def acuity_from_logmar(logmar):
    return 10 ** (-logmar)

def get_blur_radius(acuity):
    if acuity >= 1.0:
        return 0
    return int((1.0 - acuity) * 15)

# --- Carregar imagem original e máscaras ---
img_path = "A_high-resolution_digital_photograph_captures_an_o.png"
original_img = Image.open(img_path).convert("RGB")

mask_files = {
    "infinito": "mask_infinito.png",
    "pessoa": "mask_pessoa.png",
    "laptop": "mask_laptop.png",
    "celular": "mask_celular.png",
    "livro": "mask_livro.png"
}

def load_binary_mask(path, size):
    mask = Image.open(path).convert("L").resize(size)
    mask = mask.point(lambda p: 255 if p > 128 else 0)
    return mask

masks = {k: load_binary_mask(v, original_img.size) for k, v in mask_files.items()}

# --- Zonas ordenadas decrescentemente por dioptria para entrada da curva ---
zonas_ordenadas = [
    ("+1.50D", -1.5), ("+1.00D", -1.0), ("+0.50D", -0.5),
    ("0.00D", 0.0), ("-0.50D", 0.5), ("-1.00D", 1.0),
    ("-1.50D", 1.5), ("-2.00D", 2.0), ("-2.50D", 2.5), ("-3.00D", 3.0)
]

zonas_com_mascara_set = {
    "0.00D": "infinito",
    "-1.00D": "pessoa",
    "-2.00D": "laptop",
    "-2.50D": "celular",
    "-3.00D": "livro"
}

# --- Interface ---
st.set_page_config(layout="wide")
st.title("🔍 Simulador de Desfoque Visual + Curva de Defocus")

# --- Inicializa os inputs com valores default ---
inputs = {}
x = []
logmars = []

for label, d in zonas_ordenadas:
    inputs[label] = "20/20"
    lm = logmar_from_snellen(inputs[label])
    if lm is None:
        lm = 1.0
    x.append(d)
    logmars.append(lm)

# --- Plotar curva inicial com valores default ---
fig, ax1 = plt.subplots(figsize=(8, 4))
x_array, logmars_array = zip(*sorted(zip(x, logmars)))
ax1.plot(x_array, logmars_array, 'o-', color='blue')
ax1.set_xlabel("Defocus (D)")
ax1.set_ylabel("logMAR", color='blue')
ax1.set_ylim(1.1, -0.3)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)
ax1.set_title("Curva de Acuidade Visual (logMAR e Snellen)")

xtick_labels = [f"{-val:+.1f}" for val in x_array]
ax1.set_xticks(x_array)
ax1.set_xticklabels(xtick_labels)

logmar_ticks = np.round(np.arange(-0.3, 1.1, 0.1), 2)
snellen_labels = [snellen_from_logmar(lm) for lm in logmar_ticks]
ax2 = ax1.twinx()
ax2.set_ylabel("Snellen", color='green')
ax2.set_yticks(logmar_ticks)
ax2.set_yticklabels(snellen_labels[::-1])
ax2.tick_params(axis='y', labelcolor='green')

st.pyplot(fig)

# --- Inputs abaixo da curva ---
st.subheader("✍Acuidade Visual em cada ponto da curva")
columns = st.columns(len(zonas_ordenadas))
inputs = {}
x = []
logmars = []

for i, (label, d) in enumerate(zonas_ordenadas):
    with columns[i]:
        val = st.text_input(f"{label}", value="20/20", key=f"in_{label}")
        lm = logmar_from_snellen(val)
        st.caption(f"logMAR {lm:.2f}" if lm is not None else "logMAR inválido")
        inputs[label] = val
        x.append(d)
        logmars.append(lm if lm is not None else 1.0)

# --- Gerar imagem borrada com base nos inputs ---
final = original_img.copy()
for label, d in zonas_ordenadas:
    lm = logmar_from_snellen(inputs[label])
    if lm is None:
        lm = 1.0
    if label in zonas_com_mascara_set:
        zona_mascara = zonas_com_mascara_set[label]
        acuity = acuity_from_logmar(lm)
        blur_radius = get_blur_radius(acuity)
        blurred = original_img.filter(ImageFilter.GaussianBlur(blur_radius))
        final.paste(blurred, mask=masks[zona_mascara])

st.image(final, caption="🖼️ Simulação Visual com Zonas Borradas", use_column_width=True)

# --- Atualizar curva com dados reais e exportar ---
fig2, ax1 = plt.subplots(figsize=(8, 4))
x_array, logmars_array = zip(*sorted(zip(x, logmars)))
ax1.plot(x_array, logmars_array, 'o-', color='blue')
ax1.set_xlabel("Defocus (D)")
ax1.set_ylabel("logMAR", color='blue')
ax1.set_ylim(1.1, -0.3)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)
ax1.set_title("Curva de Acuidade Visual (logMAR e Snellen)")
ax1.set_xticks(x_array)
ax1.set_xticklabels([f"{-val:+.1f}" for val in x_array])

logmar_ticks = np.round(np.arange(-0.3, 1.1, 0.1), 2)
snellen_labels = [snellen_from_logmar(lm) for lm in logmar_ticks]
ax2 = ax1.twinx()
ax2.set_ylabel("Snellen", color='green')
ax2.set_yticks(logmar_ticks)
ax2.set_yticklabels(snellen_labels[::-1])
ax2.tick_params(axis='y', labelcolor='green')

pdf_buffer = io.BytesIO()
with PdfPages(pdf_buffer) as pdf:
    pdf.savefig(fig2, bbox_inches='tight')
    pdf_buffer.seek(0)

st.download_button(
    label="📄 Baixar curva em PDF",
    data=pdf_buffer,
    file_name="curva_defocus.pdf",
    mime="application/pdf"
)
