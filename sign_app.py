import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Alman Trafik Levhası Tanıma Sistemi",
    page_icon="🚦",
    layout="centered",
)

# GTSRB Sınıf Sözlüğü (43 Kategori)[cite: 6]
classes = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing veh over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Veh > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End no passing veh > 3.5 tons",
}


# Eğitilmiş modeli yükleme
@st.cache_resource
def load_traffic_model():
  model = load_model("traffic_sign_model.h5")
  return model


st.title("🚦 Alman Trafik Levhası Tanıma Sistemi (GTSRB)")
st.write(
    "Bu uygulama, VGG19 tabanlı derin öğrenme modeli kullanarak yüklenen trafik"
    " levhası görselini sınıflandırır."
)

uploaded_file = st.file_uploader(
    "Bir trafik levhası görseli yükleyin (JPG, PNG)...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
  image_pil = Image.open(uploaded_file)
  st.image(image_pil, caption="Yüklenen Tabela Görseli", use_column_width=True)

  if st.button("Levhayı Tanı"):
    with st.spinner("Model tahmin yapıyor..."):
      # Model eğitimi sırasında kullanılan 32x32 boyutuna uyarlama
      image_resized = image_pil.resize((32, 32))
      img_array = image.img_to_array(image_resized)
      img_array = np.expand_dims(img_array, axis=0) / 255.0

      try:
        model = load_traffic_model()
        predictions = model.predict(img_array)[0]
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class] * 100

        sign_name = classes.get(predicted_class, "Bilinmeyen Tabela")

        st.subheader("Tahmin Sonucu:")
        st.success(
            f"🎯 **Tabela:** {sign_name} (Sınıf ID: {predicted_class})"
        )
        st.info(f"📊 **Güven Oranı:** %{confidence:.2f}")

      except Exception as e:
        st.error(
            "Eğitilmiş model dosyası ('traffic_sign_model.h5') bulunamadı! Lütfen"
            " önce modeli eğittiğinizden emin olun."
        )