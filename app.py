"""
app.py

Basit, gecici bir test arayuzu: bir fotograf yukleyin, model urunleri
tespit edip sinif bazinda saysin. Tarayicida acilir, kod yazmadan
kullanilir.

Calistirmak icin:
    python app.py

Sonra tarayicida otomatik acilan (veya terminalde yazan) localhost
adresine gidin.
"""

from collections import Counter

import gradio as gr
from ultralytics import YOLO

MODEL_PATH = "models/v1/best.pt"
model = YOLO(MODEL_PATH)


def analiz_et(image, guven_esigi):
    if image is None:
        return None, "Once bir fotograf yukleyin."

    results = model.predict(source=image, conf=guven_esigi, save=False, verbose=False)
    result = results[0]

    counts = Counter()
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        counts[class_name] += 1

    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB

    if counts:
        summary = "\n".join(f"{name}: {count}" for name, count in sorted(counts.items()))
    else:
        summary = "Hicbir urun tespit edilmedi. Guven esigini dusurmeyi deneyin."

    return annotated, summary


demo = gr.Interface(
    fn=analiz_et,
    inputs=[
        gr.Image(type="filepath", label="Dolap Fotoğrafı Yükle"),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.4, step=0.05, label="Güven Eşiği"),
    ],
    outputs=[
        gr.Image(label="Tespit Sonucu"),
        gr.Textbox(label="Ürün Sayımı", lines=6),
    ],
    title="Dolap Stok Analiz — Pilot Test Arayüzü",
    description="Bir fotoğraf yükleyin, model ürünleri tespit edip sınıf bazında saysın. (Geçici test arayüzü — ileride geliştirilecek.)",
)

if __name__ == "__main__":
    demo.launch()
