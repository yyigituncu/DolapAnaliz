"""
server.py

Mobil-uyumlu web uygulamasi: telefon tarayicisindan hem FOTOGRAF hem
VIDEO analizi yapabilmek icin. Fotografta anlik tespit+sayim, videoda
ByteTrack ile takip tabanli (cift saymayan) sayim calisir.

Calistirmak icin:
    python mobile_app/server.py

Sonra telefonunuzdan (BILGISAYARINIZLA AYNI WIFI AGINDA):
    http://<bilgisayarin-yerel-IP-adresi>:5000
(IP'yi ogrenmek icin PowerShell'de "ipconfig" -> "IPv4 Address")
"""

import base64
import io
import subprocess
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import imageio_ffmpeg
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageOps
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "v1" / "best.pt"
TRACKER_PATH = BASE_DIR / "configs" / "tracker_custom.yaml"
UPLOAD_DIR = BASE_DIR / "data" / "detections" / "web_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250MB (video icin)

model = YOLO(str(MODEL_PATH))


def convert_to_web_mp4(input_path, output_path):
    """Ultralytics'in urettigi videoyu (avi/mp4v gibi tarayicinin oynatamadigi
    kodekler) telefon tarayicilarinin oynatabildigi standart H.264 MP4'e cevirir."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg_exe, "-y", "-i", str(input_path),
            "-vcodec", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )


def analyze_photo(file_storage):
    image = Image.open(file_storage.stream)
    image = ImageOps.exif_transpose(image)  # telefonun EXIF donme bilgisini piksellere uygula
    image = image.convert("RGB")
    results = model.predict(source=image, conf=0.5, verbose=False)
    result = results[0]

    counter = Counter()
    for box in result.boxes:
        class_name = model.names[int(box.cls[0])]
        counter[class_name] += 1

    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
    annotated_pil = Image.fromarray(annotated)
    buffer = io.BytesIO()
    annotated_pil.save(buffer, format="JPEG")
    image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return dict(sorted(counter.items())), image_data


def analyze_video(file_storage, min_hits=6):
    run_id = uuid.uuid4().hex[:8]
    tmp_path = UPLOAD_DIR / f"upload_{run_id}{Path(file_storage.filename).suffix or '.mp4'}"
    file_storage.save(tmp_path)

    run_name = f"run_{run_id}"
    results = model.track(
        source=str(tmp_path),
        tracker=str(TRACKER_PATH),
        persist=True,
        conf=0.55,
        save=True,
        project=str(UPLOAD_DIR),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    class_names = model.names
    # Her takip ID'si icin, hangi sinifin kac kere tahmin edildigini say
    # (ayni ID bir karede "limon" bir karede "mango" tahmin edilebiliyor -
    # cogunluk oylamasiyla tek bir sinifa karar verecegiz, iki sinifa
    # birden yazilmasini onleyecegiz)
    track_class_votes = defaultdict(Counter)  # track_id -> Counter(sinif_adi -> kac kez)
    track_total_hits = defaultdict(int)  # track_id -> toplam kac karede gorundu

    for frame_result in results:
        boxes = frame_result.boxes
        if boxes.id is None:
            continue
        for cls_tensor, id_tensor in zip(boxes.cls, boxes.id):
            class_name = class_names[int(cls_tensor)]
            track_id = int(id_tensor)
            track_class_votes[track_id][class_name] += 1
            track_total_hits[track_id] += 1

    counts = defaultdict(int)
    for track_id, total_hits in track_total_hits.items():
        if total_hits < min_hits:
            continue  # yeterince surekli gorunmedi, supheli/gurultu - atla
        majority_class = track_class_votes[track_id].most_common(1)[0][0]
        counts[majority_class] += 1
    counts = dict(sorted(counts.items()))

    tmp_path.unlink(missing_ok=True)

    # Isaretlenmis cikti videosunu bul ve tarayici-uyumlu formata cevir
    run_dir = UPLOAD_DIR / run_name
    video_url = None
    if run_dir.exists():
        found = list(run_dir.glob("*.avi")) + list(run_dir.glob("*.mp4"))
        if found:
            raw_output = found[0]
            web_output = run_dir / "web_output.mp4"
            try:
                convert_to_web_mp4(raw_output, web_output)
                video_url = f"/media/{run_name}/{web_output.name}"
            except Exception:
                # Donusum basarisiz olursa, en azindan orijinal dosyayi sun
                video_url = f"/media/{run_name}/{raw_output.name}"

    return counts, video_url


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze/photo", methods=["POST"])
def analyze_photo_route():
    uploaded = request.files.get("photo")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Fotoğraf bulunamadı"}), 400

    counts, image_data = analyze_photo(uploaded)
    return jsonify({"counts": counts, "image": image_data})


@app.route("/analyze/video", methods=["POST"])
def analyze_video_route():
    uploaded = request.files.get("video")
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "Video bulunamadı"}), 400

    counts, video_url = analyze_video(uploaded)
    return jsonify({"counts": counts, "video_url": video_url})


@app.route("/media/<run_name>/<path:filename>")
def media(run_name, filename):
    return send_from_directory(UPLOAD_DIR / run_name, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
