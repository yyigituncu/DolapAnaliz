"""
collect_frames.py

Soguk dolap videolarindan veri toplama scripti.
Video dosyalarindan (veya canli kamera/RTSP akisindan) belirli araliklarla
kare (frame) cikarir ve etiketleme (Roboflow/CVAT) icin diske kaydeder.

Kullanim ornekleri:
    # Tek video, 1 saniyede 1 kare
    python scripts/collect_frames.py --input data/raw_videos/video1.mp4 --output data/frames --interval 1.0

    # Klasordeki tum videolar, 0.5 saniyede 1 kare, bulanik kareleri atla
    python scripts/collect_frames.py --input data/raw_videos --output data/frames --interval 0.5 --blur-skip

    # Webcam (index 0), 1 saniyede 1 kare, en fazla 200 kare
    python scripts/collect_frames.py --input 0 --output data/frames --interval 1.0 --max-frames 200
"""

import argparse
import sys
from pathlib import Path

import cv2


def is_blurry(frame, threshold=100.0):
    """Laplacian varyansi ile basit bulaniklik tespiti. Varyans dusukse kare bulaniktir."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold


def extract_frames(source, output_dir, interval_sec, blur_skip, blur_threshold, max_frames):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[HATA] Kaynak acilamadi: {source}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(fps * interval_sec)))

    source_name = f"webcam{source}" if isinstance(source, int) else Path(str(source)).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved_count = 0
    skipped_blur = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if blur_skip and is_blurry(frame, blur_threshold):
                skipped_blur += 1
            else:
                timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                filename = f"{source_name}_{frame_idx:06d}_{timestamp_ms}.jpg"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), frame)
                saved_count += 1

                if max_frames and saved_count >= max_frames:
                    break

        frame_idx += 1

    cap.release()
    print(f"[{source_name}] {saved_count} kare kaydedildi, {skipped_blur} bulanik kare atlandi.")
    return saved_count


def collect_inputs(input_path):
    """input bir video dosyasi, video klasoru ya da webcam index'i (0,1,...) olabilir."""
    if str(input_path).isdigit():
        return [int(input_path)]

    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        video_exts = {".mp4", ".avi", ".mov", ".mkv"}
        videos = sorted(f for f in p.iterdir() if f.suffix.lower() in video_exts)
        return [str(f) for f in videos]

    print(f"[HATA] Girdi bulunamadi: {input_path}")
    return []


def main():
    parser = argparse.ArgumentParser(description="Dolap videolarindan veri toplama araci")
    parser.add_argument("--input", required=True,
                         help="Video dosyasi, video klasoru veya webcam index (0, 1, ...)")
    parser.add_argument("--output", default="data/frames",
                         help="Karelerin kaydedilecegi klasor (varsayilan: data/frames)")
    parser.add_argument("--interval", type=float, default=1.0,
                         help="Kac saniyede bir kare alinsin (varsayilan: 1.0)")
    parser.add_argument("--blur-skip", action="store_true",
                         help="Bulanik kareleri otomatik atla")
    parser.add_argument("--blur-threshold", type=float, default=100.0,
                         help="Bulaniklik esik degeri, dusuk deger = daha siki filtre (varsayilan: 100.0)")
    parser.add_argument("--max-frames", type=int, default=0,
                         help="Her kaynak icin maksimum kare sayisi (0 = sinirsiz)")
    args = parser.parse_args()

    sources = collect_inputs(args.input)
    if not sources:
        sys.exit(1)

    total = 0
    for src in sources:
        total += extract_frames(
            src, args.output, args.interval, args.blur_skip, args.blur_threshold, args.max_frames
        )

    print(f"\nToplam {total} kare kaydedildi -> {args.output}")


if __name__ == "__main__":
    main()
