"""
detect_and_count.py

Egitilmis YOLO modelini kullanarak bir fotografta (veya bir klasordeki
tum fotograflarda) urunleri tespit eder, sinif bazinda sayar ve sonucu
ekrana yazdirir. Tespit kutulari cizilmis gorselleri de diske kaydeder,
boylece gozle dogrulama yapabilirsiniz.

Kullanim:
    python scripts/detect_and_count.py --model models/v1/best.pt --source foto.jpg
    python scripts/detect_and_count.py --model models/v1/best.pt --source data/test_fotograflar --conf 0.4
"""

import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Tespit + sinif bazli sayim")
    parser.add_argument("--model", default="models/v1/best.pt", help="Egitilmis model dosyasi (.pt)")
    parser.add_argument("--source", required=True, help="Fotograf dosyasi veya fotograf klasoru")
    parser.add_argument("--conf", type=float, default=0.4, help="Guven esigi (varsayilan: 0.4)")
    parser.add_argument("--output", default="data/detections", help="Isaretlenmis gorsellerin kaydedilecegi klasor")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        project=args.output,
        name="run",
        exist_ok=True,
    )

    class_names = model.names

    for result in results:
        image_name = Path(result.path).name
        counts = Counter()

        for box in result.boxes:
            class_id = int(box.cls[0])
            counts[class_names[class_id]] += 1

        print(f"\n=== {image_name} ===")
        if not counts:
            print("  Hicbir urun tespit edilmedi.")
        else:
            for class_name, count in sorted(counts.items()):
                print(f"  {class_name}: {count}")

    print(f"\nIsaretlenmis gorseller kaydedildi -> {args.output}/run")


if __name__ == "__main__":
    main()
