"""
link_dataset_images.py

Label Studio'dan YOLO formatinda export edilen etiket (labels/) klasorundeki
dosya isimlerini kullanarak, data/frames/ altindaki orijinal fotograflari
bulup dataset klasorundeki images/ klasorune kopyalar.

Label Studio export'u bazen sadece etiket (.txt) dosyalarini verir,
gorsel dosyalarini vermez (images/ klasoru bos kalir). Bu script o
eksigi, elimizde zaten olan orijinal kareleri kullanarak kapatir.

Label Studio, yuklenen her dosyanin basina 8 karakterlik bir kod ekliyor
(orn: 00a04adc-IMG_6409_000468_7805.txt). Bu script o kodu atip orijinal
dosya adini (IMG_6409_000468_7805) data/frames altinda arar.

Kullanim:
    python scripts/link_dataset_images.py --labels data/dataset_v1/labels --frames data/frames --output data/dataset_v1/images
"""

import argparse
import shutil
from pathlib import Path


def build_frame_index(frames_dir):
    """data/frames altindaki tum jpg dosyalarini {dosya_adi: tam_yol} sozlugune cikarir."""
    index = {}
    frames_path = Path(frames_dir)
    for jpg_file in frames_path.rglob("*.jpg"):
        index[jpg_file.stem] = jpg_file
    return index


def main():
    parser = argparse.ArgumentParser(description="Label Studio etiketlerine karsilik gelen gorselleri bulup kopyalar")
    parser.add_argument("--labels", required=True, help="YOLO etiket (.txt) dosyalarinin oldugu klasor")
    parser.add_argument("--frames", required=True, help="Orijinal karelerin oldugu ana klasor (data/frames)")
    parser.add_argument("--output", required=True, help="Gorsellerin kopyalanacagi klasor (images/)")
    args = parser.parse_args()

    labels_dir = Path(args.labels)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Orijinal kareler taraniyor...")
    frame_index = build_frame_index(args.frames)
    print(f"{len(frame_index)} orijinal kare bulundu.")

    label_files = sorted(labels_dir.glob("*.txt"))
    matched = 0
    unmatched = []

    for label_file in label_files:
        label_stem = label_file.stem  # ornek: 00a04adc-IMG_6409_000468_7805

        if "-" in label_stem:
            original_stem = label_stem.split("-", 1)[1]
        else:
            original_stem = label_stem

        if original_stem in frame_index:
            src = frame_index[original_stem]
            dst = output_dir / f"{label_stem}.jpg"
            shutil.copy2(src, dst)
            matched += 1
        else:
            unmatched.append(label_stem)

    print(f"\n{matched}/{len(label_files)} gorsel eslesip kopyalandi -> {output_dir}")
    if unmatched:
        print(f"UYARI: {len(unmatched)} etiket icin gorsel bulunamadi:")
        for u in unmatched[:10]:
            print(f"  - {u}")
        if len(unmatched) > 10:
            print(f"  ... ve {len(unmatched) - 10} tane daha")


if __name__ == "__main__":
    main()
