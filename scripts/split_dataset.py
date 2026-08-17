"""
split_dataset.py

Etiketli veri setini (images/ + labels/) egitim (train) ve dogrulama (val)
olarak ikiye ayirir. YOLO'nun bekledigi klasor yapisini olusturur:

    dataset_v1/
    |-- images/
    |   |-- train/
    |   `-- val/
    `-- labels/
        |-- train/
        `-- val/

Neden train/val ayrimi gerekli: Model, train setiyle egitilir. Val seti ise
egitim sirasinda hic gormedigi gorsellerle "gercekten ogrendi mi, yoksa
ezberledi mi" diye test etmek icin kullanilir.

Kullanim:
    python scripts/split_dataset.py --dataset data/dataset_v1 --val-ratio 0.15
"""

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Veri setini train/val olarak ayirir")
    parser.add_argument("--dataset", required=True, help="images/ ve labels/ klasorlerini iceren ana klasor")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Dogrulama setine ayrilacak oran (varsayilan: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Rastgelelik tohumu (tekrarlanabilirlik icin)")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        print(f"[HATA] {images_dir} icinde .jpg dosyasi bulunamadi. Zaten bolunmus olabilir.")
        return

    random.seed(args.seed)
    random.shuffle(image_files)

    val_count = max(1, int(len(image_files) * args.val_ratio))
    val_files = set(image_files[:val_count])
    train_files = image_files[val_count:]

    for split_name, files in [("train", train_files), ("val", val_files)]:
        (images_dir / split_name).mkdir(parents=True, exist_ok=True)
        (labels_dir / split_name).mkdir(parents=True, exist_ok=True)

        for img_path in files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                print(f"[UYARI] Etiket bulunamadi, atlaniyor: {img_path.name}")
                continue

            shutil.move(str(img_path), str(images_dir / split_name / img_path.name))
            shutil.move(str(label_path), str(labels_dir / split_name / label_path.name))

    print(f"Train: {len(train_files)} gorsel")
    print(f"Val: {len(val_files)} gorsel")
    print("\nBitti. images/ ve labels/ altinda train/ ve val/ alt klasorleri olustu.")


if __name__ == "__main__":
    main()
