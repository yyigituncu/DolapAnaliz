"""
count_from_video.py

Videodan (kamera rafi/dolabi tararken cekilen) urunleri TAKIP EDEREK
(ByteTrack) sayar. Amac: ayni fiziksel urunun birden fazla karede
gorunmesinden dolayi cift sayilmasini onlemek.

Mantik: her tespite bir "takip kimligi" (track ID) atanir. Ayni urun
ardisik karelerde ayni ID'yi tasir (kamera uzerinden gecip gitse bile).
Toplam sayim = kare basina tespit sayisinin toplami DEGIL, video
boyunca gorulen FARKLI ID sayisidir.

Uc ek guvenlik onlemi var:
1. Ozel tracker ayari (configs/tracker_custom.yaml): bir urun kisa
   sureligine kapansa bile (el, baska urun onune gecmesi gibi) ayni
   ID'yi korumasi icin takip belleği uzatildi.
2. --min-hits filtresi: bir ID, video boyunca en az bu kadar karede
   gorulmediyse sayilmaz. Bu, bir anlik yanlis/gurultulu tespitlerin
   sahte bir ID olusturup sayimi sisirmesini engeller.
3. Cogunluk oylamasi: ayni ID bazen bir karede "limon" bazen "mango"
   gibi farkli siniflar tahmin edilebiliyor (model kararsizsa). Her ID
   video boyunca en cok hangi sinif tahmin edildiyse SADECE o sinifa
   sayilir - tek urun iki farkli sinifa yazilmaz.

Kullanim:
    python scripts/count_from_video.py --model models/v1/best.pt --source video.mp4
    python scripts/count_from_video.py --model models/v1/best.pt --source video.mp4 --min-hits 5
"""

import argparse
from collections import Counter, defaultdict

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Video uzerinde takip tabanli (cift saymayan) sayim")
    parser.add_argument("--model", default="models/v1/best.pt", help="Egitilmis model dosyasi (.pt)")
    parser.add_argument("--source", required=True, help="Video dosyasi")
    parser.add_argument("--conf", type=float, default=0.55, help="Guven esigi (varsayilan: 0.55)")
    parser.add_argument("--tracker", default="configs/tracker_custom.yaml", help="Tracker ayar dosyasi")
    parser.add_argument("--min-hits", type=int, default=6,
                         help="Bir ID'nin sayilmasi icin en az kac karede gorunmesi gerektigi (varsayilan: 6)")
    parser.add_argument("--output", default="data/detections", help="Isaretlenmis videonun kaydedilecegi klasor")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.track(
        source=args.source,
        tracker=args.tracker,
        persist=True,
        conf=args.conf,
        save=True,
        project=args.output,
        name="track_run",
        exist_ok=True,
    )

    class_names = model.names
    # track_id -> Counter(sinif_adi -> kac kez tahmin edildi)
    track_class_votes = defaultdict(Counter)
    track_total_hits = defaultdict(int)

    for frame_result in results:
        boxes = frame_result.boxes
        if boxes.id is None:
            continue  # bu karede henuz onaylanmis/kararli bir takip yok

        for cls_tensor, id_tensor in zip(boxes.cls, boxes.id):
            class_name = class_names[int(cls_tensor)]
            track_id = int(id_tensor)
            track_class_votes[track_id][class_name] += 1
            track_total_hits[track_id] += 1

    counts = defaultdict(int)
    filtered_out = 0
    for track_id, total_hits in track_total_hits.items():
        if total_hits < args.min_hits:
            filtered_out += 1
            continue
        majority_class = track_class_votes[track_id].most_common(1)[0][0]
        counts[majority_class] += 1

    print("\n=== Video Boyunca Benzersiz Urun Sayimi (cift sayim + gurultu filtrelendi) ===")
    if not counts:
        print("Hicbir urun takip edilemedi.")
    else:
        for class_name, count in sorted(counts.items()):
            print(f"  {class_name}: {count}")
    if filtered_out:
        print(f"\n(filtrelenen supheli ID sayisi: {filtered_out})")

    print(f"\nIsaretlenmis video (kutular + ID'ler) kaydedildi -> {args.output}/track_run")


if __name__ == "__main__":
    main()
