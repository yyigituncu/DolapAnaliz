"""
download_external_dataset.py

Roboflow Universe'deki hazir (public) veri setlerini indirip
data/external/ altina kaydetmek icin yardimci script.

Kullanim:
    1) Once Roboflow hesabinizdan API key alin (asagidaki adimlarda anlatiliyor)
    2) Ortam degiskeni olarak ayarlayin:
         Windows PowerShell:  $env:ROBOFLOW_API_KEY="xxxxx"
         Windows cmd:         set ROBOFLOW_API_KEY=xxxxx
         Mac/Linux:            export ROBOFLOW_API_KEY=xxxxx
    3) Scripti calistirin:
         python scripts/download_external_dataset.py --workspace new-workspace-ihwlk --project coca-cola-fqmpn --version 2 --format yolov8 --output data/external/coca-cola-fqmpn
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Roboflow Universe veri seti indirici")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace adi (URL'deki ilk kisim)")
    parser.add_argument("--project", required=True, help="Roboflow proje adi (URL'deki ikinci kisim)")
    parser.add_argument("--version", type=int, default=1, help="Veri seti versiyonu (varsayilan: 1)")
    parser.add_argument("--format", default="yolov8", help="Export formati (varsayilan: yolov8)")
    parser.add_argument("--output", required=True, help="Kaydedilecek klasor")
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("[HATA] ROBOFLOW_API_KEY ortam degiskeni bulunamadi.")
        print("Once Roboflow hesabinizdan (Settings > API Keys) key alip ortam degiskeni olarak ayarlayin.")
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[HATA] 'roboflow' paketi kurulu degil. Once calistirin: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download(args.format, location=args.output)

    print(f"\nIndirme tamamlandi -> {dataset.location}")


if __name__ == "__main__":
    main()
