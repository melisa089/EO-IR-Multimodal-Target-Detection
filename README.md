# Elektro-Optik(EO/IR) Görüntülerde Hedef Tespiti, Sınıflandırma ve Dil Modeli ile Raporlama

Bu depo, Bilecik Şeyh Edebali Üniversitesi Elektrik-Elektronik Mühendisliği Bölümü Bitirme Projesi kapsamında geliştirilen **Çift Akışlı Orta Seviye Füzyon (`MidFusionBlock`)** tabanlı çok-modlu (RGB-IR) hava aracı hedef tespiti ve sınıflandırma altyapısını içermektedir.

## Proje Özeti ve Mimari Yaklaşım
Zorlu aydınlatma koşullarında (gece, düşük ışık, parlama ve sis) tek modaliteli algılayıcıların yaşadığı performans kayıplarını gidermek amacıyla görünür bant (RGB) ile kızılötesi (IR) termal görüntüleri birleştiren uçtan uca bir derin öğrenme mimarisi tasarlanmıştır:

* **Veri Seti ve Ön İşleme (`prepare_dataset.py`):** VisDrone DroneVehicle veri seti üzerinde öncelik tabanlı sınıf dengeleme, `640x512` merkez kırpma, CIELAB renk uzayında CLAHE + Unsharp Masking (RGB) ve kontrast iyileştirme (IR) uygulanarak **3.622 Eğitim, 1.467 Doğrulama ve 1.972 Bağımsız Test** çifti oluşturulmuştur.
* **Çift Akışlı Orta Seviye Füzyon (`MidFusionBlock`):** RGB ve IR görüntüleri ayrı omurgalardan (Dual-Backbone) geçirilerek P3, P4 ve P5 çok-ölçekli öznitelik haritaları elde edilmiş ve **Squeeze-and-Excitation (SE) Kanal Dikkati** mekanizması ile dinamik olarak ağırlıklandırılarak birleştirilmiştir.
* **Kapsamlı Model Karşılaştırması:** Tek modaliteli referans modeller (`YOLO26s-RGB`, `YOLO26s-IR`) ile dört farklı YOLO neslinin (`YOLOv8s`, `YOLO11s`, `YOLO12s`, `YOLO26s`) çift akışlı füzyon mimarileri aynı deneysel protokol altında kıyaslanmıştır.

## Bağımsız Test Kümesi Sonuçları (1.972 RGB-IR Çifti)

| Model | mAP@50 | mAP@50:95 | Precision | Recall | F1-Score | Macro-AUC | Gecikme (ms) | FPS | Parametre (M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `YOLO26s-RGB` | 0.7257 | 0.4293 | 0.7568 | 0.6773 | 0.7149 | 0.8800 | 5.15 | 194.2 | 9.95 |
| `YOLO26s-IR` | 0.8408 | 0.6100 | 0.8061 | 0.7955 | 0.8007 | 0.9437 | 5.02 | 199.2 | 9.95 |
| `YOLOv8s-Fusion` | 0.8486 | 0.6213 | 0.7987 | 0.8081 | 0.8034 | 0.9496 | 8.35 | 119.7 | 17.62 |
| `YOLO11s-Fusion` | 0.8497 | 0.6216 | 0.8088 | 0.8064 | 0.8076 | **0.9520** | 8.61 | 116.2 | 16.48 |
| `YOLO12s-Fusion` | 0.8298 | 0.6082 | 0.7932 | 0.7991 | 0.7962 | 0.9413 | 12.62 | 79.2 | 16.31 |
| **`YOLO26s-Fusion`** | **0.8605** | **0.6295** | **0.8127** | **0.8152** | **0.8139** | 0.9516 | **8.71** | **114.9** | **17.03** |

## Veri Seti Erişimi
Ön işlenmiş ve hizalanmış 7.061 görüntü çiftinden oluşan füzyon veri setine (`Dataset_Preprocessed_Fusion`) aşağıdaki bağlantı üzerinden erişebilirsiniz:
* **Google Drive İndirme Bağlantısı:** [Dataset_Preprocessed_Fusion İndir](https://drive.google.com/file/d/1AQB63x_YrITa-6gYcpk0obLYU-mF_7bA/view?usp=sharing)

## Depo Yapısı
* `prepare_dataset.py`: Ham VisDrone DroneVehicle verisinin dengelenmesi, hizalanması ve RGB-IR ön işleme adımları.
* `YOLOv26s_Baseline/`: Tek modaliteli RGB ve IR referans modellerinin eğitim kodları ve sonuçları.
* `YOLOv8s_Fusion/`, `YOLOv11s_Fusion/`, `YOLOv12s_Fusion/`, `YOLOv26s_Fusion/`: Çift akışlı `MidFusionBlock` eğitim mimarileri ve ağırlık/eğitim kayıtları.
* `RT-DETR+Swin-Tiny/`: Transformer tabanlı çok-modlu tespit denemeleri.
* `Compare/`: Bağımsız test kümesi üzerinde tüm modellerin ablasyon, hız/gecikme, karmaşıklık matrisi ve ROC-AUC (`roc.py`) analiz scriptleri ile üretilen grafik/CSV çıktıları.
