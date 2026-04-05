# Cilt Kanseri Tespiti - AI Agent Dokümantasyonu

## Proje Özeti

Bu proje, derin öğrenme tabanlı bir cilt kanseri tespit sistemidir. Cilt lezyonlarının görüntülerini analiz ederek iyi huylu (benign) veya kanserli (malignant) olarak sınıflandırır. Proje, Tıbbi Görüntü İşleme (Medical Image Segmentation) alanında kullanılan modern yapay zeka tekniklerini içerir.

### Temel Özellikler
- **İkili Sınıflandırma:** İyi huylu (0) vs Kanserli (1) cilt lezyonları
- **Mimari:** EfficientNet-V2-S (Google) - Transfer öğrenme ile ImageNet ağırlıkları
- **Veri Seti:** Kaggle tarzı yapılandırılmış görüntü veri seti (CSV metadata + JPG görüntüler)
- **Sınıf Dengesizliği Yönetimi:** Focal Loss + WeightedRandomSampler ile ağır sınıf dengesizliği çözümü

---

## Teknoloji Yığını (Technology Stack)

| Bileşen | Kullanım Alanı | Versiyon |
|---------|---------------|----------|
| PyTorch | Derin öğrenme framework'ü | >=2.1.0 |
| torchvision | Önceden eğitilmiş modeller ve görüntü dönüşümleri | >=0.16.0 |
| pandas | CSV veri işleme ve submission dosyaları | >=2.0.0 |
| Pillow (PIL) | Görüntü okuma/yazma | >=10.0.0 |
| scikit-learn | Model seçimi ve değerlendirme metrikleri | >=1.3.0 |
| tqdm | Eğitim ilerleme çubukları | >=4.65.0 |
| matplotlib | Grafik ve görselleştirme | >=3.7.0 |
| numpy | Sayısal işlemler | >=1.24.0 |
| seaborn | Confusion matrix görselleştirme | - |

### Donanım Desteği
- **Apple Silicon (M1/M2/M3):** Metal Performance Shaders (MPS) desteği ile GPU hızlandırma
- **CPU:** MPS mevcut değilse otomatik CPU fallback

---

## Proje Yapısı

```
.
├── main.py              # Ana eğitim betiği
├── dataset.py           # Veri seti ve DataLoader tanımları
├── model.py             # EfficientNet-V2-S model yapılandırması
├── trainer.py           # Eğitim döngüsü ve metrik hesaplama
├── evaluate.py          # Model değerlendirme ve görselleştirme
├── predict.py           # Tek görüntü tahmini için CLI aracı
├── submit.py            # Kaggle submission dosyası üretimi
├── utils.py             # Focal Loss implementasyonu
├── requirements.txt     # Python bağımlılıkları
├── Mimari_Karsilastirma_Raporu.md  # Mimari karşılaştırma raporu (ResNet-50 vs EfficientNet-V2-S)
└── Untitled.ipynb       # Jupyter notebook veri keşfi
```

---

## Veri Seti Yapısı

Proje, aşağıdaki dizin yapısını bekler:

```
dataset/
├── train.csv            # Eğitim metadata: image_name, target sütunları
├── test.csv             # Test metadata: image_name sütunu
├── train/               # Eğitim görüntüleri (.jpg)
│   ├── ISIC_001.jpg
│   ├── ISIC_002.jpg
│   └── ...
└── test/                # Test görüntüleri (.jpg)
    ├── ISIC_003.jpg
    └── ...
```

### CSV Formatı
**train.csv:**
| image_name | target |
|------------|--------|
| ISIC_001 | 0 |
| ISIC_002 | 1 |

**test.csv:**
| image_name |
|------------|
| ISIC_003 |

> **Not:** `dataset/` klasörü `.gitignore`'a eklenmiştir ve Git'e yüklenmez.

---

## Kod Organizasyonu ve Modüller

### 1. dataset.py
- `SkinCancerDataset`: PyTorch Dataset sınıfı
- `get_dataloaders()`: Stratified split, data augmentation ve WeightedRandomSampler ile DataLoader'ları oluşturur
- **Data Augmentation:** RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(20), ColorJitter
- **Normalizasyon:** ImageNet standartları (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### 2. model.py
- `get_model()`: EfficientNet-V2-S modeli yükler
- Son katman (classifier[1]) ikili sınıflandırma için 1 çıkış nöronuna değiştirilir
- `pretrained=True`: ImageNet ağırlıklarını kullanır

### 3. trainer.py
- `train_model()`: Ana eğitim döngüsü
  - Train/Validation fazları
  - ROC-AUC, Recall, Precision, F1 metrikleri
  - Early stopping (best AUC bazlı model kaydı)
- `plot_history()`: Eğitim geçmişi grafikleri

### 4. evaluate.py
- Validation seti üzerinde değerlendirme
- ROC eğrisi (roc_curve.png)
- Çoklu eşik değerleri için confusion matrix (50%, 75%, 85%)
- Terminal çıktısı: TP, FP, FN, TN sayıları

### 5. predict.py
- Tek görüntü tahmini için CLI aracı
- Kullanım: `python predict.py <resim_yolu>`
- Çıktı: Sınıf etiketi ve kanser olasılığı yüzdesi

### 6. submit.py
- Test seti üzerinde tahminler üretir
- `submission.csv` dosyası oluşturur (Kaggle formatı)

### 7. utils.py
- `FocalLoss`: Sınıf dengesizliği için özel kayıp fonksiyonu
  - `alpha=0.85`: Pozitif sınıf ağırlığı
  - `gamma=2.0`: Odaklanma parametresi

---

## Çalıştırma Komutları

### 1. Eğitim
```bash
python main.py
```
**Çıktılar:**
- `best_model.pth`: En iyi model ağırlıkları
- `training_history.png`: Eğitim/validation loss ve metrik grafikleri

### 2. Değerlendirme
```bash
python evaluate.py
```
**Çıktılar:**
- `roc_curve.png`: ROC eğrisi ve AUC alanı
- `confusion_matrix_50.png`: %50 eşik değerli karmaşıklık matrisi
- `confusion_matrix_75.png`: %75 eşik değerli karmaşıklık matrisi
- `confusion_matrix_85.png`: %85 eşik değerli karmaşıklık matrisi

### 3. Tek Görüntü Tahmini
```bash
python predict.py dataset/test/ornek.jpg
```

### 4. Submission Üretimi
```bash
python submit.py
```
**Çıktı:** `submission.csv`

---

## Geliştirme Kuralları ve Stil Rehberi

### Kod Stili
- **Yorumlar:** Türkçe (proje sahibinin tercihi)
- **Değişken İsimleri:** snake_case
- **Sınıf İsimleri:** PascalCase
- **Dosya Kodlaması:** UTF-8

### Önemli Teknik Detaylar

1. **Cihaz Yönetimi:**
   ```python
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   ```
   Her betikte cihaz otomatik olarak ayarlanır.

2. **Sınıf Dengesizliği:**
   - Dataset: ~98% iyi huylu, ~2% kanserli
   - Çözüm: `WeightedRandomSampler` + `FocalLoss(alpha=0.85, gamma=2.0)`

3. **Model Çıkışı:**
   - Ham çıkış: `[batch_size, 1]` (logits)
   - Olasılık: `torch.sigmoid(output)`
   - Tahmin: `prob >= 0.5`

4. **Hiperparametreler (main.py):**
   - `batch_size`: 32
   - `img_size`: 224
   - `num_epochs`: 5
   - `learning_rate`: 1e-4
   - `weight_decay`: 1e-4 (AdamW)

5. **Veri Bölme:**
   - Stratified split (%80 train, %20 validation)
   - `random_state=42` (reprodüksiyon için)

---

## Test Stratejisi

Proje, manuel test ve değerlendirme yöntemleri kullanır:

1. **Eğitim Sırasındaki Validasyon:**
   - Her epoch sonunda validation metrikleri hesaplanır
   - En iyi AUC değerine sahip model kaydedilir

2. **Değerlendirme Betiği (evaluate.py):**
   - ROC-AUC skoru
   - Çoklu eşik değerlerinde confusion matrix
   - Precision, Recall, F1, Accuracy

3. **Görselleştirme:**
   - Eğitim loss eğrisi
   - ROC eğrisi
   - Confusion matrix'ler (farklı eşik değerleri)

> **Not:** Birim testleri (unit tests) mevcut değildir. Testler betik çalıştırma ile manuel olarak yapılır.

---

## Güvenlik Hususları

1. **Model Dosyaları:** `.pth` ve `.pt` dosyaları `.gitignore`'a eklenmiştir
2. **Veri Seti:** `dataset/` klasörü ve `*.csv` dosyaları Git'e yüklenmez
3. **Çıktı Dosyaları:** PNG grafikleri isteğe bağlı olarak Git'e eklenebilir (şu anda dahil)

---

## Mimari Geçmişi

Proje ilk olarak **ResNet-50** ile geliştirilmiş, ardından aşağıdaki nedenlerle **EfficientNet-V2-S**'e geçilmiştir:

- **Daha az parametre:** 21M vs 25.5M
- **Daha iyi termal verimlilik:** Fan'sız sistemlerde (MacBook Air) tercih edilir
- **Compound Scaling:** Orantılı büyütme ile daha verimli öğrenme
- **Training-Aware NAS:** Daha iyi genelleme yeteneği

Detaylı karşılaştırma için `Mimari_Karsilastirma_Raporu.md` dosyasına bakınız.

---

## Ortam Kurulumu

```bash
# Sanal ortam oluşturma (tavsiye edilir)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Bağımlılıkları yükleme
pip install -r requirements.txt

# Apple Silicon için (isteğe bağlı)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## Sık Karşılaşılan Sorunlar

1. **MPS Hafıza Hatası:** `batch_size` değerini düşürün
2. **Dataset Bulunamadı:** `dataset/` klasörünün proje kökünde olduğundan emin olun
3. **Model Dosyası Yok:** Önce `python main.py` ile eğitim yapmalısınız

---

## İletişim ve Katkı

Bu proje, cilt kanseri erken teşhisine katkı sağlamak amacıyla akademik bir çalışma olarak geliştirilmiştir. Kod incelemeleri ve iyileştirme önerileri için ilgili dosyalardaki yorum satırlarını inceleyebilirsiniz.
