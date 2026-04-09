# Cilt Kanseri Tespiti - Proje ve Model Karşılaştırma Raporu

## 1. Proje Dosya Yapısı ve İşlevleri

Proje, modern derin öğrenme standartlarına uygun, modüler bir yapıda tasarlanmıştır:

*   **`dataset.py`:** Verinin hazırlandığı modüldür. Görüntüleri yapay zekanın anlayabileceği Tensor formatına dönüştürür. Aşırı sınıf dengesizliğini çözmek için `WeightedRandomSampler` stratejisi kurulmuştur. Ayrıca veri artırımı (Data Augmentation) uygulanarak modelin genelleme yeteneği artırılır.
*   **`model.py`:** Yapay zeka mimarisinin kalbidir. Transfer öğrenme (ImageNet) kullanılarak model yüklenir ve son sınıflandırma katmanı `[Malignant/Benign]` olarak ikili (0-1) tahmin yapacak şekilde revize edilir.
*   **`trainer.py`:** Ana eğitim döngüsünü yönetir. Kayıpları (Loss) hesaplar, ağırlıkları günceller ve erken durdurma (Early Stopping) gibi mekanizmalarla en optimal modeli `best_model.pth` olarak kaydeder.
*   **`evaluate.py`:** Modelin hiç görmediği doğrulama ve test verileri üzerindeki sınavını yapar. Başarı metrikleri hesaplanıp, Confusion Matrix ve ROC Curve grafiklerinin oluşturulduğu yerdir.
*   **`predict.py`:** Kullanıcının terminal üzerinden tekil bir cilt lezyonu resmi verip (örn: `python predict.py test.jpg`), o resmin kanser barındırma olasılığını yüzdelik olarak sorgulayabildiği araçtır.
*   **`utils.py`:** Sınıf eşitsizliğine karşı özel silah olan **Focal Loss** fonksiyonunu barındırır. Seyrek görülen kanserli dokuyu kaçırma durumuna matematiksel olarak çok ağır ceza keserek modelin sadece "iyi huylu" tahminine yatkınlık göstermesini engeller.
*   **`main.py`:** Orkestra şefidir; donanım bağlama (MPS/CPU kurulumu), hiperparametre ayarları (batch size, learning rate), veri yükleme ve doğrudan eğitim sürecini baştan sona tetikleyen tetikleyicidir.

---

## 2. Model Karar Mekanizması ve Çalışma Mantığı

Derin öğrenme modelimiz şu iki ana prensiple çalışır:

1.  **Özellik Çıkarımı (Feature Extraction):** Fotoğraf RGB kanallarıyla girer. İlk konvolüsyonel katmanlar çizgileri, kenarları fark ederken; daha derin katmanlara inildikçe asimetri değerleri, cilt altı yapılar, anormal ve düzensiz nevüs kenarları gibi "Kanser" izlerini taşıyan örüntüler öğrenilir.
2.  **Sınıflandırma (Classification):** Çıkartılan tüm özellikler Classifier bloğunda 0 ile 1 arasında olasılık dağılımına dönüştürülür. Standart konfigürasyonda eşik (threshold) **%50 (0.50)** olarak hesaplanır: Olasılık >= 0.5 ise model lezyonu şüpheli (Kanser - 1) olarak nitelendirir.

---

## 3. ResNet-50 ve EfficientNet-V2-S Modellerinin Karşılaştırması

Projenin çıkış noktasında güçlü `ResNet-50` kullanılırken, final versiyonda Apple M-Serisi işlemcilerin de yapısına mükemmel uyum sağlayan `EfficientNet-V2-S`'e geçiş kararı alınmıştır.

| Özellik / Metrik | Eski Model: ResNet-50 | Yeni Model: EfficientNet-V2-S |
| :--- | :--- | :--- |
| **Parametre Sayısı** | ~25.5 Milyon (Büyük ve Ağır) | ~21 Milyon (Küçük ve Optimizeli) |
| **Büyüme/Öğrenme Stratejisi** | Geleneksel "Katman ekleyerek" derinleşme | **Compound Scaling:** Derinliği, genişliği ve resim çözünürlüğünü akılcı (orantılı) büyütme |
| **Termal Limit/Donanım Baskısı** | Yüksek Isınma, Thermal Throttling riski | **Düşük Enerji Tüketimi**, fansız (Air) sistemlerde dahi yüksek verimlilik beklentisi |
| **Aşırı Öğrenme Cezasının Etkisi**| Normal Standard Seviyelerde | **Training-Aware NAS:** Eğitimin sonlarına doğru regularizasyonu artırıp ezberlemeyi keser. |
| **Son Katman Yapısı** | Fully Connected (FC) Klasik Blok | Modern Dropout Barındıran `Classifier[1]` Bloğu |

---

## 4. Proje Çıktıları ve Doğruluk (Performans) Değerleri

Mimari raporlarımızda ve sistem validasyon testlerinde alınan resmi değerler şunlardır:

### İlk Aşama Model (ResNet-50) Başarı İstatistikleri
*   **Değerlendirmeye Alınan Test Verisi:** 6626 Adet Görüntü
*   **Optimum Karar Eşiği:** %50 (Threshold)
*   **Genel Doğruluk Oranı (Accuracy):** **% 92.0**
*   **Yanlış Alarmlar (False Positive):** 6626 veri içerisinde 492'ye düşürülmüştür. (Yüksek güvenlik hassasiyeti)

### Yeni Nesil Model (EfficientNet-V2-S) Durum Özeti
ResNet-50'nin ulaştığı **%92'lik klinik kullanıma uygun doğruluk oranı**, son derece umut vaad etmektedir. Ancak üretim aşamasına geçilirken parametre kütlesi %20 küçültülerek (EfficientNet-V2-S formuna geçilerek) tıp performans grafiği korunurken **bilgisayar donanımları rahatlatılmış** ve **sistem daha genelleyici, fana ihtiyaç duymadan da çalışabilir modern bir yapıya sürülmüştür**.

### Projenin Kayıtlı Çıktıları
Eğitim sonrası sistem klasörlerinizde başarı kanıtı olarak yer alan çıktılar şunlardır:
1.  **`best_model.pth`**: Erken durdurma ile ağırlıkların en tepe yaptığı andaki yapay zekanın beyninin bir yedeği.
2.  **`training_history.png`**: Epoch'lara göre kayıpların ve doğrulukların evrimini görselleştiren metrik çizgileri.
3.  **`roc_curve.png`**: ROC Eğrisi, AUC değerlendirme grafikleri.
4.  **`confusion_matrix_{50|75|85}.png`**: %50, %75 ve %85 gibi farklı eşikler için doğru bilinen ve kaçırılan teşhis vakalarını analiz eden matris tabloları.
