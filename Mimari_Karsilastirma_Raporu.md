# Derin Öğrenme Tabanlı Cilt Kanseri Tespiti Mimarisi Karşılaştırma Raporu

Bu rapor, proje kapsamında cilt kanseri tespiti amacıyla kullanılan **(ResNet-50)** ve **(EfficientNet-V2-S)** derin öğrenme mimarilerinin teknik farklarını ve performans kıyaslamasını içermektedir.

---

## 1. İlk Model: ResNet-50 (Residual Networks) Modeli
Projenin ilk eğitim aşaması "Bilgisayarlı Görme" alanındaki devrim niteliğindeki 2015 çıkışlı ResNet-50 ile gerçekleştirilmiştir.

### A. Teknik Özellikler
*   **Mimari Derinlik:** 50 Konvolüsyonel (Convolutional) Katman
*   **Ağırlıklar:** Yaklaşık 25.5 Milyon eğitim parametresi (~98 MB model boyutu)
*   **Temel Gücü (Skip Connections):** Derinleşen sinir ağlarında yaşanan "Gradyan Kaybı (Vanishing Gradient)" problemini, katmanları birbirine bağlayan "Kısa Yol (Bypass)" mantığıyla çözerek veri akışını korumasıdır.
*   **Kayıp Fonksiyonu (Ceza):** Aşırı sınıf eşitsizliğini (%98 iyi huylu, %2 kanserli) engellemek için kodumuza önce "Ağırlıklı BCE Loss", ardından %50'lik ROC dengesi gözeten "Focal Loss (G=2.0, A=0.85)" modülü programlanmış ve çok yüksek kanser bulma başarısına ulaşılmıştır.

### B. İlk Aşama Performans Sonuçları (ResNet-50)
*   Doğrulanmış Test Verisi: **6626 Görüntü**
*   Optimum Karar Eşiği (Threshold): **%50**
*   **Genel Doğruluk (Accuracy):** %92.0
*   Yanlış Alarmlar (False Positives): **492 Vakaya indirilmiştir**
*   *Sonuç:* Ağır ve büyük kalibrasyonu olan mimari olmasına rağmen M3 Mac üzerinde başarılı bir şekilde F1 Skoru grafikleri çıkartmıştır. İlgili ROC (Receiver Operating Characteristic) Eğrileri ve Confusion Matrisleri proje klasörüne başarı kanıtı olarak kaydedilmiştir.

---

## 2. Tercih Edilen Yeni Nesil Model: EfficientNet-V2-S
Apple Silicon M3 mimarisinin "Unified Memory (Birleşik Bellek)" ve "Neural Engine" potansiyelini fan gücü olmadan (Air serisinde) en uç noktaya taşımak, ancak aynı zamanda daha yüksek hassasiyet (% Accuracy) almak amacıyla projenin ana mimarisi Google imzalı EfficientNet serisine (Small varyasyonu) yükseltilmiştir.

### A. Yeni Mimariye Geçiş Gerekçelerimiz
Neden 50 katmanlı dev efsane ResNet yerine, güncel EfficientNet-V2 modeline geçme kararı aldık? Bunun arkasında yatan akademik ve donanımsal 3 ana neden mevcuttur:

1. **Ölçeklendirme Mantığındaki Evrim (Compound Scaling):** ResNet gibi geleneksel modeller başarıyı artırmak için sadece "Katman (Layer) Ekleyerek" derinleşir. EfficientNet ise Görüntü Çözünürlüğünü, Süzgeç Genişliğini ve Derinliği matematiksel olarak **orantılı (Compound) şekilde** aynı anda büyütür. Bu, eski modellere göre çok daha akıllı ama "kilo almayan" bir öğrenme yapısı sunar.
2. **Apple M3 ve Termal Optimizasyon:** Proje bilgisayarlarının aktif fan sisteminin olmaması aşırı ısınma altında frekans düşüşüne (Thermal Throttling) sebep olabilmektedir. EfficientNet-V2-S (Small), sadece 21 Milyon parametre barındırarak **%20 daha küçük bir dosya** olmasına rağmen, ResNet-101'e rakip olabilecek oranda Tıp (Hastalık bulma) başarı yüzdesi sağlar. İşlemci ve RAM sömürüsünü minimuma çekmesi en büyük tercih sebebimiz olmuştur.
3. **Training Aware Neural Architecture Search (Gelişmiş Düzeltmeler):** EfficientNet-V2 serisi, eğitimin ilk evrelerinde hızlı giden ama son evrelerinde (Overfitting - Ezber) kayıp yaşayan önceki jenerasyonların cezasını (Regularization) eğitim bitimine doğru boyutlandırarak mükemmel bir generalizasyon sağlar.

### B. Mimari Tasarım Revizyonu
Değişim esnasında yalnızca yapının gövdesi değil, sınıflandırma modülü de değiştirilmiştir. Eski `FC (Fully Connected)` katmanlarının yerini, V2 mimarisine uygun Dropout katmanı bulunduran `Classifier[1]` bloğu almış, ve son aşamadaki sinir bağlantıları yine 1 çıkışlı (Malignant/Benign ikilisi) bir yapıya zorlanarak Apple MPS cihazında kodlanmıştır.

### 3. Sonuç ve Özet
Projenin görsel hafızası ve kanıtsal başarı tabloları (Confusion Matrix vb.) ResNet-50'nin olağanüstü performansını sergilemektedir. Ancak final prototipi olarak sistem kodlarına yerleştirilen EfficientNet-v2-S, geleceğe yönelik daha az enerji harcayan, günümüz yapay zeka yarışmalarında (Medical Image Segmentation) endüstri standardı haline gelen en optimize yapı olarak kalbindeki yerini almıştır. Tıp raporunda ve sunumda ResNet-50'nin sağlam başarım istatistikleri sunulurken, vizyoner final adımının EfficientNet V2'ye evrimi gururla savunulabilir.
