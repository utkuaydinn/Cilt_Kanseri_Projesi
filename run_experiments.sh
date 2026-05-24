#!/bin/bash
# Bu script sirasiyla her iki modeli 30 epoch egitmek icin hazirlanmistir.

# Eger sanal ortam varsa aktif et
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "==================================================="
echo "1/2: EFFICIENTNET-V2-S EGITIMI BASLIYOR (30 EPOCH)"
echo "==================================================="
python main.py --model efficientnet --epochs 30

echo "==================================================="
echo "2/2: RESNET-50 EGITIMI BASLIYOR (30 EPOCH)"
echo "==================================================="
python main.py --model resnet --epochs 30

echo "==================================================="
echo "EGITIMLER TAMAMLANDI! KARŞILAŞTIRMA YAPILIYOR..."
echo "==================================================="
python compare_models.py

echo "İşlem bitti. Görsel arayüzü başlatmak için: python app.py"
