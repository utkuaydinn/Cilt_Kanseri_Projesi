import json

with open('runs/resnet_B32_20260524_064757/results.json', 'r') as f:
    data = json.load(f)['results']['history']

print("\n## 5. Epoch-by-Epoch Performans Gelişimi (ResNet-50)\n")
print("| Epoch | Val Loss (Kayıp) | Val AUC (Genel Başarı) | Val Precision (Kesinlik) | Val Recall (Hassasiyet) |")
print("| :---: | :---: | :---: | :---: | :---: |")

for i in range(len(data['val_loss'])):
    loss = data['val_loss'][i]
    auc = data['val_auc'][i]
    prec = data['val_precision'][i]
    rec = data['val_recall'][i]
    opt_str = " | *(Optimum Model)*" if i == 4 else " |"  # Epoch 5 is index 4
    
    print(f"| {i+1} | {loss:.4f} | %{auc*100:.2f} | %{prec*100:.2f} | %{rec*100:.2f}{opt_str}")
