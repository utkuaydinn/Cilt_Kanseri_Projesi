import json
import sys

with open('runs/efficientnet_B32_20260523_221037/results.json', 'r') as f:
    data = json.load(f)['results']['history']

print("| Epoch | Val Loss | Val AUC | Val Precision | Val Recall |")
print("| :---: | :---: | :---: | :---: | :---: |")

for i in range(len(data['val_loss'])):
    loss = data['val_loss'][i]
    auc = data['val_auc'][i]
    prec = data['val_precision'][i]
    rec = data['val_recall'][i]
    
    print(f"| {i+1} | {loss:.4f} | %{auc*100:.2f} | %{prec*100:.2f} | %{rec*100:.2f} |")
