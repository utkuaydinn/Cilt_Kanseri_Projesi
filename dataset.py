import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

class SkinCancerDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image_name'] + '.jpg'
        img_path = os.path.join(self.image_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx]['target']
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def get_dataloaders(csv_path, img_dir, batch_size=32, img_size=224, test_size=0.2):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['target', 'image_name'])
    
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['target'], random_state=42)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SkinCancerDataset(train_df, img_dir, transform=train_transform)
    val_dataset = SkinCancerDataset(val_df, img_dir, transform=val_transform)
    
    class_counts = train_df['target'].value_counts().sort_index()
    count_0 = class_counts.get(0, 1)
    count_1 = class_counts.get(1, 1)
    
    weight_0 = 1.0 / count_0
    weight_1 = 1.0 / count_1
    
    sample_weights = [weight_0 if label == 0 else weight_1 for label in train_df['target']]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, train_df, val_df
