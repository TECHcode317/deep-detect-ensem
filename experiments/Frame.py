
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import io



device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()



DATASET = "/kaggle/input/1000-deepfake-videos/1000_videos"



class CrossDomainTransform:
    """Simulate real-world variability: compression, blur, color, noise, frame artifacts"""
    def __init__(self):
        self.base = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
            ),
        ])
    
    def __call__(self, img):
        img = self.base(img)
        img = self.simulate_compression(img)
        img = self.simulate_blur(img)
        img = self.add_gaussian_noise(img)
        return transforms.ToTensor()(img)
    
    def simulate_compression(self, img):
        """Random JPEG compression simulating different platforms/codecs"""
        buffer = io.BytesIO()
        quality = random.randint(30, 90)
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

    def simulate_blur(self, img):
        """Random blur to simulate low-quality platforms"""
        if random.random() < 0.3:
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius))
        return img

    def add_gaussian_noise(self, img):
        """Add adversarial-like Gaussian noise"""
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 5, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)



class DeepfakeDataset(Dataset):
    def __init__(self, root, phase="train"):
        self.samples = []
        self.phase = phase
        for f in os.listdir(root+"/Fake"):
            if f.endswith(".png"):
                self.samples.append((root+"/Fake/"+f, 0))
        for f in os.listdir(root+"/Real"):
            if f.endswith(".png"):
                self.samples.append((root+"/Real/"+f, 1))
        if phase=="train":
            self.transform = CrossDomainTransform()
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        if self.phase=="train":
            img = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(img)
        return img, l


dataset = DeepfakeDataset(DATASET, phase="train")
train_size = int(0.8*len(dataset))
val_size = len(dataset)-train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

print("Train samples:", len(train_ds))
print("Validation samples:", len(val_ds))



def build_model(name):
    if name=="mobilenet":
        model = models.mobilenet_v3_small(weights="DEFAULT")
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features,2)
    elif name=="shufflenet":
        model = models.shufflenet_v2_x1_0(weights="DEFAULT")
        model.fc = torch.nn.Linear(model.fc.in_features,2)
    elif name=="efficientnet":
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features,2)
    return model.to(device)


def train_model(name, epochs=5, lr=3e-4):
    model = build_model(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total, correct = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            pred = out.argmax(1)
            correct += (pred==labels).sum().item()
            total += labels.size(0)
        train_acc = correct/total

        # validation
        model.eval()
        vtotal, vcorrect = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                pred = out.argmax(1)
                vcorrect += (pred==labels).sum().item()
                vtotal += labels.size(0)
        val_acc = vcorrect/vtotal
        print(f"{name} Epoch {epoch+1} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

   
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"/kaggle/working/{name}.pth")
    return best_acc



print("\nTraining MobileNetV3...")
train_model("mobilenet", epochs=5)

print("\nTraining ShuffleNetV2...")
train_model("shufflenet", epochs=5)

print("\nTraining EfficientNet-B0...")
train_model("efficientnet", epochs=5)



mobilenet = build_model("mobilenet")
mobilenet.load_state_dict(torch.load("/kaggle/working/mobilenet.pth"))
mobilenet.eval()

shufflenet = build_model("shufflenet")
shufflenet.load_state_dict(torch.load("/kaggle/working/shufflenet.pth"))
shufflenet.eval()

efficientnet = build_model("efficientnet")
efficientnet.load_state_dict(torch.load("/kaggle/working/efficientnet.pth"))
efficientnet.eval()



y_true, y_pred = [], []

for imgs, labels in val_loader:
    imgs = imgs.to(device)
    with torch.no_grad():
        p1 = mobilenet(imgs)
        p2 = shufflenet(imgs)
        p3 = efficientnet(imgs)
        # soft voting
        pred = ((torch.softmax(p1,1)+torch.softmax(p2,1)+torch.softmax(p3,1))/3).argmax(1).cpu()
    y_true.extend(labels.numpy())
    y_pred.extend(pred.numpy())



print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Fake","Real"]))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(j,i,cm[i,j], ha="center", color="white", fontsize=16)
plt.xticks([0,1], ["Fake","Real"])
plt.yticks([0,1], ["Fake","Real"])
plt.show()
