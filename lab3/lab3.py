#%%
import random
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
#%%
BASE = Path("data")
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
MY_TEST_DIR = Path("data/test")
TEST_DIR = Path("data/data_from_test")
URLS = ["https://storage.googleapis.com/download.tensorflow.org/data/rps.zip",
        "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/rps.zip"]

CLASSES = ["rock", "paper", "scissors"]
PER_CLASS = 100
VAL_SHARE = 0.2
IMG_SIZE = 128
BATCH = 32
EPOCHS = 10
LR = 1e-3
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = random.randint(0, 100)
torch.manual_seed(SEED)
print("устройство:", DEVICE)
#%%
def download_dataset():
    if (BASE / "rps").exists():
        return
    BASE.mkdir(exist_ok=True)
    zip_path = BASE / "rps.zip"
    for url in URLS:
        try:
            print(f"скачивание {url}")
            urllib.request.urlretrieve(url, zip_path)
            break
        except Exception as e:
            print(f"ошибка: {e}")
    else:
        raise RuntimeError("не удалось скачать датасет")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(BASE)
    zip_path.unlink()


def pick_files():
    train, val = [], []
    n_val = int(PER_CLASS * VAL_SHARE)
    for label, name in enumerate(CLASSES):
        files = sorted((BASE / "rps" / name).glob("*.png"))
        files = random.sample(files, PER_CLASS)
        val += [(f, label) for f in files[:n_val]]
        train += [(f, label) for f in files[n_val:]]
    return train, val


def load_test_files():
    files = []
    test_folders = {
        "rock": "rock",
        "paper": "paper",
        "scissors": "scissors"
    }

    for label, name in enumerate(CLASSES):
        folder = TEST_DIR / test_folders[name]

        if not folder.exists():
            raise FileNotFoundError(f"Нет папки {folder}")

        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                files.append((f, label))

    return files

def load_my_test_files():
    files = []
    test_folders = {
        "rock": "rock",
        "paper": "paper",
        "scissors": "scissors"
    }

    for label, name in enumerate(CLASSES):
        folder = MY_TEST_DIR / test_folders[name]

        if not folder.exists():
            raise FileNotFoundError(f"Нет папки {folder}")

        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                files.append((f, label))

    return files

def show_grid(items, out_name, cols=3):
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (path, title, color) in zip(axes, items):
        ax.imshow(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
        ax.set_title(title, color=color)
    plt.tight_layout()
    plt.savefig(RESULTS / out_name)
    plt.show()


download_dataset()
train_files, val_files = pick_files()
print(f"train: {len(train_files)}, val: {len(val_files)}")
#%%
samples = []
for label, name in enumerate(CLASSES):
    own = [f for f, l in train_files if l == label][:3]
    samples += [(f, name, "black") for f in own]
show_grid(samples, "samples.png")
#%%
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=255),
    transforms.ToTensor(),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


class RPSDataset(Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path, label = self.files[i]
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return self.transform(img), label


train_loader = DataLoader(RPSDataset(train_files, train_tf), batch_size=BATCH, shuffle=True)
val_loader = DataLoader(RPSDataset(val_files, eval_tf), batch_size=BATCH)
#%%
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * (IMG_SIZE // 8) ** 2, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = SimpleCNN().to(DEVICE)
#print(model)
print("параметров:", sum(p.numel() for p in model.parameters()))
#%%
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def run_epoch(loader, train):
    model.train(train)
    total_loss = correct = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n
#%%
hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
for epoch in range(1, EPOCHS + 1):
    tl, ta = run_epoch(train_loader, train=True)
    vl, va = run_epoch(val_loader, train=False)
    for k, v in zip(hist, (tl, ta, vl, va)):
        hist[k].append(v)
    print(f"эпоха {epoch:2d}: train loss {tl:.4f}, acc {ta:.3f} | val loss {vl:.4f}, acc {va:.3f}")
#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ep = range(1, EPOCHS + 1)
ax1.plot(ep, hist["train_loss"], label="train")
ax1.plot(ep, hist["val_loss"], label="val")
ax1.set_title("Loss")
ax2.plot(ep, hist["train_acc"], label="train")
ax2.plot(ep, hist["val_acc"], label="val")
ax2.set_title("Accuracy")
for ax in (ax1, ax2):
    ax.set_xlabel("эпоха")
    ax.legend()
plt.tight_layout()
plt.savefig(RESULTS / "curves.png")
plt.show()

gap = hist["train_acc"][-1] - hist["val_acc"][-1]
print(f"train accuracy {hist['train_acc'][-1]:.3f}, val accuracy {hist['val_acc'][-1]:.3f}")
print(f"val loss: минимум {min(hist['val_loss']):.4f} на эпохе {hist['val_loss'].index(min(hist['val_loss'])) + 1}")
#%%
#тест на тесте
test_files = load_test_files()
print(f"total test: {len(test_files)}")

test_loader = DataLoader(RPSDataset(test_files, eval_tf), batch_size=BATCH)

model.eval()
preds = []
with torch.no_grad():
    for x, y in test_loader:
        preds += model(x.to(DEVICE)).argmax(1).tolist()
labels = [l for _, l in test_files]

cm = np.zeros((3, 3), dtype=int)
for t, p in zip(labels, preds):
    cm[t][p] += 1
test_acc = np.trace(cm) / cm.sum()
print(f"test accuracy: {test_acc:.3f} ({np.trace(cm)} из {cm.sum()})")
for i, name in enumerate(CLASSES):
    print(f"  {name}: {cm[i, i]} из {cm[i].sum()}")

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3))
ax.set_xticklabels(CLASSES)
ax.set_yticks(range(3))
ax.set_yticklabels(CLASSES)
ax.set_xlabel("предсказано")
ax.set_ylabel("правильно")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig(RESULTS / "confusion_matrix.png")
plt.show()
#%%
idx = random.sample(range(len(test_files)), min(12, len(test_files)))
items = []
for i in idx:
    ok = preds[i] == labels[i]
    title = f"верно: {CLASSES[labels[i]]}\nсеть: {CLASSES[preds[i]]}"
    items.append((test_files[i][0], title, "green" if ok else "red"))
show_grid(items, "test_examples.png", cols=4)
#%%
print(f"train accuarcy: {hist['train_acc'][-1]:.3f}")
print(f"val accuarcy:   {hist['val_acc'][-1]:.3f}")
print(f"test accuarcy:  {test_acc:.3f} ({np.trace(cm)} из {cm.sum()})")
#%%
#тест на моих скринах
test_files = load_my_test_files()
print(f"total test: {len(test_files)}")

test_loader = DataLoader(RPSDataset(test_files, eval_tf), batch_size=BATCH)

model.eval()
preds = []
with torch.no_grad():
    for x, y in test_loader:
        preds += model(x.to(DEVICE)).argmax(1).tolist()
labels = [l for _, l in test_files]

cm = np.zeros((3, 3), dtype=int)
for t, p in zip(labels, preds):
    cm[t][p] += 1
test_acc = np.trace(cm) / cm.sum()
print(f"test accuracy: {test_acc:.3f} ({np.trace(cm)} из {cm.sum()})")
for i, name in enumerate(CLASSES):
    print(f"  {name}: {cm[i, i]} из {cm[i].sum()}")

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3))
ax.set_xticklabels(CLASSES)
ax.set_yticks(range(3))
ax.set_yticklabels(CLASSES)
ax.set_xlabel("предсказано")
ax.set_ylabel("правильно")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig(RESULTS / "my_confusion_matrix.png")
plt.show()
#%%
idx = random.sample(range(len(test_files)), min(12, len(test_files)))
items = []
for i in idx:
    ok = preds[i] == labels[i]
    title = f"верно: {CLASSES[labels[i]]}\nсеть: {CLASSES[preds[i]]}"
    items.append((test_files[i][0], title, "green" if ok else "red"))
show_grid(items, "my_test_examples.png", cols=4)
#%%
print(f"train accuracy: {hist['train_acc'][-1]:.3f}")
print(f"val accuracy:   {hist['val_acc'][-1]:.3f}")
print(f"test accuracy:  {test_acc:.3f} ({np.trace(cm)} из {cm.sum()})")