import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import re
import matplotlib.pyplot as plt

# Definē direktoriju, kur ir saglabāti žestu CSV faili
data_dir = "gesture_samples"

# Inicializē sarakstus secuences un etiķešu glabāšanai
sequences = []
labels = []

# Regulārais izteiciens failu nosaukumu parsēšanai (gesture_<label>_<timestamp>.csv)
pattern = re.compile(r"gesture_(\d)_\d{8}-\d{6}\.csv")

# Iterē caur visiem CSV failiem direktorijā
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        match = pattern.match(filename)
        if not match:
            print(f"❌ Neizdevās parsēt faila nosaukumu: {filename}")
            continue

        gesture_label = int(match.group(1))
        filepath = os.path.join(data_dir, filename)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Kļūda nolasot {filename}: {e}")
            continue

        # Izvēlamies koordinātas no 3. kolonnas (x0, y0, z0 ...) līdz beigām
        sequence = df.iloc[:, 2:].values.astype(np.float32)
        sequences.append(sequence)
        labels.append(gesture_label)

print(f"✅ Iegūti {len(sequences)} secuences.")

# Ja secuences garumi atšķiras, padē tos līdz fiksētam garumam
maxlen = 30
padded_sequences = np.zeros((len(sequences), maxlen, 63), dtype=np.float32)
for i, seq in enumerate(sequences):
    length = min(len(seq), maxlen)
    padded_sequences[i, :length, :] = seq[:length]

# Konvertē etiķetes uz tensoru
labels = np.array(labels)

# Sadala datu kopu apmācībai un validācijai
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Pārvērš PyTorch tenzoros
X_train = torch.tensor(X_train).unsqueeze(1)  # Pievieno kanālu dimensiju
X_val = torch.tensor(X_val).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)



# Izveido PyTorch Dataset
class GestureDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_dataset = GestureDataset(X_train, y_train)
val_dataset = GestureDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# Definē 3D CNN modeli
class Gesture3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Gesture3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, (3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(32, 64, (3, 3, 3), padding=1)
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.fc1 = nn.Linear(64 * (maxlen // 2) * (63 // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

num_classes = len(set(labels))
model = Gesture3DCNN(num_classes)

# Definē zaudējuma funkciju un optimizatoru
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

# Apmācības cilpa
epochs = 50
train_losses, val_losses, train_acc, val_acc = [], [], [], []

for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    train_losses.append(total_loss / len(train_loader))
    train_acc.append(correct / total)

    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    val_losses.append(total_loss / len(val_loader))
    val_acc.append(correct / total)

    print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}")

# Attēlo apmācības un validācijas zaudējumu un precizitāti
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Apmācības un validācijas zaudējums')

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Apmācības un validācijas precizitāte')

plt.tight_layout()
plt.show()

# Saglabā modeli
torch.save(model.state_dict(), "gesture_3dcnn_model.pth")
print("✅ Modelis saglabāts kā gesture_3dcnn_model.pth")
