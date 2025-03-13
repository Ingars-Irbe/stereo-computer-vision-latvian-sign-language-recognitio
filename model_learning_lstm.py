import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
import re
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

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
maxlen = 80
X = pad_sequences(sequences, maxlen=maxlen, dtype='float32', padding='post', truncating='post')

# Konvertē etiķetes uz one-hot formu
num_classes = len(set(labels))
y = to_categorical(labels, num_classes=num_classes)

# Sadala datu kopu apmācībai un validācijai
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Veido LSTM modeli
model = Sequential([
    Masking(mask_value=0., input_shape=(maxlen, 63)),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])



learning_rate = 0.00005 # Piemēram, lēnāks mācīšanās ātrums
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Apmāca modeli
history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_val, y_val))

# Attēlo apmācības un validācijas zaudējumu un precizitāti
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Apmācības un validācijas zaudējums')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Apmācības un validācijas precizitāte')

plt.tight_layout()
plt.show()

# Saglabā modeli
model.save("gesture_model.h5")
print("✅ Modelis saglabāts kā gesture_model.h5")
