import os
import pandas as pd
import numpy as np
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking

# Definē direktoriju, kur ir saglabāti žestu CSV faili
data_dir = "gesture_samples"

# Inicializē sarakstus secuences un etiķešu glabāšanai
sequences = []
labels = []

# Definē žestu marķējumu karti (piemēram, "Gesture 1" -> 0, "Gesture 2" -> 1)
gesture_label_map = {"Gesture 1": 0, "Gesture 2": 1}

# Iterē caur visiem CSV failiem direktorijā
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        filepath = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"Kļūda nolasot {filename}: {e}")
            continue
        # Piemērs: faila header: gesture_id, timestamp, x0, x1, ... x20, y0,...,y20, z0,...,z20
        # Izvēlamies izmantot no 3. kolonnas līdz beigām (63 vērtības)
        sequence = df.iloc[:, 2:].values.astype(np.float32)
        sequences.append(sequence)
        
        # Izvelk žesta marķējumu no faila nosaukuma
        # Pieņemsim, ka faila nosaukums ir "gesture_Gesture_1_sample_1.csv"
        parts = filename.split('_')
        # Mēs varam apvienot 2. un 3. elementu, lai iegūtu "Gesture 1"
        if len(parts) >= 3:
            gesture_name = parts[1] + " " + parts[2]
            label = gesture_label_map.get(gesture_name, None)
            if label is None:
                print(f"Nezināms žesta marķējums: {gesture_name} failā {filename}")
                continue
            labels.append(label)
        else:
            print(f"Neatbilstošs faila nosaukums: {filename}")
            continue

print(f"Iegūti {len(sequences)} secuences.")

# Ja secuences garumi atšķiras, padē tos līdz fiksētam garumam, piemēram, 50 kadrus
maxlen = 50
X = pad_sequences(sequences, maxlen=maxlen, dtype='float32', padding='post', truncating='post')
# X forma būs (num_samples, maxlen, 63)

# Konvertē etiķetes uz one-hot formu
num_classes = len(gesture_label_map)
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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Apmāca modeli – pielāgo epochs un batch_size atbilstoši jūsu datiem
model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_val, y_val))

# Saglabā modeli
model.save("gesture_model.h5")
