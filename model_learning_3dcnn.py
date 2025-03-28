import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import matplotlib.pyplot as plt
import seaborn as sns

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
maxlen = 90
padded_sequences = np.zeros((len(sequences), maxlen, 63), dtype=np.float32)
for i, seq in enumerate(sequences):
    length = min(len(seq), maxlen)
    padded_sequences[i, :length, :] = seq[:length]

# Pievieno kanālu dimensiju (1 kanāls)
padded_sequences = np.expand_dims(padded_sequences, axis=-1)

# Konvertē etiķetes uz one-hot formu
labels = np.array(labels)
num_classes = len(set(labels))
y = to_categorical(labels, num_classes=num_classes)

# Sadala datu kopu apmācībai un validācijai
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

# Pievieno dimensijas, lai iegūtu 5D ievadi TensorFlow (batch_size, depth, height, width, channels)
X_train = np.expand_dims(X_train, axis=2)
X_val = np.expand_dims(X_val, axis=2)

print(f"📝 X_train forma: {X_train.shape}")  # Vajadzētu būt (batch_size, 80, 1, 63, 1)

# Veido 3D CNN modeli
model = Sequential([
    Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=(maxlen, 1, 63, 1)),
    MaxPooling3D((2, 1, 2)),
    Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
    MaxPooling3D((2, 1, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Kompilē modeli
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])

# Parāda modeļa kopsavilkumu
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
# Prognozes uz validācijas kopas
y_pred = model.predict(X_val)
y_pred_labels = np.argmax(y_pred, axis=1)
y_val_labels = np.argmax(y_val, axis=1)

# Izveido pārpratuma matricu
conf_matrix = confusion_matrix(y_val_labels, y_pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Pārpratuma matrica')
plt.show()

# Parāda detalizētu klasifikācijas pārskatu
print(classification_report(y_val_labels, y_pred_labels))
# Saglabā modeli
model.save("gesture_3dcnn_model_tf.h5")
print("✅ Modelis saglabāts kā gesture_3dcnn_model_tf.h5")