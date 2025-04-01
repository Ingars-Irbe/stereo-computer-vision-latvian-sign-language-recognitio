import os
import pandas as pd
import numpy as np
import cv2

# Mape, kur atrodas CSV faili
csv_folder = "2d_gesture_recording"
calib_folder = "calibration_files"
csv_3d_folder = "3d_gesture_recording"
if not os.path.exists(csv_3d_folder):
    os.makedirs(csv_3d_folder)

csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

# Grupējam failus pa pāriem, izmantojot kreiso un labo failu nosaukumu daļas
pairs = {}
for file in csv_files:
    if "_left" in file:
        right_file = file.replace("_left", "_right")  # Nomainām "_left" uz "_right"
        if right_file in csv_files:  # Pārliecināmies, vai pastāv arī labais fails
            pairs[file] = right_file



# Apstrādājam katru failu pāri
for left_file, right_file in pairs.items():
    print(f"Apstrādāju failus: {left_file} un {right_file}")
    
    # Ielādē kreisās un labās kameras datus
    df_left = pd.read_csv(os.path.join(csv_folder, left_file))
    df_right = pd.read_csv(os.path.join(csv_folder, right_file))

        # Iegūstam kopīgo daļu no faila nosaukuma (gesture_0 un datums)
    left_name, right_name  = left_file.split('_left')  # Piemēram, "gesture_0_20250324-125936"
    right_name = right_name.split('.')[0]
    # Kalibrācijas faila nosaukums (pievienojam _calibration.npz)
    calib_file = f"{left_name}_calibration{right_name}.npz"
    calib_path = os.path.join(calib_folder, calib_file)

    # Pārbaudām, vai kalibrācijas fails eksistē
    if not os.path.exists(calib_path):
        print(f"❌ Nav atrasts kalibrācijas fails: {calib_file}")
        continue  # Ja fails netiek atrasts, pāriet uz nākamo failu pārim

    # Ielādē kalibrācijas parametrus
    calib = np.load(calib_path)
    mtx_l, dist_l = calib['mtx_l'], calib['dist_l']
    mtx_r, dist_r = calib['mtx_r'], calib['dist_r']
    R, T = calib['R'], calib['T']

    # Izveido projekcijas matricas
    P1 = np.hstack((mtx_l, np.zeros((3, 1))))  # P1 = K [I | 0]
    P2 = np.hstack((mtx_r @ R, mtx_r @ T))     # P2 = K [R | T]

    data_3d = []

    for i in range(len(df_left)):
        row = df_left.iloc[i].copy()  # Saglabājam laiku un žestu
        gesture, time = row["gesture"], row["time"]

        points_left, points_right = [], []

        for j in range(57):  # 42 roku punkti + 15 pose punkti
            x_left, y_left = df_left.iloc[i][f"x{j}"], df_left.iloc[i][f"y{j}"]
            x_right, y_right = df_right.iloc[i][f"x{j}"], df_right.iloc[i][f"y{j}"]

            if x_left == -1 or x_right == -1:  # Ja trūkst dati, ignorē šo punktu
                points_left.append([np.nan, np.nan])
                points_right.append([np.nan, np.nan])
            else:
                points_left.append([x_left, y_left])
                points_right.append([x_right, y_right])

        points_left = np.array(points_left, dtype=np.float32).T
        points_right = np.array(points_right, dtype=np.float32).T

        # Aprēķina 3D punktus ar triangulāciju
        points_4d_hom = cv2.triangulatePoints(P1, P2, points_left, points_right)
        points_3d = points_4d_hom[:3] / points_4d_hom[3]  # Pārvērš homogēnajās koordinātēs

        # Saglabā rezultātus
        row_3d = {"gesture": gesture, "time": time}
        for j in range(57):
            row_3d[f"x{j}"] = points_3d[0, j] if not np.isnan(points_3d[0, j]) else -1
            row_3d[f"y{j}"] = points_3d[1, j] if not np.isnan(points_3d[1, j]) else -1
            row_3d[f"z{j}"] = points_3d[2, j] if not np.isnan(points_3d[2, j]) else -1

        data_3d.append(row_3d)

    # Izveido DataFrame un saglabā
    df_3d = pd.DataFrame(data_3d)
    output_file = os.path.join(csv_3d_folder, left_file.replace("_left", "_3d"))
    df_3d.to_csv(output_file, index=False)

    print(f"Izveidots fails: {output_file}")
