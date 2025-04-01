import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# CSV failu mape
csv_folder = "3d_gesture_recording"
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

for csv_file in csv_files:
    csv_path = os.path.join(csv_folder, csv_file)
    df = pd.read_csv(csv_path)
    
    print(f"Vizualizēju failu: {csv_file}")
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Fiksēts diapazons asīm


    # Pirmajā reizē uzstādām uzstādīto diapazonu, pēc tam nemainām
    ax.set_xlabel("X koordināta")
    ax.set_ylabel("Y koordināta")
    ax.set_zlabel("Z koordināta")
    ax.legend()
    
    # Attēloto punktu funkcija
    def plot_points(ax, row, start_idx, num_points, color, label):
        """Attēlo punktus 3D telpā, filtrējot nederīgos punktus"""
        points = []
        for i in range(num_points):
            
            x, y, z = row[start_idx + i*3], row[start_idx + i*3 + 1], row[start_idx + i*3 + 2]
            if x != -1 and y != -1 and z != -1:  # Izlaižam trūkstošos datus
                points.append([x, y, z])
       
        if points:
            points = np.array(points)
            X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
            Z = abs(Z)
            X = abs(X)
            Y = abs(Y)
            ax.scatter(X, Z, Y, c=color, label=label, s=20)

    for index, row in df.iterrows():
        # Attēlot punktus
        ax.clear()
        ax.set_xlabel("X koordināta")
        ax.set_ylabel("Y koordināta")
        ax.set_zlabel("Z koordināta")
        ax.set_xlim([0, 70])  # X ass diapazons
        ax.set_ylim([15, 75])  # Y ass diapazons
        ax.set_zlim([70, 0])     # Z ass diapazons
        plot_points(ax, row, 2, 21, 'g', 'Labā roka')
        plot_points(ax, row, 65, 21, 'b', 'Kreisā roka')
        plot_points(ax, row, 128, 15, 'r', 'Pozīcija')
        
        ax.set_title(f"3D Punkti no {csv_file}, kadrs {index}")
        
        plt.draw()
        plt.pause(0.1)  # Ir nepieciešams neliels pauzēšana, lai attēlošana būtu pamanāma
    
    plt.close(fig)
