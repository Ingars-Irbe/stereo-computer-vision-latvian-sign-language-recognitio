import cv2
import numpy as np
import pandas as pd
import os

# CSV failu mape
csv_folder = "recorded_videos"
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

# Iestatījumi attēla izmēram
image_width = 640
image_height = 480

for csv_file in csv_files:
    csv_path = os.path.join(csv_folder, csv_file)
    df = pd.read_csv(csv_path)

    print(f"Vizualizēju failu: {csv_file}")

    for index, row in df.iterrows():
        # Izveido tukšu attēlu (melnu)
        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

        def draw_points(image, row, start_idx, num_points, color):
            """Zīmē punktus uz attēla no CSV rindas"""
            for i in range(num_points):
                x, y = row[start_idx + i*2], row[start_idx + i*2 + 1]
                if x != -1 and y != -1:  # Ja punkts ir atpazīts
                    cx, cy = int(x * image_width), int(y * image_height)
                    cv2.circle(image, (cx, cy), 5, color, -1)

        # Roku punkti
        draw_points(image, row, 2, 21, (0, 255, 0))   # Labā roka (zaļa)
        draw_points(image, row, 44, 21, (255, 0, 0))  # Kreisā roka (zila)

        # Pose punkti
        draw_points(image, row, 86, 15, (0, 0, 255))  # Pozas punkti (sarkana)

        # Parāda attēlu
        cv2.imshow(csv_file, image)

        # FPS atkarīgs no CSV datiem (var pielāgot)
        key = cv2.waitKey(50)  # Pauze starp kadriem (50ms)
        if key == ord('q'):  # Iziet ar 'q'
            break

    cv2.destroyAllWindows()
