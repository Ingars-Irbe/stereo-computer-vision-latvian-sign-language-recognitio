import cv2
import numpy as np
import os
import time

# Izveido direktoriju, ja tāda vēl nav
output_dir = "recorded_videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Inicializē kameras
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(2)

# Pārbauda, vai kameras ir atvērtas
if not cap_left.isOpened() or not cap_right.isOpened():
    print("Neizdevās atvērt vienu vai abas kameras.")
    exit()

# Iegūst kadra izmērus
frame_width = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap_left.get(cv2.CAP_PROP_FPS)) or 25  # Noklusējuma FPS, ja nav pieejams

# Ielādē kalibrācijas datus
calib = np.load('stereo_calib.npz')
mtx_l, dist_l = calib['mtx_l'], calib['dist_l']
mtx_r, dist_r = calib['mtx_r'], calib['dist_r']
R, T, E, F = calib['R'], calib['T'], calib['E'], calib['F']

recording = False
start_time = 0
current_label = 0  # Noklusējuma žesta marķējums

print("Spiediet 's', lai sāktu ierakstīšanu. Spiediet 'q', lai izietu.")
print("Izmantojiet ciparus 0-9, lai marķētu žestus.")

while True:
    ret_l, frame_left = cap_left.read()
    ret_r, frame_right = cap_right.read()

    if not ret_l or not ret_r:
        print("⚠️ Kļūda, nolasot video straumi no vienas vai abām kamerām.")
        break

    # Parāda video tiešraidi
    cv2.putText(frame_left, f"Current Label: {current_label}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Left Camera', frame_left)
    cv2.imshow('Right Camera', frame_right)

    # Pārbauda, vai ir jāsāk ierakstīšana
    if recording:
        # Ģenerē failu nosaukumus, pievienojot žesta marķējumu
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        video_left_filename = os.path.join(output_dir, f"gesture_{current_label}_left_{timestamp}.avi")
        video_right_filename = os.path.join(output_dir, f"gesture_{current_label}_right_{timestamp}.avi")
        calibration_filename = os.path.join(output_dir, f"gesture_{current_label}_calibration_{timestamp}.npz")

        # Inicializē video rakstītājus
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_left = cv2.VideoWriter(video_left_filename, fourcc, fps, (frame_width, frame_height))
        out_right = cv2.VideoWriter(video_right_filename, fourcc, fps, (frame_width, frame_height))

        print(f"📹 Sāk ierakstīšanu: {video_left_filename} un {video_right_filename}")

        record_start = time.time()

        while time.time() - record_start < 4:
            ret_l, frame_left = cap_left.read()
            ret_r, frame_right = cap_right.read()

            if not ret_l or not ret_r:
                print("⚠️ Kļūda, nolasot video straumi ierakstīšanas laikā.")
                break


            # cv2.putText(frame_left, f"Recording Label: {current_label}", (30, 50),
                        # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Left Camera', frame_left)
            cv2.imshow('Right Camera', frame_right)
            
            out_left.write(frame_left)
            out_right.write(frame_right)

        # Saglabā kalibrācijas failu
        np.savez(calibration_filename, mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r, R=R, T=T, E=E, F=F)
        print(f"📁 Kalibrācijas dati saglabāti kā: {calibration_filename}")

        # Atbrīvo resursus
        out_left.release()
        out_right.release()

        print("✅ Ierakstīšana pabeigta.")
        recording = False

    # Sāk ierakstīšanu, kad nospiesta 's'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not recording:
        recording = True
        start_time = time.time()
        print("⏳ Ierakstīšana sāksies pēc 1 sekundes...")

    # Maina žesta marķējumu, nospiežot ciparus 0-9
    if ord('0') <= key <= ord('9'):
        current_label = int(chr(key))
        print(f"🔖 Izvēlēts žesta marķējums: {current_label}")

    # Beidz programmu, kad nospiests 'q'
    if key == ord('q'):
        print("🛑 Programma beidz darbu.")
        break

# Atbrīvo resursus
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
