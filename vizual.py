import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ielādē kalibrācijas parametrus
calib = np.load('stereo_calib.npz')
mtx_l, dist_l = calib['mtx_l'], calib['dist_l']
mtx_r, dist_r = calib['mtx_r'], calib['dist_r']
R, T = calib['R'], calib['T']

# Izveido projekcijas matricas
P1 = np.hstack((mtx_l, np.zeros((3, 1))))  # P1 = K [I | 0]
P2 = np.hstack((mtx_r @ R, mtx_r @ T))     # P2 = K [R | T]

# Inicializē MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inicializē kameras
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(2)

# Sagatavo Matplotlib 3D vizualizācijas logu
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')

def update_3d_plot(points_3d):
    """Atjauno 3D vizualizāciju ar jauniem punktiem."""
    ax.clear()
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Rokas Punktu Mākoņa Vizualizācija")

    # Filtrē nederīgus punktus (Z vērtības)
    valid_mask = abs(points_3d[2]) > 0  # Atmet negatīvus Z (priekšmetiem aiz kameras)
    X, Y, Z = points_3d[:, valid_mask]  # Atlasām tikai derīgos punktus
    Z = abs(Z)
    X = abs(X)
    Y = abs(Y)
    # Pārbauda, vai ir derīgi punkti
    if len(X) == 0:
        print("❌ NAV DERĪGU 3D PUNKTU!")
        return

    ax.scatter(X, Z, Y, c='b', marker='o')

    # Pielāgo asu robežas dinamiski
    ax.set_xlim([25, 10])
    ax.set_ylim([0.2, 1])
    ax.set_zlim([25, 10])

    # Savieno punktus, lai attēlotu pirkstus
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Īkšķis
        (0, 5), (5, 6), (6, 7), (7, 8),  # Rādītājpirksts
        (0, 9), (9, 10), (10, 11), (11, 12),  # Vidējais pirksts
        (0, 13), (13, 14), (14, 15), (15, 16),  # Zeltnesis
        (0, 17), (17, 18), (18, 19), (19, 20)  # Mazais pirksts
    ]

    for i, j in connections:
        if valid_mask[i] and valid_mask[j]:
            ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]], 'r')

    plt.draw()
    plt.pause(0.001)

plt.ion()  # Interaktīvs režīms

while cap_left.isOpened() and cap_right.isOpened():
    ret_l, frame_left = cap_left.read()
    ret_r, frame_right = cap_right.read()
    if not ret_l or not ret_r:
        break

    # Konvertē attēlus uz RGB (MediaPipe prasa RGB)
    frame_left_rgb = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)
    frame_right_rgb = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
    
    # Apstrādā attēlus ar MediaPipe Hands
    results_left = hands.process(frame_left_rgb)
    results_right = hands.process(frame_right_rgb)
    
    if results_left.multi_hand_landmarks and results_right.multi_hand_landmarks:
        hand_left = results_left.multi_hand_landmarks[0]
        hand_right = results_right.multi_hand_landmarks[0]

        h_left, w_left, _ = frame_left.shape
        h_right, w_right, _ = frame_right.shape
        
        points_left, points_right = [], []

        # Apstrādā visus 21 rokas punktus
        for i in range(21):  
            lm_left = hand_left.landmark[i]
            lm_right = hand_right.landmark[i]
            
            x_left, y_left = int(lm_left.x * w_left), int(lm_left.y * h_left)
            x_right, y_right = int(lm_right.x * w_right), int(lm_right.y * h_right)

            cv2.circle(frame_left, (x_left, y_left), 5, (0, 255, 0), -1)
            cv2.circle(frame_right, (x_right, y_right), 5, (0, 255, 0), -1)

            points_left.append([x_left, y_left])
            points_right.append([x_right, y_right])

        # Pārvērš punktus par NumPy masīviem (2xN formātā)
        points_left = np.array(points_left, dtype=np.float32).T  # (2, N)
        points_right = np.array(points_right, dtype=np.float32).T  # (2, N)

        # Triangulācija: 2D -> 3D pārveide
        points_4d_hom = cv2.triangulatePoints(P1, P2, points_left, points_right)
        points_3d = points_4d_hom[:3] / points_4d_hom[3]  # Pārvērš homogēnās koordinātas uz 3D

        # Atjauno 3D vizualizāciju
        update_3d_plot(points_3d)

    # Parāda attēlus
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
plt.ioff()  # Beidz interaktīvo režīmu
plt.show()
