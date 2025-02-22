import numpy as np
import cv2

# Šaha galdiņa parametri
CHECKERBOARD = (7, 5)  # Iekšējie stūru punkti (platums, augstums)

# Sagatavojam 3D objektu punktus
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Saglabāšanas struktūras
obj_points = []     # 3D punkti
img_points_left = []  # 2D punkti kreisajai kamerai
img_points_right = [] # 2D punkti labajai kamerai

# Inicializējam kameras
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(2)

# Pārbaudam kameru pieejamību
if not cap_left.isOpened() or not cap_right.isOpened():
    print("Kļūda: Nevar atvērt kameru!")
    exit()

# Iegūstam sākotnējo attēlu izmēru
ret, img = cap_left.read()
if not ret:
    print("Kļūda: Nevar nolasīt attēlu no kreisās kameras!")
    exit()

img_size = img.shape[1::-1]  # Iegūstam (width, height)

# Kalibrācijas parametri
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

print("Nospiediet 's', lai saglabātu attēlus kalibrācijai")
print("Nospiediet 'q', lai pārtrauktu un sāktu kalibrāciju")

while True:
    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()

    if not ret_l or not ret_r:
        print("Kļūda: Nevar nolasīt attēlus!")
        break

    gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

    # Meklējam šaha galdiņa stūrus
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)
    # if ret_l:
    #     print("L")
    # if ret_r:
    #     print("R")
    # Ja abās kamerās atrodam stūrus
    if ret_l and ret_r:
        # Uzlabojam stūru precizitāti
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria)
        # print("Atrasts")
        # Zīmējam atrastos stūrus
        cv2.drawChessboardCorners(frame_l, CHECKERBOARD, corners_l, ret_l)
        cv2.drawChessboardCorners(frame_r, CHECKERBOARD, corners_r, ret_r)

    # Parādām attēlus
    cv2.imshow('Left Camera', frame_l)
    cv2.imshow('Right Camera', frame_r)

    key = cv2.waitKey(1)
    if key == ord('s') and ret_l and ret_r:
        # Saglabājam punktus
        obj_points.append(objp)
        img_points_left.append(corners_l)
        img_points_right.append(corners_r)
        print(f"Saglabāti {len(obj_points)} attēli")

    elif key == ord('q'):
        break

# Notīram logus
cv2.destroyAllWindows()

# Pārbaudam, vai ir pietiekami daudz datu
if len(obj_points) < 10:
    print("Kļūda: Nepietiekami daudz kalibrācijas attēlu (vajag vismaz 10)!")
    exit()

# Veicam stereo kalibrāciju
print("Sākas kalibrācija...")
flags = cv2.CALIB_FIX_INTRINSIC
ret, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    obj_points,
    img_points_left,
    img_points_right,
    None, None, None, None,
    img_size,
    criteria=criteria,
    flags=flags
)

# Saglabājam rezultātus
np.savez("stereo_calib.npz",
         mtx_l=mtx_l, dist_l=dist_l,
         mtx_r=mtx_r, dist_r=dist_r,
         R=R, T=T, E=E, F=F)

print("Kalibrācija pabeigta!")
print("Reprojekcijas kļūda:", ret)

# Parādām undistortētus attēlus
ret, frame_l = cap_left.read()
ret, frame_r = cap_right.read()

if ret:
    undistorted_l = cv2.undistort(frame_l, mtx_l, dist_l)
    undistorted_r = cv2.undistort(frame_r, mtx_r, dist_r)

    cv2.imshow('Undistorted Left', undistorted_l)
    cv2.imshow('Undistorted Right', undistorted_r)
    cv2.waitKey(0)

# Atbrīvojam resursus
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()