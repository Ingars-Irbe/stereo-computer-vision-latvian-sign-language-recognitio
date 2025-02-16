import cv2

# Atver abas kameras (0 un 1 ir tipiskas vērtības web kamerām)
cap1 = cv2.VideoCapture(0)  # Pirmā kamera
cap2 = cv2.VideoCapture(2)  # Otrā kamera

# Pārbauda, vai kameras ir atvērtas
if not cap1.isOpened() or not cap2.isOpened():
    print("Kļūda: Nevar atvērt kameru!")
    exit() 

while True:
    # Iegūst attēlus no abām kamerām
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # Pārbauda, vai attēli ir veiksmīgi iegūti
    if not ret1 or not ret2:
        print("Kļūda: Nevar nolasīt attēlus no kamerām!")
        break

    for y in range(0, frame1.shape[0], 50):
        cv2.line(frame1, (0, y), (frame1.shape[1], y), (0, 255, 0), 1)
        cv2.line(frame2, (0, y), (frame2.shape[1], y), (0, 255, 0), 1)

    # Parāda attēlus
    cv2.imshow('Kamera 1', frame1)
    cv2.imshow('Kamera 2', frame2)

    # Pārtrauc ciklu, ja tiek nospiests 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Atbrīvo resursus
cap1.release()
cap2.release()
cv2.destroyAllWindows()