import cv2
import mediapipe as mp
import numpy as np

# MediaPipe atpazīšanas risinājumi
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands  

# Norādi video failu
video_path1 = "recorded_videos/gesture_0_left_20250324-130111.avi"

# Atver video failu
cap1 = cv2.VideoCapture(video_path1)

# Inicializē MediaPipe Holistic modeli
with mp_holistic.Holistic(
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9) as holistic:

    while cap1.isOpened():
        ret1, frame1 = cap1.read()
        if not ret1:
            print("Video beidzies vai nevar nolasīt attēlu!")
            break

        # Pārveido attēlu no BGR uz RGB
        image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        # Apstrādā attēlu un atpazīst pozas un rokas
        results1 = holistic.process(image1)

        # Pārveido attēlu atpakaļ uz BGR
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)

        def draw_landmarks_with_indices(image, results):
            """
            Zīmē orientierus un pievieno to indeksus:
            - No `pose_landmarks` attēlo tikai punktus 0-14.
            - Plaukstas attēlo tikai no `hand_landmarks`.
            """
            # Zīmē pozu (tikai 0-14 punkti)
            if results.pose_landmarks:
                for idx in range(15):  # Tikai punkti 0-14
                    lm = results.pose_landmarks.landmark[idx]
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 3, (255, 0, 0), -1)  # Zilā krāsā
                    cv2.putText(image, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.4, (255, 0, 0), 1, cv2.LINE_AA)

            # Zīmē plaukstu orientierus no `hand_landmarks`
            for hand_landmarks in [results.right_hand_landmarks, results.left_hand_landmarks]:
                if hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for idx, lm in enumerate(hand_landmarks.landmark):
                        h, w, _ = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)  # Zaļā krāsā
                        cv2.putText(image, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # Uzzīmē orientierus ar indeksiem
        draw_landmarks_with_indices(image1, results1)

        # Parāda attēlu
        cv2.imshow('Roku un Sejas atpazīšana (Video)', image1)

        # Iegūst FPS un kontrolē ātrumu
        fps = int(cap1.get(cv2.CAP_PROP_FPS))
        delay = int(1000 / fps) if fps > 0 else 30  

        # Pārtrauc ciklu, ja tiek nospiests 'q'
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

# Atbrīvo resursus
cap1.release()
cv2.destroyAllWindows()
