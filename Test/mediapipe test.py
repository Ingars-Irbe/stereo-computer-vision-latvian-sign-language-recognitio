import cv2
import mediapipe as mp

# MediaPipe atpazīšanas risinājumi
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands  # Lai iegūtu HAND_CONNECTIONS

# Atver kameru
cap = cv2.VideoCapture(2)

# Inicializē MediaPipe Holistic modeli
with mp_holistic.Holistic(
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Kļūda: Nevar nolasīt attēlu no kameras!")
            break

        # Pārveido attēlu no BGR uz RGB (MediaPipe prasa RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apstrādā attēlu un atpazīst pozas un rokas
        results = holistic.process(image)

        # Pārveido attēlu atpakaļ uz BGR, lai parādītu ar OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Ja ir atpazītas rokas, uzzīmē to locītavas
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Ja ir atpazīta poza, zīmē tikai plecus, elkoņus un sejas punktus
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks

            # Izveido savienojumus starp pleciem un elkoņiem
            connections = [
                (mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.LEFT_ELBOW),
                (mp_holistic.PoseLandmark.RIGHT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_ELBOW)
            ]

            # Sejas orientieri (galvenie punkti)
            face_points = [
                mp_holistic.PoseLandmark.NOSE, 
                mp_holistic.PoseLandmark.LEFT_EYE, mp_holistic.PoseLandmark.RIGHT_EYE,
                mp_holistic.PoseLandmark.LEFT_EAR, mp_holistic.PoseLandmark.RIGHT_EAR,
                mp_holistic.PoseLandmark.MOUTH_LEFT, mp_holistic.PoseLandmark.MOUTH_RIGHT,
                mp_holistic.PoseLandmark.LEFT_EYE_INNER, mp_holistic.PoseLandmark.RIGHT_EYE_INNER
            ]

            # Uzzīmē tikai plecus un elkoņus, kā arī savieno tos
            for start, end in connections:
                start_point = pose_landmarks.landmark[start]
                end_point = pose_landmarks.landmark[end]

                # Zīmē savienojumu starp pleciem un elkoņiem
                cv2.line(image, 
                         (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0])),
                         (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0])),
                         (0, 255, 0), 2)

                # Zīmē punktus (plecus un elkoņus)
                cv2.circle(image,
                           (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0])),
                           5, (255, 0, 0), -1)
                cv2.circle(image,
                           (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0])),
                           5, (255, 0, 0), -1)

            # Zīmē sejas orientierus
            for point in face_points:
                lm = pose_landmarks.landmark[point]
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                cv2.circle(image, (x, y), 4, (0, 255, 255), -1)  # Dzeltens punkts

        # Parāda attēlu
        cv2.imshow('Roku un Sejas atpazīšana', image)

        # Pārtrauc ciklu, ja tiek nospiests 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Atbrīvo resursus
cap.release()
cv2.destroyAllWindows()
