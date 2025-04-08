# MBS3523-Asn2-Q4.py
import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)


is_on = False
gesture_detected = False
last_gesture_state = False
gesture_counter = 0
# Number of consecutive frames needed to switch state
GESTURE_FRAMES_THRESHOLD = 5

# Setup MediaPipe hand model
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Unable to get webcam feed.")
            continue

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)

        # Convert to RGB for MediaPipe processing
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Get image dimensions
        image_height, image_width, _ = image.shape

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Get thumb and index finger coordinates
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Convert to pixel coordinates
                thumb_x, thumb_y = int(thumb_tip.x * image_width), int(thumb_tip.y * image_height)
                index_x, index_y = int(index_tip.x * image_width), int(index_tip.y * image_height)

                # Calculate distance between the two fingertips
                distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

                # Get midpoint between thumb and index finger for heart display
                mid_x = (thumb_x + index_x) // 2
                mid_y = (thumb_y + index_y) // 2

                # Detect "heart" gesture - when two fingertips are close enough
                threshold = 50  # Adjust this threshold based on camera and hand size
                current_gesture = distance < threshold

                # Draw a line between fingertips
                if current_gesture:
                    heart_size = 20
                    cv2.line(image, (thumb_x, thumb_y), (index_x, index_y), (0, 0, 255), 2)
                    cv2.ellipse(image, (mid_x - heart_size // 4, mid_y), (heart_size // 2, heart_size // 2),
                                0, 0, 180, (0, 0, 255), -1)
                    cv2.ellipse(image, (mid_x + heart_size // 4, mid_y), (heart_size // 2, heart_size // 2),
                                0, 0, 180, (0, 0, 255), -1)
                    triangle_pts = np.array([[mid_x - heart_size // 2, mid_y],
                                             [mid_x + heart_size // 2, mid_y],
                                             [mid_x, mid_y + heart_size // 2]])
                    cv2.fillPoly(image, [triangle_pts], (0, 0, 255))
                else:
                    cv2.line(image, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 2)

                # State switching logic - use counter to ensure gesture stability
                if current_gesture != last_gesture_state:
                    gesture_counter = 0
                else:
                    gesture_counter += 1

                # If gesture remains stable for enough frames, switch state
                if gesture_counter >= GESTURE_FRAMES_THRESHOLD:
                    if current_gesture and not gesture_detected:
                        is_on = not is_on  # Toggle on/off state
                        gesture_detected = True
                    elif not current_gesture:
                        gesture_detected = False

                last_gesture_state = current_gesture

        status_text = "TURN ON" if is_on else "TURN OFF"
        status_color = (0, 255, 0) if is_on else (0, 0, 255)

        cv2.putText(image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, status_color, 4, cv2.LINE_AA)

        cv2.imshow('MediaPipe Gesture Control', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
