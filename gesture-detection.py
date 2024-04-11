import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands.Hands()

# Function to extract fixed number of landmarks from a frame
def extract_fixed_landmarks(frame, num_landmarks=21):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb_frame)
    landmarks = []
    # Extract fixed no. of hand landmarks if available
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark[:num_landmarks]:
                landmarks.append((lm.x, lm.y, lm.z))
    return landmarks

# Resize image with maintaining aspect ratio
def resize_with_aspect_ratio(image, target_size):
    height, width = image.shape[:2]
    target_width, target_height = target_size

    aspect_ratio = width / height

    # Calculate new dimensions while maintaining aspect ratio
    if aspect_ratio > 1:  
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:  
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

# Function to annotate detected gesture on the frame
def annotate_detection(frame):
    cv2.putText(frame, "DETECTED", (frame.shape[1] - 170, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Compare gesture representation from the video with gesture representation from the image
def compare_gesture(video_path, image_path):
    
    image = cv2.imread(image_path)
    image = resize_with_aspect_ratio(image, (1000, 800))
    # Extract landmarks from the resized image
    image_landmarks = extract_fixed_landmarks(image)
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_with_aspect_ratio(frame, (1000, 800))
        frame_landmarks = extract_fixed_landmarks(frame)
        
        similarity_score = compare_landmarks(frame_landmarks, image_landmarks)
        
        if similarity_score > 0.79:
            annotate_detection(frame)
        
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function to compare hand landmarks from video frame with gesture representation from an image
def compare_landmarks(landmarks1, landmarks2):
    if len(landmarks1) != len(landmarks2):
        return 0.0  
    distances = []
    for lm1, lm2 in zip(landmarks1, landmarks2):
        dist = np.linalg.norm(np.array(lm1) - np.array(lm2))
        distances.append(dist)
    # Calculate similarity score as the mean of the distances normalized by the maximum possible distance
    max_distance = np.linalg.norm(np.array([1.0, 1.0, 1.0]))  #
    similarity_score = 1.0 - (np.mean(distances) / max_distance)
    return similarity_score

# Paths to video and image
video_path = "test_video3.mp4"
image_path = "gesture_image.jpg"

compare_gesture(video_path, image_path)
