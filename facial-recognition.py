
import cv2 
import face_recognition
import time
import RPi.GPIO as GPIO
import os

# Set the tolerance for face comparisons
TOLERANCE = 0.4

# GPIO setup
DOOR_OPEN_PIN = 24  # GPIO pin to open the door
 
GPIO.setmode(GPIO.BCM)
GPIO.setup(DOOR_OPEN_PIN, GPIO.OUT)
GPIO.output(DOOR_OPEN_PIN, GPIO.LOW)  # Initialize pin to low
 
# Names of people to recognize
people_names = ["bilal", "Person1", "Person2","Person3"]

# Load several pictures of each person and learn how to recognize each.
known_face_encodings = []
known_face_names = []
for person_name in people_names:
    person_folder = os.path.join("/home/billnumber1/Desktop/faces", person_name)
    for i in range(1, 51):
        image_path = os.path.join(person_folder, f"photo{i}.jpg")
        if not os.path.isfile(image_path):
            print(f"Image {image_path} does not exist")
            continue
        person_image = face_recognition.load_image_file(image_path)
        person_face_encoding_list = face_recognition.face_encodings(person_image)
        if person_face_encoding_list:
            person_face_encoding = person_face_encoding_list[0]
            known_face_encodings.append(person_face_encoding)
            known_face_names.append(person_name)

# Initialize some variables
face_locations = []
face_encodings = []

# Initialize the camera
video_capture = cv2.VideoCapture("rtsp://admin:L22937AB@192.168.0.102:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif")

# Frame counter setup
PROCESS_EVERY_NTH_FRAME = 7  # Change this to process every nth frame
frame_counter = 0

# Time tracker for delaying further signals
last_detection_time = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    frame_counter += 1
    if frame_counter % PROCESS_EVERY_NTH_FRAME != 0:
        continue

    # If the frame was successfully retrieved
    if ret:
        # Resize frame for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Find all faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_encoding in face_encodings:
            # Check if the face is a match for the known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, TOLERANCE)

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                current_time = time.time()
                if current_time - last_detection_time > 300:  # 5 minutes (300 seconds)
                    print(f"{name} detected!")
                    GPIO.output(DOOR_OPEN_PIN, GPIO.HIGH)  # Set the door open pin to high
                    last_detection_time = current_time

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        print("Failed to grab frame")
        break

# When everything is done, release the capture and cleanup GPIO
video_capture.release()
GPIO.output(DOOR_OPEN_PIN, GPIO.LOW)  # Set the door open pin back to low
GPIO.cleanup()

# Close all OpenCV windows
cv2.destroyAllWindows()
