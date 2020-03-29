import face_recognition
import cv2
import numpy as np
speed=3# change this value to improve performance

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

### Load a sample picture and learn how to recognize it.
harsh_image = face_recognition.load_image_file("harsh.png")
harsh_face_encoding = face_recognition.face_encodings(harsh_image)[0]

# Load a second sample picture and learn how to recognize it.
tamanna_image = face_recognition.load_image_file("tamanna.png")
tamanna_face_encoding = face_recognition.face_encodings(tamanna_image)[0]

# Load a second sample picture and learn how to recognize it.
shruti_image = face_recognition.load_image_file("shruti.png")
shruti_face_encoding = face_recognition.face_encodings(shruti_image)[0]

# Load a second sample picture and learn how to recognize it.
palak_image = face_recognition.load_image_file("palak.png")
palak_face_encoding = face_recognition.face_encodings(palak_image)[0]


# Load a second sample picture and learn how to recognize it.
sajjan_image = face_recognition.load_image_file("sajjan.png")
sajjan_face_encoding = face_recognition.face_encodings(sajjan_image)[0]

# Load a second sample picture and learn how to recognize it.
gita_image = face_recognition.load_image_file("gita.png")
gita_face_encoding = face_recognition.face_encodings(gita_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    harsh_face_encoding,
    tamanna_face_encoding,
    shruti_face_encoding,
    palak_face_encoding,
    sajjan_face_encoding,
    gita_face_encoding,
    
]
known_face_names = [
    "Harsh",
    "Tamanna",
    "shruti",
    "palak",
    "sajjan",
    "gita"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame=0
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    if ((process_this_frame%speed)==0):
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    if(process_this_frame==speed+1):
        process_this_frame=0
    else:
        process_this_frame+=1

        
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
