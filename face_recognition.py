# Face Recognition

# Importing the libraries
import cv2

# Loading the cascades with respect to eyes and face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Defining a function that will do the detections
def detect(gray, frame):
    # 1.3 and 5 are good combo, sizescale is to magnify the faces in xml
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)
    # x and y are coordinates, w and h are the size
    for (x, y, w, h) in faces:
        #(x,y) is upper left corner and (x+w,y+h), blue and edge is 2 that is smoother 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #detect eyes in face frame
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        #two eyes and coordinates of two eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    # return two frams 
    return frame

# Doing some Face Recognition with the webcam(internal 0 and external 1)
video_capture = cv2.VideoCapture(0)


while True:
    #first element is not useful
    _, frame = video_capture.read()
    # color transformation
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect function
    canvas = detect(gray, frame)
    
    cv2.imshow('Video', canvas)
     if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
