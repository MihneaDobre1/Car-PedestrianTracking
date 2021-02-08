import cv2
import random
trained_car_data = cv2.CascadeClassifier('Resources/cars.xml')
#trained_pedestrian_data = cv2.CascadeClassifier('Resources/haarcascade_fullbody.xml')
#Capture camera 0 - default camera
webcam = cv2.VideoCapture("Resources/dokkerDepasire.mp4")

while True:
    #Read the current frame
    (successful_frame_read, frame) = webcam.read()
    #Must convert to grayscale
    if successful_frame_read:
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    car_coordinates = trained_car_data.detectMultiScale(grayscaled_img)
    #pedestrian_coordinates = trained_pedestrian_data.detectMultiScale(grayscaled_img,scaleFactor=10.1,minNeighbors=2)

    # for i in range(len(pedestrian_coordinates)):
    #
    #     (m, n, o, p) = pedestrian_coordinates[i]
    #     # print(len(face_coordinates))
    #
    #     cv2.rectangle(frame, (m, n), (m + o, n + p),
    #                   (0, 255, 255), 1)

    for i in range(len(car_coordinates)):

        (x, y, w, h) = car_coordinates[i]
        # print(len(face_coordinates))

        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 0, 255), 1)
    cv2.imshow("Car Recongition",frame)
    cv2.waitKey(1)


print("Code Completed")
