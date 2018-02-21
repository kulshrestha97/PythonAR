import cv2
import numpy as np

image_add = cv2.imread('specs.png',-1)
image_mask = image_add[:,:,3]
image_inv_mask = cv2.bitwise_not(image_mask)
image_add = image_add[:,:,0:3]
orspecs_height, orspecs_width = image_add.shape[:2] 
video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("C:/Users/rajat/Documents/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("C:/Users/rajat/Documents/opencv/build/etc/haarcascades/haarcascade_eye.xml")
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the face
    # TODO: adjust x1 x2 and y1 y2 according to the face mask value
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyesCascade.detectMultiScale(roi_gray) 
        for (ex,ey,ew,eh) in eyes:
            
            #cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 1)
            specsWidth = 3 * ew
            specsHeight = (specsWidth*(orspecs_height/orspecs_width))
            x1 = ew - int(specsWidth/4)
            x2 = x + ew + int(specsWidth/4)
            y1 = int(y/3) + eh - int(specsHeight/2)
            y2 = int(y/3) + eh + int(specsHeight/2)
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x2 > w:
                x2 = w
            if y2 > h:
                y2 = h

            specsWidth = x2 - x1
            specsHeight = y2 - y1
            specs = cv2.resize(image_add, (specsWidth,specsHeight), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(image_mask, (specsWidth,specsHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(image_inv_mask, (specsWidth,specsHeight),interpolation=cv2.INTER_AREA)
            roi = roi_color[y1:y2, x1:x2]
            roi_bg = cv2.bitwise_and(roi, roi,mask= mask_inv)
            roi_fg = cv2.bitwise_and(specs,specs, mask = mask)
            dst = cv2.add(roi_bg,roi_fg)
            roi_color[y1:y2, x1:x2] = dst
            break


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()