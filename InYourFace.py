'''
In Your Face
A program that automatically scrolls webpages when you blink
Author: Derek Xu
derek_yhx@gmail.com
GitHub: https://github.com/dxiled

Blink detection adapted from eye-blink-detection-demo by mans-men on GitHub
https://github.com/mans-men/eye-blink-detection-demo

Mouth open detection is my own
'''

# import the necessary packages
#    computer vision
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
#    scroll control
import pyautogui

def eye_aspect_ratio(eye):
     # compute the euclidean distances between the two sets of
     # vertical eye landmarks (x, y)-coordinates
     A = dist.euclidean(eye[1], eye[5])
     B = dist.euclidean(eye[2], eye[4])

     # compute the euclidean distance between the horizontal
     # eye landmark (x, y)-coordinates
     C = dist.euclidean(eye[0], eye[3])

     # compute the eye aspect ratio
     ear = (A + B) / (2.0 * C)

     # return the eye aspect ratio
     return ear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",default="shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="camera",
                help="path to input video file")
ap.add_argument("-t", "--threshold", type = float, default=0.19,
                help="threshold to determine closed eyes")
ap.add_argument("-f", "--frames", type = int, default=2,
                help="the number of consecutive frames the eye must be below the threshold")
ap.add_argument("-m", "--mthreshold", type = float, default=0.7,
                help="threshold to determine closed mouth")

def main() :
     args = vars(ap.parse_args())
     EYE_AR_THRESH = args['threshold']
     EYE_AR_CONSEC_FRAMES = args['frames']
     MOUTH_AR_THRESH = args['mthreshold']

     # initialize the frame counters and the total number of blinks
     COUNTER = 0
     BLINK = 0
     MCOUNTER = 0
     MOUTH = 0

     # initialize dlib's face detector (HOG-based) and then create
     # the facial landmark predictor
     print("[INFO] loading facial landmark predictor...")
     detector = dlib.get_frontal_face_detector()
     predictor = dlib.shape_predictor(args["shape_predictor"])

     # grab the indexes of the facial landmarks for the left and
     # right eye, and mouth, respectively
     (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
     (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
     (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

     # start the video stream thread
     print("[INFO] starting video stream thread...")
     print("[INFO] print q to quit...")
     if args['video'] == "camera":
          vs = VideoStream(src=0).start()
          fileStream = False
     else:
          vs = FileVideoStream(args["video"]).start()
          fileStream = True

     time.sleep(1.0)

     # loop over frames from the video stream
     while True:
          # if this is a file video stream, then we need to check if
          # there any more frames left in the buffer to process
          if fileStream and not vs.more():
               break

          # grab the frame from the threaded video file stream, resize
          # it, and convert it to grayscale
          # channels)
          frame = vs.read()
          frame = imutils.resize(frame, width=450)
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

          # detect faces in the grayscale frame
          rects = detector(gray, 0)

          # loop over the face detections
          for rect in rects:
               # determine the facial landmarks for the face region, then
               # convert the facial landmark (x, y)-coordinates to a NumPy
               # array
               shape = predictor(gray, rect)
               shape = face_utils.shape_to_np(shape)

               # extract the left and right eye coordinates, then use the
               # coordinates to compute the eye aspect ratio for both eyes
               leftEye = shape[lStart:lEnd]
               rightEye = shape[rStart:rEnd]
               leftEAR = eye_aspect_ratio(leftEye)
               rightEAR = eye_aspect_ratio(rightEye)
               
               # extract the mouth coordinates, then use the coordinates to
               # build a rectangle and compute the mouth aspect ratio
               mouth = shape[mStart:mEnd]
               mouthHull = cv2.convexHull(mouth)
               mx, my, mw, mh = cv2.boundingRect(mouthHull)
               
               MAR = mh/mw
               # average the eye aspect ratio together for both eyes
               ear = (leftEAR + rightEAR) / 2.0

               # compute the convex hull for the left and right eye, then
               # mouth, then visualize each of the features
               leftEyeHull = cv2.convexHull(leftEye)
               rightEyeHull = cv2.convexHull(rightEye)
               cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
               cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
               cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
               cv2.rectangle(frame, (mx, my), (mx+mw, my+mh), (0, 255, 0), 1)

               # check to see if the eye aspect ratio is below the blink
               # threshold, and if so, increment the blink frame counter
               if ear < EYE_AR_THRESH:
                    COUNTER += 1

               # otherwise, the eye aspect ratio is not below the blink
               # threshold
               else:
                    # if the eyes were closed for a sufficient number of
                    # then either count a blink, or reset the blink and scroll down
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                         BLINK = 1

                    # reset the eye frame counter
                    COUNTER = 0
               
               if MAR < MOUTH_AR_THRESH:
                    MCOUNTER += 1
               else:
                    if MCOUNTER >= EYE_AR_CONSEC_FRAMES:
                         MOUTH = 1
                    
                    MCOUNTER = 0
               
               # draw the total number of blinks on the frame along with
               # the computed eye aspect ratio for the frame
               cv2.putText(frame, "Blink: {}".format(BLINK), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
               cv2.putText(frame, "Mouth: {}".format(MOUTH), (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
               
               #Scroll if the user blinked
               if BLINK:
                    pyautogui.scroll(-100)
                    BLINK = 0
               if MOUTH:
                    pyautogui.scroll(100)
                    MOUTH = 0

          # show the frame
          cv2.imshow("In Your Face", frame)
          key = cv2.waitKey(1) & 0xFF

          # if the 'q' key was pressed, break from the loop
          if key == ord("q"):
               break

     # do a bit of cleanup
     cv2.destroyAllWindows()
     vs.stop()
     
if __name__ == '__main__' :
     main()
