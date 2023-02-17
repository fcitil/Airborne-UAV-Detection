from datetime import datetime
import time
import glob
import cv2
import torch
from cv2 import rectangle
import numpy as np
from os.path import realpath, dirname, join
import time
from net import SiamRPNvot
#from net import SiamRPNotb
#from net import SiamRPNBIG

# modules for tracking model
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect, im_to_numpy, area, in_locking_rect, large_enough, fsm

# get supported device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Algorithm runs on {}".format(device))

# load net for Tracker
tracking_net = SiamRPNvot()
tracking_net.load_state_dict(
    torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model'),
               map_location=device))
tracking_net.eval().to(device)

# Opencv DNN for YOLOv4_tiny
detection_net = cv2.dnn.readNet("yolov4_tiny_custom_best.weights",
                                "yolov4_tiny_custom_test.cfg")

# Create detection model
detection_model = cv2.dnn_DetectionModel(detection_net)
detection_model.setInputParams(size=(320, 320), scale=1 / 255)

# initial states for fsm
states = [False, False, False, False, False]
"""
# Saving video
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('output.avi', fourcc, 30.0, (1080, 720))
"""

# Load class lists for detection

classes = []
with open("classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        # print(class_name)
        class_name = class_name.strip()  # spaces between lines
        classes.append(class_name)

# Initialize camera/ video
cap = cv2.VideoCapture("Misty FPV Close Proximity Chase.mp4")

cnt = 0  # frame counter
detection_initiated = False  # False until the first detection, then True
object_detected = False
rect = [0, 0, 0, 0]  # bounding box of the object
# timestamps of the locking and unlocking of the object
locking_timestamps = [0, 0]
start_time = time.time()  # start time of the iteration
while True:

    isLocked = False
    # Get frames
    ret, im = cap.read()
    cnt += 1
    # EXPERIMENTAL
    if cnt <= 100:
        print("Frame: ", cnt)
        continue
    # EXPERIMENTAL
    # Timestamp for synchronization
    duration_tic = time.time()
    print(duration_tic)
    """"
    currentDateandtime = datetime.now()
    print(currentDateandtime.strftime('%Y-%m-%d %H:%M:%S.%f'))
    print(currentDateandtime.strftime('%Y-%m-%d %H:%M:%S.%f')[-1:])
    print("for frame : " + str(cnt) + "   saat: ", str(currentDateandtime.hour))
    print("for frame : " + str(cnt) + "   dakika: ", str(currentDateandtime.minute))
    print("for frame : " + str(cnt) + "   saniye: ", str(currentDateandtime.second))
    print("for frame : " + str(cnt) + "   milisaniye: ", str(currentDateandtime.microsecond)[:-3])
    """
    """
    #############EXPERIMENTAL#############
    cv2.resize(im, (1200, 1920))
    im = im[300:900, 240:1660]
    if cnt <= 100:
        print(cnt)
        continue
    ###############END#####################
    """
    im_x, im_y = im.shape[1], im.shape[0]
    print("Current Frame is:{}".format(cnt))
    tic_detection = cv2.getTickCount()
    tic = cv2.getTickCount()
    # number frame to skip detectionq
    frame_per_detection = 10
    if cnt % frame_per_detection == 0 or not detection_initiated or not object_detected:
        detection_initiated = True
        # Object Detection
        (class_ids, scores, bboxes) = detection_model.detect(im,
                                                             confThreshold=0.7,
                                                             nmsThreshold=.35)
        bboxes = list(bboxes)

        if len(bboxes) > 1:
            bboxes.sort(key=area)
            bboxes = bboxes[-1:]
            print(bboxes)
        toc_detection = cv2.getTickCount()
        print("Detection FPS: {}".format(
            1 / ((toc_detection - tic_detection) / cv2.getTickFrequency())))

        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            object_detected = True
            isLocked = True
            (x, y, w, h) = bbox
            rect = [x, y, x + w, y + h]
            #cv2.rectangle(im, (x, y), (x + w, y + h), (0,0,255), 3)
            class_name = classes[class_id]

            cv2.putText(im, class_name + "" + "%.2f" % score, (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN, 3, (200, 0, 50), 1)

            cx, cy = x + w / 2, y + h / 2
            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
            tracker = SiamRPN_init(im, target_pos, target_sz, tracking_net,
                                   device)
            Tracker_is_initilized = True

    elif Tracker_is_initilized:
        # Track obejct
        state = SiamRPN_track(tracker, im, device)
        isLocked = True
        # Draw bbox
        cv2.putText(im, "Tracking Score: {:.2f}".format(state['score']),
                    (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1)
        #cv2.putText(im, str(state['score']), (20, 40), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,100), 2)
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        res = [int(l) for l in res]
        rect = [res[0], res[1], res[0] + res[2], res[1] + res[3]]
        #cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
        if state['score'] < 0.7:
            Tracker_is_initilized = False
            object_detected = False
            isLocked = False
            continue

    toc = cv2.getTickCount()
    isLocked_server = isLocked and large_enough(
        rect[2] - rect[0], rect[3] - rect[1], im_x, im_y) and in_locking_rect(
            rect, im_x, im_y)
    #isLocked_server = isLocked

    # Display frame
    cv2.putText(im, "FPS: %.2f" % (1 / ((toc - tic) / cv2.getTickFrequency())),
                (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
    cv2.putText(im, "Frame: {}".format(cnt), (900, 40), cv2.FONT_HERSHEY_PLAIN,
                3, (200, 0, 50), 2)

    isLocked_server, states = fsm(isLocked_server, states)
    print("states: ", states)

    if isLocked:
        cv2.rectangle(im, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255),
                      3)
        cv2.putText(im, "Locked", (450, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 0, 255), 3)
        print("area: ", (rect[2] - rect[0]) * (rect[3] - rect[1]))
    if not isLocked:
        cv2.putText(im, "NOTLocked", (450, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                    (5, 110, 5), 1)
    # if isLocked_server:
    if isLocked_server:
        if locking_timestamps == [0, 0]:
            locking_timestamps[0] = duration_tic
            print("Locking started at: ", locking_timestamps[0])
        else:
            cv2.putText(
                im, "Locking time: {:.2f}".format(duration_tic -
                                                  locking_timestamps[0]),
                (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        cv2.putText(im, "Locked_for_server", (450, 80), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 255), 3)
        cv2.rectangle(im, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255),
                      3)
        print("Locked and the bounding box is large enough")
        print("Hedef_merkez_X: {}".format((rect[0] + rect[2]) / 2))
        print("Hedef_merkez_Y: {}".format((rect[1] + rect[3]) / 2))
        print("Hedef_genislik: {}".format(rect[2] - rect[0]))
        print("Hedef_yukseklik: {}".format(rect[3] - rect[1]))
    # if not isLocked_server:
    if not isLocked_server:
        if not locking_timestamps == [0, 0]:
            locking_timestamps[1] = duration_tic
            """
            print("Locking finished")
            print("Locking_duration: {}".format(locking_timestamps[1]-locking_timestamps[0]))
            """
            if locking_timestamps[1] - locking_timestamps[0] > 5:
                print("Locking finished at {}".format(duration_tic) +
                      " with duraiton more than 5 seconds")
                print("Locking_duration: {}".format(locking_timestamps[1] -
                                                    locking_timestamps[0]))
            else:
                # this wont be published to the server, just for the user to see
                print("Locking finished at {}".format(duration_tic) +
                      " with duraiton less than 5 seconds")
                print("Locking_duration: {}".format(locking_timestamps[1] -
                                                    locking_timestamps[0]))
            locking_timestamps = [0, 0]

    # exp
    cv2.putText(im, "states: {}".format(states), (10, 180),
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)

    # Locking rectangle
    cv2.rectangle(im, (im_x // 4, im_y // 10), (im_x * 3 // 4, im_y * 9 // 10),(130, 0, 75), 3)
	#clock
    cv2.putText(im, "Clock: {:.2f}".format(duration_tic - start_time),
                (10, 140), cv2.FONT_HERSHEY_PLAIN, 2, (150, 250, 250), 2)
    
	# resize for display 
	frame = cv2.resize(im, (1920, 1080)) #I think we can change the output size to be one of the first parameters to put in 
	cv2.imshow("Frame", frame)
    # write to video
    # out.write(frame)

    print("Frame: {}".format(cnt))
    print("------------")

    if cv2.waitKey(1) & 0xFF == 27:
        break
    if cnt >= 4000:
        cap.release()
print("Done")
