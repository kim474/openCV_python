import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import UdpComms as U
import random

yoloClass_ids = []
yoloClass_id = []
yoloClass_idss = 0
cvClass_id = 0
name = 1
i = 0
j = 0
k = 0
yol = 0

cam = cv2.VideoCapture(2)
cam.set(3, 480)
cam.set(4, 480)

net = cv2.dnn.readNet("../yolov4-obj.weights", "../yolov4-obj.cfg")
classes = []
with open("../obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

while True:
    ret, frame = cam.read()
    h, w, c = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 정보를 화면에 표시
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.8:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # 좌표
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                if class_id == 10:
                    cv2.imwrite("../test_capture" + str(name) + ".jpg", frame)

                    capImg = cv2.imread("../test_capture" + str(name) + ".jpg")
                    crop = capImg[45:435, 0:480].copy()
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    cv2.imwrite("../test_capture" + str(name) + ".jpg", blur)

                    # glass plate test
                    glassImg = cv2.imread("../glass_plate1.jpg")
                    test3 = cv2.imread("../test_capture" + str(name) + ".jpg")
                    name += 1

                    glassTest = [glassImg, test3]
                    gHists = []
                    gCount = 0
                    i = 0

                    for i, gImg in enumerate(glassTest):
                        gHist = cv2.calcHist([gImg], [0], None, [256], [0, 256])

                        cv2.normalize(gHist, gHist, 0, 1, cv2.NORM_MINMAX)
                        gHists.append(gHist)

                    gHist1 = gHists[0]
                    gHist2 = gHists[1]

                    gResult1 = cv2.compareHist(gHist1, gHist2, cv2.HISTCMP_CORREL)
                    if gResult1 > 0.99:
                        gCount += 1

                    gResult2 = cv2.compareHist(gHist1, gHist2, cv2.HISTCMP_CHISQR)
                    if gResult2 < 3:
                        gCount += 1

                    gResult3 = cv2.compareHist(gHist1, gHist2, cv2.HISTCMP_INTERSECT)
                    gResult3 = gResult3 / np.sum(gHist1)
                    if gResult3 >= 0.79:
                        gCount += 1

                    gResult4 = cv2.compareHist(gHist1, gHist2, cv2.HISTCMP_BHATTACHARYYA)
                    if gResult4 < 0.26:
                        gCount += 1

                    if gCount >= 4:
                        if yoloClass_idss != 18:
                            yoloClass_idss = 18
                            print(yoloClass_idss)
                    else:
                        # paper plate test
                        name -= 1
                        paperImg = cv2.imread("../paper_plate1.jpg")
                        test4 = cv2.imread("../test_capture" + str(name) + ".jpg")
                        name += 1

                        paperTest = [paperImg, test4]
                        pHists = []
                        pCount = 0
                        i = 0

                        for i, pImg in enumerate(paperTest):
                            pHist = cv2.calcHist([pImg], [0], None, [256], [0, 256])

                            cv2.normalize(pHist, pHist, 0, 1, cv2.NORM_MINMAX)
                            pHists.append(pHist)

                        pHist1 = pHists[0]
                        pHist2 = pHists[1]

                        pResult1 = cv2.compareHist(pHist1, pHist2, cv2.HISTCMP_CORREL)
                        if pResult1 > 0.99:
                            pCount += 1

                        pResult2 = cv2.compareHist(pHist1, pHist2, cv2.HISTCMP_CHISQR)
                        if pResult2 < 0.3:
                            pCount += 1

                        pResult3 = cv2.compareHist(pHist1, pHist2, cv2.HISTCMP_INTERSECT)
                        pResult3 = pResult3 / np.sum(pHist1)
                        if pResult3 >= 0.9:
                            pCount += 1

                        pResult4 = cv2.compareHist(pHist1, pHist2, cv2.HISTCMP_BHATTACHARYYA)
                        if pResult4 < 0.16:
                            pCount += 1

                        if pCount >= 4:
                            if yoloClass_idss != 19:
                                yoloClass_idss = 19
                                print(yoloClass_idss)
                        else:
                            # plastic plate test
                            name -= 1
                            plasticImg = cv2.imread("../plastic_plate1.jpg")
                            test5 = cv2.imread("../test_capture" + str(name) + ".jpg")
                            name += 1

                            plasticTest = [plasticImg, test5]
                            ppHists = []
                            ppCount = 0
                            i = 0

                            for i, ppImg in enumerate(plasticTest):
                                ppHist = cv2.calcHist([ppImg], [0], None, [256], [0, 256])

                                cv2.normalize(ppHist, ppHist, 0, 1, cv2.NORM_MINMAX)
                                ppHists.append(ppHist)

                            ppHist1 = ppHists[0]
                            ppHist2 = ppHists[1]

                            ppResult1 = cv2.compareHist(ppHist1, ppHist2, cv2.HISTCMP_CORREL)
                            if ppResult1 > 0.98:
                                ppCount += 1

                            ppResult2 = cv2.compareHist(ppHist1, ppHist2, cv2.HISTCMP_CHISQR)
                            if (1 < ppResult2) & (ppResult2 < 3.5):
                                ppCount += 1

                            ppResult3 = cv2.compareHist(ppHist1, ppHist2, cv2.HISTCMP_INTERSECT)
                            ppResult3 = ppResult3 / np.sum(ppHist1)
                            if ppResult3 > 0.83:
                                ppCount += 1

                            ppResult4 = cv2.compareHist(ppHist1, ppHist2, cv2.HISTCMP_BHATTACHARYYA)
                            if ppResult4 < 0.24:
                                ppCount += 1

                            if ppCount >= 3:
                                if yoloClass_idss != 20:
                                    yoloClass_idss = 20
                                    print(yoloClass_idss)
                            else:
                                #print("NO MATCHES WERE FOUND")
                                yoloClass_idss = random.randint(1, 26)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(0, len(class_ids)):
        if len(class_ids) != 0:
            yoloClass_ids.append(int(class_ids[0]))
            if len(yoloClass_ids) == 1:
                yoloClass_ids[0] = class_ids[0]
                yoloClass_idss = yoloClass_ids[0]
                print(yoloClass_idss)
            else:
                if yoloClass_ids[j - 1] != yoloClass_ids[j]:
                    # print(yoloClass_ids[j-1])
                    yoloClass_id.append(yoloClass_ids[j])
                    # print(yoloClass_id)

                    for k in range(0, len(yoloClass_id)):
                        yol = len(yoloClass_id)
                        k += 1
                    # print(yoloClass_id[yol-1])
                    yoloClass_idss = yoloClass_id[yol - 1]
                    if yoloClass_idss != 10:
                        print(yoloClass_idss)
                    """
                    sock = U.UdpComms(udpIP="", portTX=8000, portRX=8001, enableRX=True,
                                      suppressWarnings=True)
                    sock.SendData(str(yoloClass_idss))
                    time.sleep(i + 1)
                    """
            j += 1

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
    cv2.imshow("YOLO_Image", frame)

    if cv2.waitKey(100) > 0:
        break

cam.release()
cv2.destroyAllWindows()
