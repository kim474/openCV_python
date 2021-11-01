import cv2
import numpy as np
import time
import UdpComms as U

yoloClass_ids = []
yoloClass_id = []
yoloClass_idss = 0
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
    sock = U.UdpComms(udpIP="192.168.0.12", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
    ret, frame = cam.read()
    h, w, c = frame.shape

    # detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # showing info
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.8:
                # object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # coordinates of boxes
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                # class_ids[2]가 10일때 start image processing; 너무 여러번 진행하지 않도록 하기 위해 [2]로 진행
                if len(class_ids) > 2 and class_ids[2] == 10:
                    crop = frame[45:435, 0:480].copy()
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)

                    # 검출한 class_id가 10이라면, 재질 판단 시작 / class_ids[0] -> [2]로 변경 test 해보기
                    if len(class_ids) != 0 and class_ids[0] == 10:
                        # start glass plate test
                        glassImg = cv2.imread("../glass_plate1.jpg")
                        test3 = blur

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
                        if gResult1 > 0.95:
                            gCount += 1

                        gResult2 = cv2.compareHist(gHist1, gHist2, cv2.HISTCMP_CHISQR)
                        if 200 > gResult2 > 100:
                            gCount += 1

                        gResult3 = cv2.compareHist(gHist1, gHist2, cv2.HISTCMP_INTERSECT)
                        gResult3 = gResult3 / np.sum(gHist1)
                        if 0.97 > gResult3 > 0.80:
                            gCount += 1

                        gResult4 = cv2.compareHist(gHist1, gHist2, cv2.HISTCMP_BHATTACHARYYA)
                        if 0.38 < gResult4 <= 0.45:
                            gCount += 1
                        """
                        print(gResult1)
                        print(gResult2)
                        print(gResult3)
                        print(gResult4)
                        """
                        if gCount >= 3:
                            # glass plate가 맞다면, class_id로 18 전송
                            if yoloClass_idss != 18:
                                yoloClass_idss = 18
                                # ("1여기")
                                print(yoloClass_idss)
                                sock.SendData(str(yoloClass_idss))
                                # print("send")
                                time.sleep(i + 1)
                                blur = None
                        else:
                            # start plastic plate test
                            plasticImg = cv2.imread("../plastic_plate1.jpg")
                            test5 = blur

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

                            if 0.60 < ppResult1 < 0.75:
                                ppCount += 1

                            ppResult2 = cv2.compareHist(ppHist1, ppHist2, cv2.HISTCMP_CHISQR)

                            if 2000 < ppResult2 < 6000:
                                ppCount += 1

                            ppResult3 = cv2.compareHist(ppHist1, ppHist2, cv2.HISTCMP_INTERSECT)
                            ppResult3 = ppResult3 / np.sum(ppHist1)

                            if 1.0 > ppResult3 > 0.97:
                                ppCount += 1

                            ppResult4 = cv2.compareHist(ppHist1, ppHist2, cv2.HISTCMP_BHATTACHARYYA)

                            if 0.45 <ppResult4 < 0.53:
                                ppCount += 1
                            """
                            print(ppResult1)
                            print(ppResult2)
                            print(ppResult3)
                            print(ppResult4)
                            """
                            if ppCount >= 4:
                                # plastic plate가 맞다면, class_id로 19 전송
                                if yoloClass_idss != 19:
                                    yoloClass_idss = 19
                                    # print("2여기")
                                    print(yoloClass_idss)
                                    sock.SendData(str(yoloClass_idss))
                                    time.sleep(i + 1)
                                    blur = None
                            else:
                                # glass, plastic 모두 아니라면, class_id로 10을 넘겨줌
                                if yoloClass_idss != 10 and yoloClass_idss != 18 and yoloClass_idss != 19:
                                    # yoloClass_idss = 10
                                    # print("3여기")
                                    blur = None

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # class_id = 10일 때 중복 detection 제거
    for i in range(0, len(class_ids)):
        if len(class_ids) != 0:
            yoloClass_ids.append(int(class_ids[0]))
            if len(yoloClass_ids) == 1:
                yoloClass_ids[0] = class_ids[0]
                yoloClass_idss = yoloClass_ids[0]
                if yoloClass_idss != 10:
                    print(yoloClass_idss)
                    sock.SendData(str(yoloClass_idss))
                    time.sleep(i + 1)
            else:
                if yoloClass_ids[j - 1] != yoloClass_ids[j]:
                    yoloClass_id.append(yoloClass_ids[j])
                    for k in range(0, len(yoloClass_id)):
                        yol = len(yoloClass_id)
                        k += 1
                    yoloClass_idss = yoloClass_id[yol - 1]
                    # 검출한 class_id가 10이 아니라면, 그냥 class_id를 보냄
                    if yoloClass_idss != 10:
                        # print("4여기")
                        print(yoloClass_idss)
                        sock.SendData(str(yoloClass_idss))
                        time.sleep(i + 1)
            j += 1

    # bounding box 중복 제거
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

    cv2.imshow("YOLO_Detection", frame)

    if cv2.waitKey(100) > 0:
        break

cam.release()
cv2.destroyAllWindows()
