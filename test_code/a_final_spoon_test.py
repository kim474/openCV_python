import cv2
import numpy as np
import time
import UdpComms as U

yoloClass_ids = []
yoloClass_id = []
yoloClass_idss = 32
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
    sock = U.UdpComms(udpIP="192.168.99.110", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
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
                # Object detected
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

                # class_ids[2]가 12일때 start image processing; 너무 여러번 진행하지 않도록 하기 위해 [2]로 진행
                if len(class_ids) > 2 and class_ids[2] == 12:
                    crop = frame[45:435, 0:480].copy()
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

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

    # 검출한 class_id가 12면 재질판단 시작
    if len(class_ids) != 0 and class_ids[0] == 12:
        # start metal spoon test
        metalImg = cv2.imread("../w1_l6.jpg")
        mTest = blur

        metalTest = [metalImg, mTest]
        mHists = []
        mCount = 0
        i = 0

        for i, mImg in enumerate(metalTest):
            mHist = cv2.calcHist([mImg], [0], None, [256], [0, 256])

            cv2.normalize(mHist, mHist, 0, 1, cv2.NORM_MINMAX)
            mHists.append(mHist)

        mHist1 = mHists[0]
        mHist2 = mHists[1]

        mResult1 = cv2.compareHist(mHist1, mHist2, cv2.HISTCMP_CORREL)
        if 0.85 < mResult1 < 0.93:
            mCount += 1

        mResult2 = cv2.compareHist(mHist1, mHist2, cv2.HISTCMP_CHISQR)
        if 20 < mResult2 < 24:
            mCount += 1

        mResult3 = cv2.compareHist(mHist1, mHist2, cv2.HISTCMP_INTERSECT)
        mResult3 = mResult3 / np.sum(mHist1)
        if 0.9 < mResult3 < 0.94:
            mCount += 1

        mResult4 = cv2.compareHist(mHist1, mHist2, cv2.HISTCMP_BHATTACHARYYA)
        if 0.1 < mResult4 < 0.15:
            mCount += 1
        """
        print('1번', mResult1)
        print('2번', mResult2)
        print('3번', mResult3)
        print('4번', mResult4)

        print('mCount: ', mCount)
        """
        if mCount >= 3:
            # metal spoon이 맞다면, class_id로 30 전송
            if yoloClass_idss != 30:
                yoloClass_idss = 30
                # print("1여기")
                print('metal spoon', yoloClass_idss)
                sock.SendData(str(yoloClass_idss))
                time.sleep(i + 1)
        else:
            # start plastic spoon test
            plasticImg = cv2.imread("../p1_l6.jpg")
            pTest = blur

            plasticTest = [plasticImg, pTest]
            pHists = []
            pCount = 0
            i = 0

            for i, pImg in enumerate(plasticTest):
                pHist = cv2.calcHist([pImg], [0], None, [256], [0, 256])

                cv2.normalize(pHist, pHist, 0, 1, cv2.NORM_MINMAX)
                pHists.append(pHist)

            pHist1 = pHists[0]
            pHist2 = pHists[1]

            pResult1 = cv2.compareHist(pHist1, pHist2, cv2.HISTCMP_CORREL)

            if 0.73 < pResult1 < 0.76:
                pCount += 1

            pResult2 = cv2.compareHist(pHist1, pHist2, cv2.HISTCMP_CHISQR)

            if 15 < pResult2 < 30:
                pCount += 1

            pResult3 = cv2.compareHist(pHist1, pHist2, cv2.HISTCMP_INTERSECT)
            pResult3 = pResult3 / np.sum(pHist1)

            if 0.58 < pResult3 < 0.63:
                pCount += 1

            pResult4 = cv2.compareHist(pHist1, pHist2, cv2.HISTCMP_BHATTACHARYYA)

            if 0.20 < pResult4 < 0.23:
                pCount += 1
            """
            print('1번', pResult1)
            print('2번', pResult2)
            print('3번', pResult3)
            print('4번', pResult4)

            print('pCount: ', pCount)
            """
            if pCount >= 3:
                # plastic spoon이 맞다면, class_id로 31 전송
                if yoloClass_idss != 31:
                    yoloClass_idss = 31
                    # print("2여기")
                    print('plastic spoon', yoloClass_idss)
                    sock.SendData(str(yoloClass_idss))
                    time.sleep(i + 1)
            else:
                # start wood spoon test
                woodImg = cv2.imread("../w1_l6.jpg")
                wTest = blur

                woodTest = [woodImg, wTest]
                wHists = []
                wCount = 0
                i = 0

                for i, wImg in enumerate(woodTest):
                    wHist = cv2.calcHist([wImg], [0], None, [256], [0, 256])

                    cv2.normalize(wHist, wHist, 0, 1, cv2.NORM_MINMAX)
                    wHists.append(wHist)

                wHist1 = wHists[0]
                wHist2 = wHists[1]

                wResult1 = cv2.compareHist(wHist1, wHist2, cv2.HISTCMP_CORREL)

                if 0.82 < wResult1 < 0.85:
                    wCount += 1

                wResult2 = cv2.compareHist(wHist1, wHist2, cv2.HISTCMP_CHISQR)

                if 27 < wResult2 < 30:
                    wCount += 1

                wResult3 = cv2.compareHist(wHist1, wHist2, cv2.HISTCMP_INTERSECT)
                wResult3 = wResult3 / np.sum(wHist1)

                if 0.9 <= wResult3 < 0.93:
                    wCount += 1

                wResult4 = cv2.compareHist(wHist1, wHist2, cv2.HISTCMP_BHATTACHARYYA)

                if 0.15 <= wResult4 < 0.16:
                    wCount += 1
                """
                print('1번', wResult1)
                print('2번', wResult2)
                print('3번', wResult3)
                print('4번', wResult4)

                print('wCount: ', wCount)
                """
                if wCount >= 3:
                    # wood spoon이 맞다면, class_id로 32 전송
                    if yoloClass_idss != 32:
                        yoloClass_idss = 32
                        # print("2여기")
                        print('wood spoon', yoloClass_idss)
                        sock.SendData(str(yoloClass_idss))
                        time.sleep(i + 1)
                else:
                    # 모두 아니라면, class_id로 12을 넘겨줌
                    if yoloClass_idss != 0:
                        if yoloClass_idss != 12:
                            if yoloClass_idss != 30 and yoloClass_idss != 31 and yoloClass_idss != 32:
                                yoloClass_idss = 12
                                print('??', yoloClass_idss)
                                sock.SendData(str(yoloClass_idss))
                                time.sleep(i + 1)

    # class_id = 12일 때, 중복 detection 제거
    for i in range(0, len(class_ids)):
        if len(class_ids) != 0:
            yoloClass_ids.append(int(class_ids[0]))
            if len(yoloClass_ids) == 1:
                yoloClass_ids[0] = class_ids[0]
                yoloClass_idss = yoloClass_ids[0]
                if yoloClass_idss != 12:
                    print(yoloClass_idss)
                    sock.SendData(str(yoloClass_idss))
                    time.sleep(i + 1)
            else:
                if yoloClass_ids[j - 1] != yoloClass_ids[j]:

                    for k in range(0, len(yoloClass_id)):
                        yol = len(yoloClass_id)
                        k += 1
                    yoloClass_idss = yoloClass_id[yol - 1]
                    # 검출한 class_id가 12가 아니라면, 그냥 class_id를 보냄
                    if yoloClass_idss != 12:
                        print("4여기")
                        print(yoloClass_idss)
                        sock.SendData(str(yoloClass_idss))
                        time.sleep(i + 1)
            j += 1

    if cv2.waitKey(1) > 0:
        break

cam.release()
cv2.destroyAllWindows()
