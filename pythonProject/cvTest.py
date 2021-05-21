
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

cam = cv2.VideoCapture(0)
cam.set(3, 720)
cam.set(4, 720)

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
            elif 0.8 > confidence > 0.78:
                print("detection fail, start openCV")
                cv2.imwrite("../test_capture" + str(name) + ".jpg", frame)

                img1 = cv2.imread("../test_r3.jpg")
                img2 = cv2.imread("../test_capture" + str(name) + ".jpg")
                #img2 = cv2.imread("../test_capture" + str(name) + ".jpg")
                name += 1

                imgs = [img1, img2]
                hists = []
                for i, img in enumerate(imgs):
                    plt.subplot(1, len(imgs), i + 1)
                    plt.title('img%d' % (i + 1))
                    plt.axis('off')
                    #plt.imshow(img[:, :, ::-1])

                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(img, (5, 5), 0)

                    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
                    # images: input, channel: input image의 특정 channel 지정 가능, mask: 이미지 분석 영역(none이면 전체),
                    # histSize: Bins value, ranges: range value

                    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                    hists.append(hist)

                hist1 = hists[0]
                hist2 = hists[1]
                count: int = 0

                result1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                #print('correl: ', result1)
                if result1 > 0.99:
                    count += 1
                    #print('MATCH CORREL----------------------')

                result2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
                #print('chisqr: ', result2)
                if result2 < 2:
                    count += 1
                    #print('MATCH CHI-SQUARE------------------')

                result3 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
                result3 = result3 / np.sum(hist1)
                #print('intersect: ', result3)
                if result3 > 0.71:
                    count += 1
                    #print('MATCH INTERSECTION----------------')

                result4 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
                #print('bhattacharyya: ', result4)
                if result4 < 0.3:
                    count += 1
                    #print('MATCH BHATTACHARYYA DISTANCE------')

                if count >= 3:
                    print("!!!this is PET!!!")
                    yoloClass_idss = 0
                    sock = U.UdpComms(udpIP="192.168.1.102", portTX=8000, portRX=8001, enableRX=True,
                                      suppressWarnings=True)
                    sock.SendData('Sent from Python CV: ' + str(yoloClass_idss))
                    time.sleep(i + 1)
                else:
                    print("NO RESULT")
                    yoloClass_idss = random.randint(0, 26)
                    sock = U.UdpComms(udpIP="192.168.1.102", portTX=8000, portRX=8001, enableRX=True,
                                      suppressWarnings=True)
                    sock.SendData('Sent from Python CV fail: ' + str(yoloClass_idss))
                    time.sleep(i + 1)

                #plt.show()

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(0, len(class_ids)):
        yoloClass_ids.append(int(class_ids[0]))
        if yoloClass_ids[j-1] != yoloClass_ids[j]:
            #print(yoloClass_ids[j-1])
            yoloClass_id.append(yoloClass_ids[j])
            #print(yoloClass_id)

            for k in range(0, len(yoloClass_id)):
                yol = len(yoloClass_id)
                k += 1
            #print(yoloClass_id[yol-1])
            yoloClass_idss = yoloClass_id[yol-1]
            print(yoloClass_idss)

            sock = U.UdpComms(udpIP="192.168.1.102", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
            sock.SendData('Sent from Python: ' + str(yoloClass_idss))
            time.sleep(i + 1)

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


img1 = cv2.imread("../test_rb2.jpg")
img2 = cv2.imread("../test_bar1.jpg")
#img3 = cv2.imread("../test_capture." + str(name) + "jpg")

imgs = [img1, img2]
hists = []
for i, img in enumerate(imgs):
    plt.subplot(1, len(imgs), i+1)
    plt.title('img%d' % (i+1))
    plt.axis('off')
    plt.imshow(img[:, :, ::-1])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    #images: input, channel: input image의 특정 channel 지정 가능, mask: 이미지 분석 영역(none이면 전체),
    #histSize: Bins value, ranges: range value

    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    hists.append(hist)

hist1 = hists[0]
hist2 = hists[1]
count: int = 0

result1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
print('correl: ', result1)
if result1 > 0.91:
    count += 1
    print('MATCH CORREL----------------------')

result2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
print('chisqr: ', result2)
if result2 < 1:
    count += 1
    print('MATCH CHI-SQUARE------------------')

result3 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
result3 = result3/np.sum(hist1)
print('intersect: ', result3)
if result3 > 0.96:
    count += 1
    print('MATCH INTERSECTION----------------')

result4 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
print('bhattacharyya: ', result4)
if result4 < 0.1:
    count += 1
    print('MATCH BHATTACHARYYA DISTANCE------')

if count >= 3:
    print("!!!this is PET!!!")
    cvClass_id = "PET"
else:
    print("NO RESULT")

plt.show()
