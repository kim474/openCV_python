import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import UdpComms as U


id1 = 0
id2 = 0
yoloClass_ids = []
yoloClass_id = []
yoloClass_idss = 0
cvClass_id = []
name = 1
i = 0
j = 0
k = 0
yol = 0

cam = cv2.VideoCapture(0)
cam.set(3, 1280)
cam.set(4, 720)

"""
while True:
    ret, frame = cam.read()

    if(ret):
        if cv2.waitKey(1) != 1:
            cv2.imwrite("../test_videocapture.jpg", frame)
            break
cam.release()
"""
net = cv2.dnn.readNet("../yolov4-obj.weights", "../yolov4-obj.cfg")
classes = []
with open("../obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

"""
img = cv2.imread("../test_bar1.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape"""

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

            if confidence > 0.5:
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
            elif 0.5 > confidence > 0.48:
                print("detection fail")
                cv2.imwrite("../test_capture" + str(name) + ".jpg", frame)
                name += 1

            """
            else:
                cv2.imwrite("../test_capture" + str(name) + ".jpg", frame)
                time.sleep(5)
                name += 1"""
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(0, len(class_ids)):
        yoloClass_ids.append(int(class_ids[0]))
        if yoloClass_ids[j-1] != yoloClass_ids[j]:
            #print(yoloClass_ids[j-1])
            yoloClass_id.append(yoloClass_ids[j])
            print(yoloClass_id)


            for k in range(0, len(yoloClass_id)):
                yol = len(yoloClass_id)
                k += 1
            #print(yoloClass_id[yol-1])
            yoloClass_idss = yoloClass_id[yol-1]
            print(yoloClass_idss)

            sock = U.UdpComms(udpIP="192.168.35.57", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
            sock.SendData('Sent from Python: ' + str(yoloClass_idss))
            time.sleep(i + 1)

        j += 1

    """
    for i in range(0, len(class_ids)):
        if i == 0:
            yoloClass_id = class_id
        else:
            id1 = int(class_ids[i - 1])
            id2 = int(class_ids[i])
            if id1 != id2:
                yoloClass_id = int(class_ids[i])
                # print(yoloClass_id)
            i += 1"""

    """
    for i in range(0, len(class_ids)):
        if i == 0:
            yoloClass_id = int(class_ids[0])
            print(yoloClass_id)
        else:
            id1 = int(class_ids[i-1])
            id2 = int(class_ids[i])
            if id1 != id2:
                yoloClass_id = int(class_ids[i])
                #print(yoloClass_id)
            i += 1"""

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

    """
    sock = U.UdpComms(udpIP="192.168.35.57", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)

    sock.SendData('Sent from Python: ' + str(yoloClass_id))
    time.sleep(1)"""

cam.release()

"""
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
"""

""" original compareHist
query = hists[0]
methods = {'CORREL': cv2.HISTCMP_CORREL, 'CHISQR': cv2.HISTCMP_CHISQR,
           'INTERSECT': cv2.HISTCMP_INTERSECT,
           'BHATTACHARYYA': cv2.HISTCMP_BHATTACHARYYA}
#Correlation, Chi-Square, Intersection, BHattacharyya Distance
for j, (name, flag) in enumerate(methods.items()):
    print('%-10s'%name, end='\t')
    for i, (hist, img) in enumerate(zip(hists, imgs)):
        ret: int
        ret = cv2.compareHist(query, hist, flag)
        if flag == cv2.HISTCMP_INTERSECT:
            ret = ret/np.sum(query)
        print("img%d:%7.2f"% (i+1, ret), end='\t')
        if ((1.1 > ret > 0.96) | (ret < 0.1)):
            print("images are same.")
        else:
            print("images are different.")
    print()
plt.show()
"""

""" grab-cut
mask = np.zeros(img1.shape[:2], np.uint8)

bgdModel = fgdModel = np.zeros((1, 65), np.float64)

rect = (50, 50, 1400, 1400)

cv2.grabCut(img1, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
grabImg = img1 * mask2[:, :, np.newaxis]

plt.imshow(grabImg), plt.colorbar(), plt.show()
"""

""" 
blur = cv2.GaussianBlur(img1, (5, 5), 0)
hist = cv2.calcHist([blur], [0], None, [256], [0, 256])

cv2.imshow('blurImg', blur)
cv2.waitKey(1)

plt.plot(hist)
plt.show()
"""

