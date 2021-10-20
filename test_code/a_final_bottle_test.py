import cv2
import numpy as np
import time
import UdpComms as U
import socket
import sys
from timeit import default_timer as timer
from datetime import timedelta

ids = []
idss = []
ids.append(0)
c = 0
i = 0
j = 0
k = 0
send_id = 0
start = 0.0
end = 0.0

HOST = ''
PORT = 8888
sendData = ""
recved = ""

# open Socket
# address family: IPv4, socket type: TCP
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

# bind to a address and port
try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print('Bind Failed. Error code: ' + str(msg[0]) + ' Message: ' + msg[1])
    sys.exit()
print('Socket bind complete')

# Listen for incoming connections
s.listen(10)
print('Socket now listening')

# Accept connection
conn, addr = s.accept()
print('Connected with ' + addr[0] + ':' + str(addr[1]))

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
    sock = U.UdpComms(udpIP="192.168.0.107", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
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

            end = timer()
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

                k += 1
                ids.append(class_id)

                # 중복 제거 위해서 공백 time 사용
                if t2 >= timedelta(seconds=7):
                    # 검출한 class_id가 1이라면, 라즈베리로 신호 전달
                    if ids[k] == 1:
                        # print("height: ", dh)
                        # object height 기준 bottle 크기 별로 11, 12, 13 전달
                        if dh < 240:
                            sendData = "11"
                        elif 240 < dh < 320:
                            sendData = "12"
                        elif 320 < dh:
                            sendData = "13"
                        conn.sendall(sendData.encode())
                        print('Send to Raspberry Pi---', sendData)

                        # receive class_id
                        recvData = conn.recv(1024)
                        # 빈 문자열을 수신하면,
                        if not recvData:
                            print('disconnect')
                            # break
                        else:
                            send_id = int(recvData.decode())
                            # plastic_bottle class_id = 20
                            if send_id == 20:
                                print('Received & Send to Unity: ', send_id)
                                sock.SendData(str(send_id))
                                time.sleep(i + 1)
                            # glass_bottle class_id = 21
                            elif send_id == 21:
                                print('Received & Send to Unity: ', send_id)
                                sock.SendData(str(send_id))
                                time.sleep(i + 1)
                    else:
                        if (ids[k] != 0) and (ids[k] != 1):
                            print('Send to Unity: ', ids[k])
                            sock.SendData(str(ids[k]))
                            time.sleep(i + 1)
                start = timer()
            t2 = timedelta(seconds=end - start)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

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
