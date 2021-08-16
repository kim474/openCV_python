import cv2
import numpy as np
import time
import UdpComms as U
import socket
import sys

ids = []
idss = []
ids.append(0)
c = 0
i = 0
j = 0
k = 0
send_id = 0

HOST = ''
PORT = 8888
sendData = ""
recved = ""

# 1. open Socket
# address family: IPv4, socket type: TCP
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

# 2. bind to a address and port
try:
    s.bind((HOST, PORT))
except socket.error as msg:
    print('Bind Failed. Error code: ' + str(msg[0]) + ' Message: ' + msg[1])
    sys.exit()
print('Socket bind complete')

# 3. Listen for incoming connections
s.listen(10)
print('Socket now listening')

# 4. Accept connection
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
    sock = U.UdpComms(udpIP="192.168.35.62", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)
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

                # print(k, '번째: ', class_id)
                k += 1
                ids.append(class_id)
                # print(ids)
                if ids[k-1] != ids[k]:
                    idss.append(ids[k])
                    # print('idss', idss)
                    # yolo에서 검출한 class_id가 1이라면, 라즈베리로 신호 전달
                    if idss[j] == 1:
                        # 5. send "1"
                        sendData = "1"
                        conn.sendall(sendData.encode())
                        print('Send to Raspberry Pi---')

                        # 6. receive class_id
                        recvData = conn.recv(1024)
                        # 빈 문자열을 수신하면,
                        if not recvData:
                            print('disconnect')
                            # break
                        else:
                            send_id = int(recvData.decode())
                            if send_id == 20:  # plastic_bottle class_id = 20
                                print('Received & Send to Unity: ', send_id)
                                sock.SendData(str(send_id))
                                time.sleep(i + 1)
                            elif send_id == 21:  # glass_bottle class_id = 21
                                print('Received & Send to Unity: ', send_id)
                                sock.SendData(str(send_id))
                                time.sleep(i + 1)
                    else:
                        if (idss[j] != 0) and (idss[j] != 1):
                            print('Send to Unity: ', idss[j])
                            sock.SendData(str(idss[j]))
                            time.sleep(i + 1)
                    j += 1

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
