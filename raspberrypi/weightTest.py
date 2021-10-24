import socket
import time

EMULATE_HX711 = False

referenceUnit = -447
valList = []
sendList = []
i = 0
j = 0
k = 0
c = 0
l = 0
w = 0
send = ""
recved = ""
weightList = []

if not EMULATE_HX711:
    import RPi.GPIO as GPIO
    from hx711 import HX711
else:
    from emulated_hx711 import HX711


def cleanAndExit():
    print("Cleaning...")

    if not EMULATE_HX711:
        GPIO.cleanup()

    print("Bye!")
    sys.exit()


hx = HX711(5, 6)

hx.set_reading_format("MSB", "MSB")

hx.set_reference_unit(referenceUnit)

hx.reset()

hx.tare()

print("Tare done! Add weight now...")

HOST = '192.168.0.101'  # all available interfaces1
PORT = 8888

# 1. open Socket
conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

conn.connect((HOST, PORT))

# keep talking with the client
while 1:
    val = max(0, int(hx.get_weight(5)))
    valList.append(val)

    if i == 0:
        j = valList[i]
        print('start: ', j, 'gram')
        j = 0
        i += 1
    else:
        if valList[i - 1] != valList[i]:
            j = valList[i]
        i += 1
    if j != 0:
        # if j-1 <= valList[i-3] <= j+1 and j == valList[i-1] == valList[i-2] == valList[i-3]:
        if j - 1 <= valList[i - 3] <= j + 1 and l != j + 1 and l != j - 1 and j == valList[i - 1] == valList[i - 2] == \
                valList[i - 3]:
            l = j
            j = valList[i - 1]
            print(j, 'gram')
            weightList.append(j)
            w += 1
            c = 1

    hx.power_down()
    hx.power_up()
    time.sleep(1)

    if c == 1:
        # 5. receive & send data
        recvData = conn.recv(1024)
        recved = int(recvData.decode())
        if recved == 11:
            print('Received: ', recved)
            if weightList[w - 1] < 60:
                sendList.append("20")
                j = 0
            else:
                sendList.append("21")
                j = 0
            k += 1
            conn.sendall(sendList[k - 1].encode())
            print('Send: ', sendList[k - 1])
            c = 0
        elif recved == 12:
            print('Received: ', recved)
            if weightList[w - 1] < 100:
                sendList.append("20")
                j = 0
            else:
                sendList.append("21")
                j = 0
            k += 1
            conn.sendall(sendList[k - 1].encode())
            print('Send: ', sendList[k - 1])
            c = 0
        elif recved == 13:
            print('Received: ', recved)
            if weightList[w - 1] < 150:
                sendList.append("20")
                j = 0
            else:
                sendList.append("21")
                j = 0
            k += 1
            conn.sendall(sendList[k - 1].encode())
            print('Send: ', sendList[k - 1])
            c = 0

conn.close()
s.close()