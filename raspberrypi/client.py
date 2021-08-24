import socket
import time

EMULATE_HX711=False

referenceUnit = -447
valList = []
sendList = []
i = 0
j = 0
k = 0
c = 0
l = 0
send = ""
recved = ""

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

HOST = '192.168.0.104' # all available interfaces
PORT = 8888

# 1. open Socket
conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

conn.connect((HOST, PORT))

# keep talking with the client
while 1:
    val = max(0, int(hx.get_weight(5)))
    valList.append(val)
    #print(valList)
    
    if i == 0:
        j = valList[i]
        print('start: ', j, 'gram')
        j = 0
        i += 1
    else:
        if valList[i-1] != valList[i]:
            j = valList[i]
            #print('j: ', j)
        i += 1
    if j != 0:
        #if j-1 <= valList[i-3] <= j+1 and j == valList[i-1] == valList[i-2] == valList[i-3]:
         if j-1 <= valList[i-3] <= j+1 and l != j+1 and l != j-1 and j == valList[i-1] == valList[i-2] == valList[i-3]:
            l = j
            #print('l: ', l)
            j = valList[i-1]
            print(j, 'gram')
            c = 1
            if j != 0:
                if j < 60: #plastic bottle = 20
                    sendList.append("20")
                    j = 0
                else: #glass bottle = 21
                    sendList.append("21")
                    j = 0
                k += 1
                #print(sendList[k-1])
    
    hx.power_down()
    hx.power_up()
    time.sleep(1)
    
    if c == 1:
        # 5. receive & send data
        recvData = conn.recv(1024)
        recved = int(recvData.decode())
        if recved == 1:
            print('Received: ', recved)
            conn.sendall(sendList[k-1].encode())
            print('Send: ', sendList[k-1])
        c = 0
    
conn.close()
s.close()
    
