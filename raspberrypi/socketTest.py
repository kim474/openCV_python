import socket
import sys
import time

EMULATE_HX711=False

referenceUnit = -447
valList = []
sendList = []
i = 0
j = 0
k = 0
c = 0
send = ""

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

HOST = '192.168.35.143'  # all available interfaces
PORT = 8888

# 1. open Socket
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

# keep talking with the client
while 1:
    # 4. Accept connection
    #conn, addr = s.accept()
    #print('Connected with ' + addr[0] + ':' + str(addr[1]))

    # 5. receive data
    #data = conn.recv(1024)
    #if not data:
    #    break
    #else:
    #    print('Received: ', data.decode())
        
    # 6. send data
    try:
        val = max(0, int(hx.get_weight(5)))
        valList.append(val)
        if i == 0:
            #print(valList[i])
            j = valList[i]
            send = "start"
            conn.sendall(send.encode())
                #if valList[i] != 0:
                #    if valList[i] < 100: #plastic bottle = 1
                #        send = "1"
                        #data = bytes(send, 'utf-8')
                #        conn.sendall(send.encode())
                #    else: #glass bottle = 2
                #        send = "2"
                        #data = bytes(send, 'utf-8')
                #        conn.sendall(send.encode())
            print('Send: ', send)
                #print('sendList[k] is: ', sendList[k])
            k += 1
            j = 0
            i += 1
        else:
            if valList[i-1] != valList[i]:
                j = valList[i]
            i += 1
        if j != 0:
            if valList[i-2] == j:
                print(j, 'gram')
                if j != 0:
                    if j < 100: #plastic bottle = 1
                        send = "1"
                            #data = bytes(send, 'utf-8')
                        conn.sendall(send.encode())
                    else: #glass bottle = 2
                        send = "2"
                            #data = bytes(send, 'utf-8')
                        conn.sendall(send.encode())
                print('Send: ', send)
                    #print('sendList[k] is: ', sendList[k])
                k += 1
                j = 0

        hx.power_down()
        hx.power_up()
        time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        cleanAndExit()
    
    #data = conn.recv(1024)
    #print('Received: ', data.decode())
    
    #conn.sendall(data)
    #print('Send: ', data.decode())

conn.close()
s.close()
