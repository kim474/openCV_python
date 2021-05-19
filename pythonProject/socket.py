import UdpComms as U
import time
import main as M

# Create UDP socket to use for sending (and receiving)
sock = U.UdpComms(udpIP="192.168.35.57", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)

while True:
    sock.SendData('Sent from Python: ' + str(M.yoloClass_id)) # Send this string to other application
    #sock.SendData('Sent from Python: ' + str(M.cvClass_id))
    data = sock.ReadReceivedData() # read data

    if data != None: # if NEW data has been received since last ReadReceivedData function call
        print(data) # print new received data

    time.sleep(1)
