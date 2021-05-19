import socket
import struct
import traceback
import logging
import time
import numpy as np
import main as M

def sending_and_reciveing():
    s = socket.socket()
    socket.setdefaulttimeout(None)
    print('socket created ')
    port = 8000
    s.bind(('192.168.35.57', port))
    s.listen(30) #listening for connection for 30 sec?
    print('socket listensing ... ')
    while True:
        try:
            c, addr = s.accept() #when port connected
            bytes_received = c.recv(4000) #received bytes
            array_received = np.frombuffer(bytes_received, dtype=np.float32) #converting into float array

            nn_output = str(M.yoloClass_id(array_received)) #NN prediction (e.g. model.predict())

            bytes_to_send = struct.pack('%sf' % len(nn_output), *nn_output) #converting float to byte
            c.sendall(bytes_to_send) #sending back
            c.close()
        except Exception as e:
            logging.error(traceback.format_exc())
            print("error")
            c.sendall(bytearray([]))
            c.close()
            break

sending_and_reciveing()