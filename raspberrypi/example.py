import time
import sys

EMULATE_HX711=False

referenceUnit = -447
valList = []
sendList = []
i = 0
j = 0
k = 0
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

while True:
    try:
        val = max(0, int(hx.get_weight(5)))
        valList.append(val)
        if i == 0:
            print(valList[i])
            j = valList[i]
            if valList[i] != 0:
                if valList[i] < 100: #plastic bottle = 1
                    send = "1"
                else: #glass bottle = 2
                    send = "2"
            sendList.append(send)
            print('sendList[k] is: ', sendList[k])
            k += 1
            j = 0
            i += 1
        else:
            if valList[i-1] != valList[i]:
                j = valList[i]
            i += 1
        if j != 0:
            if valList[i-2] == j:
                print(i-2, ':', j)
                if j != 0:
                    if j < 100: #plastic bottle = 1
                        send = "1"
                    else: #glass bottle = 2
                        send = "2"
                sendList.append(send)
                print('sendList[k] is: ', sendList[k])
                k += 1
                j = 0

        hx.power_down()
        hx.power_up()
        time.sleep(1)

    except (KeyboardInterrupt, SystemExit):
        cleanAndExit()

