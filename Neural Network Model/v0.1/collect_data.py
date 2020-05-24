import numpy as np
import cv2
import time
#from directkeys import PressKey, ReleaseKey, W, A, S, D
#import pyautogui
#from statistics import mean
##from numpy import ones,vstack
#from numpy.linalg import lstsq
from getkeys import key_check
import os
from grabscreen import grab_screen

def keys_to_output(keys):
    #Convert pressed keys to a one-hot array
    #[A, W, D] Not worrying about slowing down now
    output = [0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name, allow_pickle=True))
else:
    print('File does not exist, starting fresh')
    training_data = []

def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    last_time = time.time()

    while(True):
        screen = grab_screen(region=(0,40,800,640))
        #Scaling down the image frames to grayscale to reduce size of data
        #One-third size reduction
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (80,60))
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])        
        
        #print('Frame took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)
main()