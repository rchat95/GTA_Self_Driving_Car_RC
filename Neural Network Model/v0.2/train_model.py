import numpy as np
from alexnet import alexnet2
from random import shuffle
import pandas as pd
import cv2

# what to start at
START_NUMBER = 1

# what to end at
hm_data = 10

# use a previous model to begin?
START_FRESH = True

WIDTH = 160
HEIGHT = 120
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'pygta5-car-balanced-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2_10', EPOCHS)
EXISTING_MODEL_NAME = ''


model = alexnet2(WIDTH, HEIGHT, LR, output=9)

if not START_FRESH:
    model.load(EXISTING_MODEL_NAME)


data_order = [i for i in range(START_NUMBER,hm_data+1)]
shuffle(data_order)
for i in data_order:
    train_data = np.load('balanced_training_data-{}.npy'.format(i+10), allow_pickle=True)
    print('Training from file balanced_training_data-{}.npy'.format(i))
    df = pd.DataFrame(train_data)
    df = df.iloc[np.random.permutation(len(df))]
    train_data = df.values.tolist()

    train = train_data[:-100]
    test = train_data[-100:]

    X = np.array([i[0] for i in train]).reshape(-1,WIDTH,HEIGHT,1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,WIDTH,HEIGHT,1)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)
    
    #tensorboard --logdir=D:/Coding/GTAAI_New/log

    model.save(MODEL_NAME)