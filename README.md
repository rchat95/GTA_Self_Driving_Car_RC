# GTA_Self_Driving_Car_RC
The objective here is to create a self driving car in the world of Grand Theft Auto 5.

Thanks to Harrison Kinsley's (@sentdex) YouTube channel for the amazing tutorials. The reference GitHub repository is https://github.com/sentdex/pygta5

My approach consists of two models - A lane detection model and a neural network based model

The primary focus of development for now is the Neural Network model.

**v0.2** - Nine class labels (w, s, a, d, wa, wd, sa, sd, no key). Here, the Convolutional Neural Network (CNN) that is being used to train the neural network is an expanded version of AlexNet. It is fed with grayscale input frames of size 160x120.

**v0.1** - Three class labels (w, a, d). Here, a standard AlexNet is used and it is given grayscale input images of size 80x60.

**General instructions:** Change the settings of GTAV's screen resolution to 800x600 as using grab_screen.py we are capturing a 800x600 area from the screen's top left corner. Get in a car and switch to first person view (optional) and run collect_data.py. This will begin creation of the training data. Optionally, you can use balance_data.py to balance the training data as there are a lot more "w" key presses as compared to the other key presses. In v0.1, I have used the balancing algorithm, whereas in v0.2, I skipped it and the car still performed quite well. Once you have enough frames (atleast 80K frames) run the train_model.py code. The model training will start and you can see realtime stats using tensorboard. The model can be tested using test_model.py. Now your computer knows how to drive a car!

The video attached shows a small glimpse of the trained nueral network driving a car.

