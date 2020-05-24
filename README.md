# GTA_Self_Driving_Car_RC
The objective here is to create a self driving car in the world of Grand Theft Auto 5.

Thanks to Harrison Kinsley's (@sentdex) YouTube channel for the amazing tutorials. The reference GitHub repository is https://github.com/sentdex/pygta5

My approach consists of two models - A lane detection model and a neural network based model

The primary focus of development for now is the Neural Network model.

**v0.2** - Nine class labels (w, s, a, d, wa, wd, sa, sd, no key). Here, the Convolutional Neural Network (CNN) that is being used to train the neural network is an expanded version of AlexNet called alexnetv2. It is fed with grayscale input frames of size 160x120.

**v0.1** - Three class labels (w, a, d). Here, a standard AlexNet is used and it is given grayscale input images of size 80x60.

**General instructions:**

