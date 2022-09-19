# Using resources from

https://github.com/nicknochnack/TFODCourse

https://github.com/open-mmlab/mmdetection

https://github.com/yehengchen/Object-Detection-and-Tracking

https://github.com/neuralmagic/sparsezoo

https://github.com/VishalBheda/ObjectRecogniserWithCOCO


# Training and test data

Images (both train and test) were taken from

https://wavesinthekitchen.com/rainbow-apples/

https://www.seriouseats.com/the-food-lab-what-are-the-best-apples-for-apple-pies-how-to-make-pie

https://jooinn.com/colorful-apples-2.html

Test images are cropped train images.

In real-world scenarios the data bank should be significant.

Tensorflow and Tensorflow-GPU are used for the training and testing.

Code tested only on a Windows machine, setup is significantly different for linux machines. Uploaded also venv because of this.


# Annotating the data

To create the labels (detection boxes) for the train and test datasets, labelImg was used.

This tools generates the xml files with the boxes.


# Protoc

Protocol Buffers is installed for data serialization.


# Models

Pre-trained models used taken from publically available https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

In use now: SSD MobileNet V2 FPNLite 320x320


# Entry points

main.py - perform the training

main_test_model.py - performs the test on a single given image
