# TensorFLow-Object_detection-Images-and-Videos-Windows10

## About
These are a series of articles that cover the entirety of Object-Detection using Tensorflow for Images and Videos in a Windows device. Starting from scratch its a journey to your own custom dataset you need to train and run inference on, in real-time.

## Content
* [Setting up TensorFlow 1.14 in bare Windows](https://medium.com/@deep12vish/setting-up-tensorflow-1-14-in-bare-windows-adc429ab792c)
> This is the 1st step towards setting up your windows 10 machine, [TensorFlow](https://www.tensorflow.org/) provides us with the necessary tools to build and deploy the model for real-world applications. We will assume the Machine is equipped with NVIDIA GPU, and if not simply skip CUDA (drivers) installation and continue along with the instructions.

* * Folder Contents: __get-pip.py__ , __req_pip.txt__ , __medium_nyc_street_ssd_5sec.mp4__

* [Practical aspects to select a Model for Object Detection](https://medium.com/@deep12vish/practical-aspects-to-select-a-model-for-object-detection-c704055ab325)
> Selecting the MODEL-Architechture for optimal performance depends on various factors all of which are not inclusive to MODEL properties. This article discusses the Practical aspects of MODEL-Selection, the real-life problems you may never see coming until its quite late in the project. It's better to be aware and prepare first hand.
* * Folder Contents: __tracking_hop_space_x1.mp4__

* [TensorFlow Object Detection in Windows (under 30 lines)](https://medium.com/@deep12vish/tensorflow-object-detection-in-windows-under-30-lines-d6776586c4ab)
> After setting up the environment , this is the part where you will run the inference by loading the model and image of your choice. In just 30 lines of code you will get all the fundamentals required to run inference successfully. 
* * Folder Contents: __ssd_mobilenet_v2_coco__ (__frozen_inference_graph.pb__), __object_detection_under_images_30_llines.py__ , __mscoco_label_map.pbtxt__ , __Euro_truck_sim.jpg__ , __Euro_truck_sim_result.jpg__ , __Times_sq1.jpg__ , __Times_sq1_result.jpg__

## Requirements
* NVIDIA GPU
* About 20 GB of HDD Space
* Patience and a lot of it.

## Inspiration
* [SENTDEX](https://pythonprogramming.net/)
