<<<<<<< HEAD
# VehicleCounting

## Introduction
The main purpose of this project is to develop an adaptive model that can detect and count the real-time vehicle on urban roads using computer vision techniques.This model is also able to classify the vehicles into six different classes:  Car, Bus, Van, Truck, Motorcycle, and Autorickshaw.

## Dataset
Custom dataset was used for the vehicle counting project. Vehicle counting dataset was collected from the parking of university of engineering and technology Lahore, that was basically based on the recorded video dataset of parking area. The details of the used dataset are following:
- Parking area recorded videos
- Video Length: 2-3 minutes
- Video Quantity: 10-12 clips

## Preprocessing
The preprocessing steps of the proposed project are following:
- Extract Image Frames from Videos
- Annotate the Extracted Image Frame
- Annotation Criteria
  - Car 
  - Bus 
  - Truck 
  - Van 
  - Motor bike 
  
## Model Training
For the vehicle counting in the parking of an institute, Faster RCNN model was trained with the annotated images.
The details of the model training are following:
- Use 2500 Annotated Images

## Results
- Use 300 Annotated Samples for Evaluation
- Calculate Mean Absolute precision (MAp)
- Got 0.8 value of MAP for validation samples


## Requirements
  * Ubuntu 16.04
  * Cuda 9.0
  * Cudnn 7.6.5
  * Tensorflow-gpu==1.8.0
  * opencv-python==4.2.0.34
  * numpy==1.18.5
  
### Pre-Trained Model
 [Downlaod pre-trained models and paste into parent directory.](https://drive.google.com/drive/folders/1pvWsbaCFb_eCnYjeH2ggezmsPJXfwAro?usp=sharing)
 
#### Test
* run python3 FrameLevelVehicleCount.py
* run python3 vehicleCount.py
=======
# Video Stabilization  

## Introduction
The aim of this project is to save video part (frames) in which any kind of object movement is detected otherwise no frame saved. For this purpose, multiple cameras are installed on the different locations. These cameras are installed on 30 to 40 feet above ground level. Due to high location of cameras and air pressure, these are camera movements due to this reason captured videos are not stable and record videos for its own movement instead of objects movement in the video. 

## Dataset
MS COCO dataset was used to detect the cars and persons in this project. MS COCO dataset was collected from the Internet. It is a large-scale image dataset containing 328000 images for every object and humans. 
•	Car
•	person

## Preprocessing
The preprocessing steps of the proposed project are following:
Due to pre trained model there was no preprocessing was involved.

## Model Training 
For the Car and Person detection from a distance, Yolo V 5 model was trained with the MS COCO dataset images (car, person). The details of the model training are following:
•	Use 2500 Annotated Images

## Results
•	Use 300 Annotated Samples for Evaluation
•	Calculate Mean Absolute precision (MAp)
•	Got 0.8 value of MAP for validation samples




>>>>>>> 40b609e2466216a5fd647e1f327342ca61c80a9b
