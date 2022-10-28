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

