# Airborne-UAV-Detection
Rival UAV Detection Algorithm for Fighter UAVs, 
add [these](https://drive.google.com/drive/folders/1QBASgyERZHDnktAR0emEWmLIq3ajhELx?usp=sharing) files to your working directory 

Image processing algorithm consists of two parts, object detection
and object tracking respectively. Once the rival UAV is detected and localized with an object
detection algorithm, its output is fed into the object tracking algorithm. Then, the object track-
ing algorithm tracks the rival UAV until N frame passes if the tracking algorithm work in an
confidence interval. Object detection and object tracking algorithms work successively in a way
that they enhance the deficiency of each other as in the pseudo code below:

![pseudo](https://github.com/fcitil/Airborne-UAV-Detection/assets/25532407/362bd3a6-9b48-48cc-bd3e-263c94f05784)

The algorithm can be used with different models, and this dataset is used for the detection model in this project:
<a href="https://universe.roboflow.com/team-rbfpa/airborne-uav-detection">
    <img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>
