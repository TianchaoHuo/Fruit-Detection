# Overview
---------------------------------------------
This project is designed to do fruit detection, traking and counting. Videos are already provided.

Co-Author: Chaohuo Wu, Xiaochen Liu, Do Kyoun Lee

<!--more-->


# Objectives
---------------------------------------------
 1 . Read in a video clip, and display the video frame by frame,
 2 . Detect all lemons and bananas in each frame, segment the corresponding pixel regions, count the number of pixels, draw a bounding box on these detected fruits.
 3 . Track the detected oranges and bananas in the entire video sequence, until they disappear.
 4 . Count the number of fruits of each type, and update the counting result in real-time.
 5 . At the end of each video clip, report the success rate of your program.



# Video
---------------------------------------------
Open the video named 'Test.mp4'. This is the one of the testing video.  



# Step
---------------------------------------------
1 . DownLoad the project material from google driver. The link is: https://drive.google.com/open?id=1yQXSwYu-GljAxXZLB0YlPoV0lx7KVMPh
2 . Open MATLAB software.
3 . Decompress the 'project material.zip' and copy the 'fruitDetectionTrackingCounting.m' into the folder.
4 . Change the path to the folder.
4 . Load the 'pal2.mat' which contains the Faster RCNN detector.
4 . Input 'global pal2' in command window.
5 . Open and run the 'FruitDetectionTrackingCounting' script.

- important: Before running the script, please make sure you have already input 'global pal2' in command window, otherwise an exception will occur</br>'function 'detect' for uint 8 variables are not defined.
- Note: Since maximum size for submission files on wattle is 200MB, some large but important materials need to be download from goole driver.

# Result
##### Visually Demo


<iframe width="727" height="409" src="" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>





</br></br>
##### Accuracy of this model
<div align=center><img src = "FriutDetectionTrackingandCounting\table.png"></div>



# Summary
---------------------------------------------------------------------------------------------------------------------------------------------

We used Transfer Learning, Alexnet network, Faster-RCNN, Kalman filtert, and others techology to implement this project and have tested all provided videos. The performance is shown in the report 'Fruit Detection, Tracking and Counting_Team 4.pdf'.


# References:

 [1]Motion-Based-Multiple-object-tracking, https://www.mathworks.com/help/vision/examples/motion-based-multiple-object-tracking.html
 [2] A. Krizhevsky, I. Sutskever and G. Hinton, "ImageNet classification with deep convolutional neural networks," Communications of the ACM, vol.60, (6), pp. 84-90, 2017. . DOI: 10.1145/3065386.
