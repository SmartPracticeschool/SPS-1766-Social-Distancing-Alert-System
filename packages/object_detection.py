    # -*- coding: utf-8 -*-


from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np
import cv2

# Take Frames from Social Distancing
def detect_people(frames, net, ln, personIdx = 0):
    #For Scaling
    (H, W) = frames.shape[:2]
    results = []
    
    # Construct a blob from the input frame and then perform 
    # pass of the YOLO object detector, giving us our bounding
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frames, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    
    # initialize our lists of detected bounding boxes, centroids 
    # confidences respectively
    boxes = []
    centroids = []
    confidences = []
    
    # For each Output Layers
    for output in layerOutputs:
        # For the frame which is inside Outer Layer
        for detection in output:
            # in YOLO there are 8 ouputs
            # confidence, x, y, h, w, pr(1), pr(2), pr(3)
            # From 5th Score starts
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == personIdx and confidence > MIN_CONF:
                box = detection[0:4] * np.array([W,H,W,H])
                (centerX, centerY, width, height)=box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
                
    # apply Non-Maxima Suppression to suppress weak, overlapping
    # bounding boxes
                
    idxs = cv2.dnn.NMSBoxes(boxes,confidences,MIN_CONF,NMS_THRESH)
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            #extract the bounding box coordinates
            (x,y) = (boxes[i][0],boxes[i][1])
            (w,h) = (boxes[i][2],boxes[i][3])
            
            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            
            r = (confidences[i], (x ,y, x + w, y + h), centroids[i] )
            results.append(r)
    
    # return the list of results
    return results

            
            
 
#Initialize our lists of deteected bounding boxes, centroid, 
#confidences, respectively       
    
#Take Frames from SD

#Pre Prosess 

#Frames, Give Back to model

#Get Outputs fronm model    

#compared - Only Persons returned    
#Nom Maxima Suppression    

# Centroid, BBox Cord, Confidence

    





