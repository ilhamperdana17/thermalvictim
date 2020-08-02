# import packages
import imutils
import numpy as np
import argparse
import cv2

# LOAD MODEL
print("[INFO] loading model...")


    # Mobile net SSD Model from Caffe
prtxt = 'example/MobileNetSSD_deploy.prototxt'
model = 'snapshot/mobilenet_iter_40000.caffemodel' 
    
    # initialize the list of class labels MobileNet SSD was trained to
    # detect with the caffe model.
CLASSES = ["background", "person"]
    
    # Load the Model
net = cv2.dnn.readNetFromCaffe(prtxt, model)
image = "google_0104.jpg"
# Randomize some colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES)+5, 3))

# Confidence threshold
conf = .21
 
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] Reading image...")

# Read image
frame = cv2.imread(image)
#frame = imutils.resize(frame, width=800)


# grab the frame dimensions and convert it to a blob
(h, w) = frame.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), (127.5, 127.5, 127.5))

 
# pass the blob through the network and obtain the detections and
# predictions
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in np.arange(0, detections.shape[2]):
	
    # extract the confidence (i.e., probability) associated with
	# the prediction
    confidence = detections[0, 0, i, 2]
 
	# filter out weak detections by ensuring confidence
	# greater than the minimum confidence
    if confidence > conf: #args["confidence"]:
	
	    # extract the index of the class label from the
	    # `detections`, then compute the (x, y)-coordinates of
        	# the bounding box for the object
        idx = int(detections[0, 0, i, 1])
	if idx ==1 :
        	box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        	(startX, startY, endX, endY) = box.astype("int")
 		print CLASSES[idx],confidence,startX,startY,endX,endY
	   # draw the prediction on the frame
       # print('idx: ',idx)
        	label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
        	cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0,0,255), 1)
        	y = startY - 15 if startY - 15 > 15 else startY + 15
        	cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

# show the output frame



cv2.imshow("Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save image
cv2.imwrite('output/detectImage-output.png', frame)
