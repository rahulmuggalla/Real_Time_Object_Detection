#! pip install imageAI
#! pip install opencv-python
#! pip install Pillow
#! pip install numpy

#importing the object detection module from imageAI library
from imageai.Detection import ObjectDetection 

#importing the OpenCV library to perform image processing operations on the image or video file 
import cv2  

obj_detect = ObjectDetection() #creating an object of the ObjectDetection class from the imageAI library to perform object detection on the image or video file using the pre-trained model of the YOLOv3 algorithm 

obj_detect.setModelTypeAsYOLOv3() #setting the model type as YOLOv3 algorithm to perform object detection on the image or video file using the pre-trained model of the YOLOv3 algorithm 

obj_detect.setModelPath(r"yolo.h5") #setting the path of the pre-trained model of the YOLOv3 algorithm to perform object detection on the image or video file using the pre-trained model of the YOLOv3 algorithm 
obj_detect.loadModel() #loading the pre-trained model of the YOLOv3 algorithm to perform object detection on the image or video file using the pre-trained model of the YOLOv3 algorithm

cam_feed = cv2.VideoCapture(0) #creating an object of the VideoCapture class from the OpenCV library to capture the live video feed from the webcam of the system 

cam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 650) #setting the width of the live video feed from the webcam of the system 
cam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 750) #setting the height of the live video feed from the webcam of the system

#creating an infinite loop to capture the live video feed from the webcam of the system 
while True:    
    ret, img = cam_feed.read() #reading the live video feed from the webcam of the system 

    #setting the input & output type as array to perform object detection on the image or video file using the pre-trained model of the YOLOv3 algorithm
    annotated_image, preds = obj_detect.detectObjectsFromImage(input_image=img, input_type="array", output_type="array", display_percentage_probability=True, display_object_name=True) 

    #displaying the annotated image on the screen 
    cv2.imshow("", annotated_image)     
    
    #creating an if condition to break the infinite loop when the user presses the 'q' or 'Esc' key on the keyboard
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):  
        break

cam_feed.release() #releasing the live video feed from the webcam of the system 
cv2.destroyAllWindows() #destroying all the windows created by the OpenCV library