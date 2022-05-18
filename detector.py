import numpy as np
import cv2 
from collections import deque 
from color_selector import color_detector
import tensorflow as tf


def detect():
    # color_arr()

    bpoints = [deque(maxlen = 512)] 
    gpoints = [deque(maxlen = 512)] 
    ypoints = [deque(maxlen = 512)] 
    rpoints = [deque(maxlen = 512)]

    # Now to mark the pointers in the above colour array we introduce some index values Which would mark their positions  

    blue_index = 0
    green_index = 0
    yellow_index = 0
    red_index = 0

    # The kernel is used for dilation of contour

    kernel = np.ones((5, 5))

    # The ink colours for the drawing purpose 
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 225, 255), (0, 0, 255)] 
    colorIndex = 0
    # Setting up the drawing board AKA The canvas 

    paintWindow = np.zeros((471, 636, 3)) + 255

    cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE) 

    cap = cv2.VideoCapture(0) 
    # calling the colour function 
    color_detector()

    while True: 

        # Reading the camera frame 
        ret, frame = cap.read() 
        # For saving video output
        # out = cv2.VideoWriter("Paint-Window.mp4", cv2.VideoWriter_fourcc(*'XVID'), 1, (frame.shape[1], frame.shape[0]))
        
        # Flipping the frame to see same side of the user  
        frame = cv2.flip(frame, 1) 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

        # Getting the new positions of the trackbar and setting the new HSV values 

        u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors") 
        u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors") 
        u_value = cv2.getTrackbarPos("Upper Value","Color detectors") 
        l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors") 
        l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors") 
        l_value = cv2.getTrackbarPos("Lower Value", "Color detectors") 
        Upper_hsv = np.array([u_hue, u_saturation, u_value]) 
        Lower_hsv = np.array([l_hue, l_saturation, l_value]) 

        # Adding the colour buttons to the live frame to choose color
        frame = cv2.rectangle(frame, (35, 1), (135, 65), (122, 122, 122), -1) 
        frame = cv2.rectangle(frame, (160, 1), (255, 65), (255, 0, 0), -1) 
        frame = cv2.rectangle(frame, (275, 1), (370, 65), (0, 255, 0), -1) 
        frame = cv2.rectangle(frame, (390, 1), (485, 65), (0, 255, 255), -1) 
        frame = cv2.rectangle(frame, (505, 1), (600, 65), (0, 0, 255), -1) 

        cv2.putText(frame, "Clear All", (38, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA) 

        cv2.putText(frame, "Blue", (185, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA) 
        
        cv2.putText(frame, "Green", (290, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA) 
        
        cv2.putText(frame, "Yellow", (405, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA) 

        cv2.putText(frame, "Red", (535, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, "Press the appropriate:", (20, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "'1' for English", (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "'2' for Roman digits", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  
        cv2.putText(frame, "'3' for Hindi", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "'4' for Hindi No.", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, "hold 'f' to recognize", (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "hold 'q' to terminate this window", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 

        # masking out the pointer for it's identification in the frame 

        Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv) 
        Mask = cv2.erode(Mask, kernel, iterations = 1) 
        Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel) 
        Mask = cv2.dilate(Mask, kernel, iterations = 1) 

        # Now contouring the pointers post identification 
        
        countours, _ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        centre = None

        # If there are any contours formed 
        if len(countours) > 0: 
            
            # sorting the contours for the biggest 
            countour = sorted(countours, key = cv2.contourArea, reverse = True)[0] 
            # Get the radius of the cirlce formed around the found contour   
            ((x, y), radius) = cv2.minEnclosingCircle(countour) 
            
            # Drawing the circle boundary around the contour 
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2) 
            
            # Calculating the centre of the detected contour 
            M = cv2.moments(countour) 
            centre = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) 
            
            # Now checking if the user clicked on another button on the screen (the 4 buttons that were mentioned Y,G,B,R and clear all)
            if centre[1] <= 65: 
                
                # Clear Button 
                if 35 <= centre[0] <= 135: 
                    bpoints = [deque(maxlen = 512)] 
                    gpoints = [deque(maxlen = 512)] 
                    ypoints = [deque(maxlen = 512)] 
                    rpoints = [deque(maxlen = 512)] 

                    blue_index = 0
                    green_index = 0
                    yellow_index = 0
                    red_index = 0

                    paintWindow[67:, :, :] = 255
                elif 160 <= centre[0] and centre[0] <= 255: 
                    colorIndex = 0 # Blue 
                        
                elif 275 <= centre[0] and centre[0] <= 370: 
                    colorIndex = 1 # Green 
                elif 390 <= centre[0] and centre[0] <= 485: 
                    colorIndex = 2 # Yellow
                elif 505 <= centre[0] and centre[0] <= 600: 
                    colorIndex = 3 # Red 
            else : 
                if colorIndex == 0: 
                    bpoints[blue_index].appendleft(centre) 
                elif colorIndex == 1: 
                    gpoints[green_index].appendleft(centre) 
                elif colorIndex == 2: 
                    ypoints[yellow_index].appendleft(centre) 
                elif colorIndex == 3: 
                    rpoints[red_index].appendleft(centre) 
                    
        # Appending the next deques if nothing is detected

        else: 
            bpoints.append(deque(maxlen = 512)) 
            blue_index += 1
            gpoints.append(deque(maxlen = 512)) 
            green_index += 1
            ypoints.append(deque(maxlen = 512)) 
            yellow_index += 1
            rpoints.append(deque(maxlen = 512)) 
            red_index += 1

        # Drawing the lines of every colour on the canvas and the track frame window
        
        points = [bpoints, gpoints, ypoints, rpoints] 
        for i in range(len(points)): 
            for j in range(len(points[i])): 
                for k in range(1, len(points[i][j])): 
                    if points[i][j][k - 1] is None or points[i][j][k] is None: 
                        continue
                        
                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 25) 
                    cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 25) 

        key = cv2.waitKey(1)            
        if key & 0xFF == ord('1'):            
            cv2.imwrite("output/last_frame.jpg", paintWindow)
            model_name = 'eng_alphabets'
            word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
            res = 28
        elif key & 0xFF == ord('2'):            
            cv2.imwrite("output/last_frame.jpg", paintWindow)
            model_name = 'roman_digits'
            word_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}
            res = 28
        elif key & 0xFF == ord('3'):            
            cv2.imwrite("output/last_frame.jpg", paintWindow)
            model_name = 'hindi_alphabets'
            word_dict = {0: 'CHECK', 1: 'ka', 2: 'kha', 3: 'ga', 4: 'gha', 5: 'kna', 6: 'cha',
                    7: 'chha', 8: 'ja', 9: 'jha', 10: 'yna', 11: 'taa', 12: 'thaa', 13: 'daa', 
                    14: 'dhaa', 15: 'adna', 16: 'ta', 17: 'tha', 18: 'da', 19: 'dha', 20: 'na', 
                    21: 'pa', 22: 'pha', 23: 'ba', 24: 'bha', 25: 'ma', 26: 'yaw', 27: 'ra', 
                    28: 'la', 29: 'waw', 30: 'sha', 31: 'sha',32: 'sa', 33: 'ha',
                    34: 'kshya', 35: 'tra', 36: 'gya', 37: 'CHECK'}
            # word_dict = {0: 'ka', 1: 'kha', 2: 'ga', 3: 'gha', 4: 'kna', 5: 'cha',
            #         6: 'chha', 7: 'ja', 8: 'jha', 9: 'yna', 10: 'taa', 11: 'thaa', 12: 'daa', 
            #         13: 'dhaa', 14: 'adna', 15: 'ta', 16: 'tha', 17: 'da', 18: 'dha', 19: 'na', 
            #         20: 'pa', 21: 'pha', 22: 'ba', 23: 'bha', 24: 'ma', 25: 'yaw', 26: 'ra', 
            #         27: 'la', 28: 'waw', 39: 'sha', 30: 'sha',31: 'sa', 32: 'ha',
            #         33: 'kshya', 34: 'tra',35: 'sra', 36: 'gya'}
            res = 32
        elif key & 0xFF == ord('4'):            
            cv2.imwrite("output/last_frame.jpg", paintWindow)
            model_name = 'devnagri_digits'
            word_dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9'}
            res = 32
        
            

        # Displaying/running all the 3 windows 
        cv2.imshow("Live Tracking", frame) 
        cv2.imshow("Paint", paintWindow) 
        cv2.imshow("mask", Mask) 
        
        # For quitting/breaking the loop - press and hold ctrl+q twice 
        if cv2.waitKey(1) & 0xFF == ord("f"):
            key = '1'
            break
        elif cv2.waitKey(1) & 0xFF == ord("q"):
            key = '0'
            model_name = ''
            word_dict = {}
            res = 0
            break

    # Releasing the camera and all the other resources of the device  
    cap.release()
    cv2.destroyAllWindows()
    return model_name ,word_dict, res, key