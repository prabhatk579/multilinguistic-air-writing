from pyexpat import model
import cv2
import numpy as np
import tensorflow as tf

def recognize(model_name, word_dict, res):
    # Prediction on external image...
    model = tf.keras.models.load_model('models/model_'+model_name+'.h5')
    # model.summary()
    img = cv2.imread('output/last_frame.jpg')
    img_copy = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640,480))

    img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    img_final = cv2.resize(img_thresh, (res,res))
    img_final = np.reshape(img_final, (1,res,res,1))


    img_pred = word_dict[np.argmax(model.predict(img_final))]

    cv2.putText(img, "Your Character: ", (20,25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color = (0,0,230))
    cv2.putText(img, "Prediction: " + img_pred, (20,450), cv2.FONT_HERSHEY_DUPLEX, 1.3, color = (255,0,30))
    cv2.putText(img, "Press q to close this window.", (20,470), cv2.FONT_HERSHEY_DUPLEX, 0.4, color = (0,0,255))
    cv2.imshow('Character recognition', img)

    cv2.waitKey()
    while(1):
        k = input()
        if k == 'z':
            key = '1'
            break
        elif k == 'q':
            key = '0'
            break
    
    cv2.destroyAllWindows()
    return key