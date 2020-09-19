import numpy as np
import cv2
import tensorflow as tf
m_new=tf.keras.models.load_model('model_digits.h5')


img = np.ones([400,400],dtype ='uint8')*255

img[50:350,50:350]=0
wname = 'Canvas'
cv2.namedWindow(wname)
state= False
def shape(event,x,y,flags,param):
    global state
    if event == cv2.EVENT_LBUTTONDOWN:
        state=True
        cv2.circle(img,(x,y),10,(255,0,0),-1)
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if(state==True):
           cv2.circle(img,(x,y),10,(255,0,0),-1)
    else:
           state=False

cv2.setMouseCallback(wname,shape)

while True:
    
    cv2.imshow(wname,img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('p'):
        out = img[50:350,50:350]
        cv2.imwrite('Output.jpg',out)
        image_test_resize=cv2.resize(out,(28,28)).reshape(1,28,28)
        result=np.argmax(m_new.predict(image_test_resize), axis=-1)
        print(result)
        
cv2.destroyAllWindows()




