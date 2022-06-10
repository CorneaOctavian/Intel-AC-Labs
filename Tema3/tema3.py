
import cv2
import os


vid = cv2.VideoCapture("video1.mp4")

try:

    
    if not os.path.exists('data'):
        os.makedirs('data')


except OSError:
    print('Folderul nu s-a creat')


currentframe = 0

while (True):

    
    success, frame = vid.read()

    if success:
        
        name = './data/frame' + str(currentframe) + '.jpg'
        print('Se extrage' + name)

       
        cv2.imwrite(name, frame)

        
        
        currentframe += 1
    else:
        break


vid.release()
cv2.destroyAllWindows()