import cv2
import argparse #command line argument

#detect faces in image

def highlightFace(net, frame, conf_threshold=0.7):    #getting location of face in the image
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    #removes all the variables or extra info(brightness,saturation,etc.)brings the image to its bare minimum, same property level as the image was in the database
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (227, 227), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()  #detect the no. of faces
    faceBoxes=[]
    for i in range(detections.shape[2]):  #for drawing rectangles on detected faces
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

# command line argument
parser=argparse.ArgumentParser()
parser.add_argument('--image') #adding argument to give i/p image to program

args=parser.parse_args()


#assign the model and weights
faceProto="opencv_face_detector.pbtxt"    #face detection
faceModel="opencv_face_detector_uint8.pb"


ageProto="age_deploy.prototxt"  #age detection and prediction
ageModel="age_net.caffemodel"


genderProto="gender_deploy.prototxt"   #gender detection
genderModel="gender_net.caffemodel"

#define the parameters
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(17-22)', '(25-32)', '(37-43)', '(48-53)', '(60-100)']
genderList=['Male', 'Female']


#load the model
faceNet=cv2.dnn.readNet(faceModel,faceProto) #directly use the pre trained model of deeplearning in openCV
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

# open a video file or an image file or camera stream
video=cv2.VideoCapture(args.image if args.image else 0)

padding=20

while cv2.waitKey(1)<0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg,faceBoxes=highlightFace(faceNet,frame)  #detect no. of faces in the frame

    for faceBox in faceBoxes:   #do the prediction for the no. of face detected
        face = frame[max(0,faceBox[1]-padding): #face info is stored in face variable
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()   #detect the gender of each face
        gender=genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)  #detect the age of the person
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]

        # to put the label/text i.e gender and age in the frame
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)