import tensorflow as tf
import cv2
import numpy as np

cls_model = tf.keras.models.load_model("all-in-one.h5",compile=False)

fract_model = tf.keras.models.load_model("fracture.h5",compile=False)

brain_model = tf.keras.models.load_model("brain.h5",compile=False)

chest_model = tf.keras.models.load_model("chest.h5",compile=False)

eye_model = tf.keras.models.load_model("eye.h5",compile=False)

kid_model = tf.keras.models.load_model("kidney.h5",compile=False)

skin_model = tf.keras.models.load_model("skin.h5",compile=False)


def classify(img):
    im = img
    lt = ["other","Bone","Brain","eye","kidney","chest","skin"] 
    im = cv2.resize(im,(52,52))
    result = cls_model.predict(np.array([im]))
    a = np.argmax(result)
    c=""
    if a==0:
        return "Enter the medical Image"
    if a==1:
        c = bone_net(im)
    if a==2:
        c = brain_net(im)
    if a==3:
        c = Eye_net(im)
    if a==4:
        c = kidney_net(im)
    if a==5:
        c = chest_net(im)
    if a==6:
        c = skin_net(im)
    return c


def bone_net(img):
    # img = cv2.resize(img,(224,224))
    lt = ['not fractured', 'fractured']
    
    result = fract_model.predict(np.array([img]))
    # result = model.predict(np.array([img]))
    ans = np.argmax(result)
    return lt[ans]

def brain_net(img):
    lt = ['pituitary', 'notumor', 'meningioma', 'glioma']
    # img = cv2.resize(img,(52,52))
    
    result = brain_model.predict(np.array([img]))
    ans = np.argmax(result)
    return lt[ans]

def chest_net(img):
    lt = ['PNEUMONIA', 'NORMAL']
    # img = cv2.resize(img,(224,224))
    result = chest_model.predict(np.array([img]))
    ans = np.argmax(result)
    return lt[ans]

def Eye_net(img):
    lt = ['glaucoma', 'normal', 'diabetic_retinopathy', 'cataract']
    # img = cv2.resize(img,(224,224))

    result = eye_model.predict(np.array([img]))
    ans = np.argmax(result)
    return lt[ans]

def kidney_net(img):
    lt = ['Cyst', 'Tumor', 'Stone', 'Normal']
    # img = cv2.resize(img,(224,224))
    
    result = kid_model.predict(np.array([img]))
    ans = np.argmax(result)
    return lt[ans]

def skin_net(img):
    lt = ['pigmented benign keratosis', 'melanoma', 'vascular lesion', 'actinic keratosis', 'squamous cell carcinoma', 'basal cell carcinoma', 'seborrheic keratosis', 'dermatofibroma', 'nevus']
    # img = cv2.resize(img,(224,224))
    result = skin_model.predict(np.array([img]))
    ans = np.argmax(result)
    return lt[ans]